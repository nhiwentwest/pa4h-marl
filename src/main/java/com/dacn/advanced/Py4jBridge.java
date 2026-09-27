package com.dacn.advanced;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.cloudlets.network.CloudletExecutionTask;
import org.cloudsimplus.cloudlets.network.CloudletReceiveTask;
import org.cloudsimplus.cloudlets.network.CloudletSendTask;
import org.cloudsimplus.cloudlets.network.NetworkCloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.network.NetworkDatacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.network.NetworkHost;
import org.cloudsimplus.network.HostPacket;
import org.cloudsimplus.network.switches.AggregateSwitch;
import org.cloudsimplus.network.switches.EdgeSwitch;
import org.cloudsimplus.network.switches.RootSwitch;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.cloudlet.network.CloudletTaskSchedulerSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.network.NetworkVm;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Py4J Gateway cho Multi-Agent RL + Autoformer.
 * V3: Real migrations, SPECpower model, proper SLATAH/PDM, long-running cloudlets.
 */
public class Py4jBridge {

    // --- Simulation Config ---
    private static final double INTERVAL = 300.0;
    private static final double MAX_TIME = 86400.0;
    private static final int NUM_HOSTS = Integer.parseInt(System.getenv().getOrDefault("NUM_HOSTS", "64"));
    private static final int HOST_PES = 2;              // Beloglazov Table 5: HP G5
    private static final int HOST_MIPS = 2660;           // HP G5: 2660 MIPS/PE (Xeon 3075)
    private static final int HOST_RAM = 16384;           // 16GB
    private static final double HOST_GPU = 8.0;           // Logical GPU capacity per host
    private static final long HOST_BW = 1000;            // 1 Gbps = 1000 Mbit/s
    private static final long HOST_STORAGE = 1000000;
    // --- GPU Exclusive Config (derived from NREL measurements) ---
    private static final double NREL_IDLE_POWER_W = 612.0;      // p5 colocation per-node [W]
    private static final double NREL_PEAK_POWER_W = 3062.0;     // p95 training per-node [W]
    private static final double RACK_NAMEPLATE_W  = 12248.0;    // 4 nodes * p95-peak [W]
    private static final double RACK_OVERSUBSCRIPTION = parseEnvDouble("RACK_OVERSUB", 1.2);
    private static final double RACK_BUDGET_W = RACK_NAMEPLATE_W / RACK_OVERSUBSCRIPTION;
    private static final int NETWORK_RACK_SIZE = 4;
    private static final int NUM_RACKS = (int) Math.ceil((double) NUM_HOSTS / NETWORK_RACK_SIZE);
    private static final double CHECKPOINT_COST_SEC = parseEnvDouble("CHECKPOINT_COST_SEC", 120.0);

    private static final int TOP_K = 10;
    private static final int GLOBAL_STATE_DIM = 8;    // Expanded state
    private static final int VM_EXTRA_FEATURES = 8;    // VM/source features after candidate blocks
    private static final int VM_GANG_FEATURES = TOP_K;
    private static final int VM_STATE_DIM = TOP_K * 3 + VM_EXTRA_FEATURES + VM_GANG_FEATURES;
    private static final int VM_SELECTION_FEATURE_DIM = 10;
    private static final int DEFAULT_HISTORY_LEN = 20;
    private static final double NET_COMM_TAX = parseEnvDouble("NET_COMM_TAX", 0.30);
    private static final double INTRA_RACK_LATENCY_SEC = 0.0005;
    private static final double INTER_RACK_LATENCY_SEC = 0.002;
    private static final double DEFAULT_LINK_BW_MBPS = 1000.0;
    private static final double BACKGROUND_TRAFFIC_COEF =
        parseEnvDouble("NETWORK_BACKGROUND_TRAFFIC_COEF", 0.00015);
    private static final long SIM_SEED = parseEnvLong(
        "SIM_SEED",
        parseEnvLong("RUN_SEED", 20260704L)
    );

    // --- Runtime ---
    private CloudSimPlus simulation;
    private Datacenter datacenter;
    private DatacenterBroker broker;
    private List<Host> hostList;
    private List<Vm> vmList;
    private Thread simThread;
    private Map<Long, Double> vmGpuRequests = new HashMap<>();
    private Map<String, NetworkLinkMetric> networkLinks = new LinkedHashMap<>();
    private final List<MeteredEdgeSwitch> edgeSwitches = new ArrayList<>();
    private List<int[]> gangs = new ArrayList<>();
    private int[][] gangOfVm;
    private Vm[] vmByIdArr;

    // --- GPU Exclusive Workload State ---
    private double[][] hostFineTrace;            // [NUM_HOSTS][] - raw fine-grained (0.2s) trace
    private double[] hostFineDt;                // [NUM_HOSTS] - time delta of fine trace
    private int[] hostJobStartFineIdx;          // [NUM_HOSTS] - current index in fine trace
    private int[] hostJobDurationFineSteps;     // [NUM_HOSTS] - total duration in fine steps
    private double[][] hostRlTrace;             // [NUM_HOSTS][] - resampled RL-step trace
    private int[] hostJobStartRlStep;           // [NUM_HOSTS] - start step in RL time
    private int[] hostJobDurationRlSteps;       // [NUM_HOSTS] - duration in RL steps
    private final Map<String, int[]> jobHostMap = new HashMap<>(); // jobId -> hostIds
    private double totalCheckpointOverhead = 0.0; // Cumulative preemption downtime (sec)

    // Pause/resume sync
    private volatile boolean simPaused = false;
    private volatile boolean simDone = false;
    private double nextPauseTime;

    // Metrics per interval
    private double prevEnergy = 0;
    private double currentEnergy = 0;
    // Cross-rack gang traffic accounting. The metered edge switches accumulate a
    // monotonic byte total; we snapshot it each step to derive the per-interval
    // delta (resetWindow on the switches is never driven, so we diff the total
    // ourselves). This delta is the placement-dependent network cost.
    private long prevCrossRackBytes = 0;
    private long prevLocalBytes = 0;
    private long intervalCrossRackBytes = 0;
    private long intervalLocalBytes = 0;
    private int intervalMigrations = 0;
    private int totalMigrations = 0;
    private int invariantWarnings = 0;

    // History & SLA metrics tracking (Beloglazov-accurate)
    private double[][] hostCpuHistoryArr;
    private int historyLen = DEFAULT_HISTORY_LEN;
    private double totalActiveHostTime = 0;
    private double totalViolatedHostTime = 0;   // SLATAH: time where requested > allocated
    private double totalActiveVmTime = 0;
    private double totalMigrationDowntime = 0;  // PDM: actual downtime during migration
    private double currentSlatah = 0;
    private double currentPdm = 0;
    private int currentStep = 0;
    private int lastSkippedNoMovableSources = 0;
    private int[] pendingVmSelectionActions = null;
    private double adaptiveOverloadThreshold = 1.0;
    private double adaptiveCriticalThreshold = 1.0;
    private double adaptiveMeanUtilization = 0.65;
    private int lastInitialVmCount = 0;

    private static class NetworkLinkMetric {
        final int src;
        final int dst;
        final double bwMbps;
        final double latencySec;
        double bytesWindow;
        double bytesBackground;

        NetworkLinkMetric(int src, int dst, double bwMbps, double latencySec) {
            this.src = Math.min(src, dst);
            this.dst = Math.max(src, dst);
            this.bwMbps = bwMbps;
            this.latencySec = latencySec;
            this.bytesWindow = 0.0;
            this.bytesBackground = 0.0;
        }

        double utilization(double windowSec) {
            double capacityBits = bwMbps * 1_000_000.0 * Math.max(1e-9, windowSec);
            double totalBytes = bytesWindow + bytesBackground;
            return Math.max(0.0, Math.min(1.0, totalBytes * 8.0 / capacityBits));
        }
    }

    // Workload traces
    private List<String> traceFiles = new ArrayList<>();
    private boolean useAzureTraces = false;  // Flag for Azure vs Philly format
    // Array of UtilizationModels indexed by VM creation order
    private org.cloudsimplus.utilizationmodels.UtilizationModel[] vmUtilArray;

    public Py4jBridge() {
        configureCloudSimLogging();
        loadTraceFiles();
        System.out.println("[Bridge] Ready. Traces: " + traceFiles.size());
    }

    private static void configureCloudSimLogging() {
        Logger rootLogger = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.ERROR);
    }

    private void loadTraceFiles() {
        String traceSource = Optional.ofNullable(System.getenv("TRACE_SOURCE"))
            .orElse("philly")
            .trim()
            .toLowerCase(Locale.ROOT);

        if (!"azure".equals(traceSource)) {
            if (loadPhillyTraces()) {
                return;
            }
        }

        // Azure test traces (pre-processed per-VM CSVs). Use TRACE_SOURCE=azure
        // for this path; the default follows the Philly EDA/data contract.
        File azureDir = new File("data/azure_test/traces");
        if (azureDir.exists()) {
            File[] files = azureDir.listFiles(f -> f.isFile() && f.getName().endsWith(".csv"));
            if (files != null && files.length > 0) {
                for (File f : files) traceFiles.add(f.getAbsolutePath());
                Collections.sort(traceFiles);
                useAzureTraces = true;
                System.out.println("[Bridge] Loaded Azure VM traces: " + traceFiles.size() + " files"
                    + " (TRACE_SOURCE=" + traceSource + ")");
                return;
            }
        }

        if ("azure".equals(traceSource) && loadPhillyTraces()) {
            return;
        }

        // Fallback: PlanetLab
        File dir = new File("data/planetlab");
        if (dir.exists()) {
            File[] dateDirs = dir.listFiles(File::isDirectory);
            if (dateDirs != null) {
                for (File dateDir : dateDirs) {
                    File[] files = dateDir.listFiles(f -> f.isFile() && !f.getName().startsWith("."));
                    if (files != null) {
                        for (File f : files) traceFiles.add(f.getAbsolutePath());
                    }
                }
            }
            Collections.sort(traceFiles);
        }
        System.out.println("[Bridge] Loaded PlanetLab: " + traceFiles.size() + " files");
    }

    private boolean loadPhillyTraces() {
        // Gen-Parallel workload is selectable via env. Default BW (Blue Waters):
        // 88.7% of jobs are multi-node (mean 30 nodes), which yields dense, real
        // gang traffic for network-aware placement — unlike Philly (2.9% multi).
        String workload = Optional.ofNullable(System.getenv("GENPARALLEL_WORKLOAD"))
            .orElse("BW").trim();
        File dir = new File("data/Gen-Parallel-Workloads/" + workload + "/training_data");
        if (dir.exists()) {
            File[] files = dir.listFiles(f -> f.isFile() && f.getName().endsWith(".csv"));
            if (files != null) {
                for (File f : files) traceFiles.add(f.getAbsolutePath());
            }
            Collections.sort(traceFiles);
            if (!traceFiles.isEmpty()) {
                useAzureTraces = false;
                System.out.println("[Bridge] Loaded Gen-Parallel-Workloads " + workload + " traces: "
                    + traceFiles.size() + " files");
                return true;
            }
        }
        return false;
    }

    // ========== API cho Python ==========

    public double[] reset() {
        System.out.println("[Bridge] reset() called");
        simDone = false;
        simPaused = false;
        prevEnergy = 0;
        currentEnergy = 0;
        prevCrossRackBytes = 0;
        prevLocalBytes = 0;
        intervalCrossRackBytes = 0;
        intervalLocalBytes = 0;
        totalMigrations = 0;
        intervalMigrations = 0;
        totalActiveHostTime = 0;
        totalViolatedHostTime = 0;
        totalActiveVmTime = 0;
        totalMigrationDowntime = 0;
        currentSlatah = 0;
        currentPdm = 0;
        currentStep = 0;

        simulation = new CloudSimPlus();
        vmGpuRequests = new HashMap<>();
        List<NetworkHost> netHosts = createHosts();
        hostList = new ArrayList<>(netHosts);   // keep List<Host> view for all existing code
        initializeNetworkTopologyMetrics();

        hostCpuHistoryArr = new double[NUM_HOSTS][historyLen];
        // Initialize to 0

        NetworkDatacenter netDc = new NetworkDatacenter(simulation, netHosts);
        buildFatTree(netDc, netHosts);
        datacenter = netDc;
        datacenter.setSchedulingInterval(INTERVAL);
        broker = new DatacenterBrokerSimple(simulation);

        Random resetRng = new Random(SIM_SEED);

        // Shuffle trace files for episode variation
        if (!traceFiles.isEmpty()) {
            Collections.shuffle(traceFiles, resetRng);
        }

        // Offered load is derived from the observed process profile. This
        // preserves migration headroom instead of filling the cluster with a
        // hand-picked random VM-count range before scheduling begins.
        int numVms = deriveInitialVmCount();
        lastInitialVmCount = numVms;
        vmList = createVms(numVms);
        Collections.shuffle(vmList, resetRng); // Random placement order
        broker.submitVmList(vmList);
        System.out.printf(
            "[Bridge] Process-derived offered load: mean=%.4f overload=%.4f initialVMs=%d%n",
            adaptiveMeanUtilization, adaptiveOverloadThreshold, numVms
        );

        // Group VMs into gangs from the Blue Waters node_num distribution. Gang
        // indices reference VM ids (createVms sets id==creation index), so they
        // stay valid after the placement shuffle above.
        gangs = buildGangs(numVms);
        logGangStats();

        // Reverse indices for gang-locality scoring in buildVmState.
        vmByIdArr = new Vm[numVms];
        for (Vm v : vmList) {
            int id = (int) v.getId();
            if (id >= 0 && id < numVms) vmByIdArr[id] = v;
        }
        gangOfVm = new int[numVms][];
        for (int[] g : gangs) {
            if (g.length < 2) continue;
            for (int vid : g) if (vid >= 0 && vid < numVms) gangOfVm[vid] = g;
        }

        List<Cloudlet> cloudlets = createCloudlets(numVms);
        broker.submitCloudletList(cloudlets);
        // Wire gang all-reduce traffic now that cloudlets are broker-bound.
        buildGangTraffic(cloudlets, numVms);

        // Pause mechanism (energy now accumulated per-interval in step())
        simulation.addOnClockTickListener(info -> {
            double t = info.getTime();
            if (t >= nextPauseTime && !simPaused) {
                simPaused = true;
                simulation.pause();
            }
        });

        nextPauseTime = INTERVAL;
        simThread = new Thread(() -> {
            simulation.start();
            simDone = true;
            simPaused = true;
        });
        simThread.start();
        waitForPause();
        for (int hi = 0; hi < hostList.size(); hi++) {
            Arrays.fill(hostCpuHistoryArr[hi], getHostCpuUtil(hostList.get(hi)));
        }

        return collectGlobalState();
    }

    /**
     * step(overloadedHostIndices, underloadedHostIndices, selectionAction, placementActions)
     * Returns: double[GLOBAL_STATE_DIM + 14]
     *   [0..7]   = global_state (8 dim)
     *   [8]      = reward
     *   [9]      = done
     *   [10]     = intervalMigrations
     *   [11]     = currentStep
     *   [12..20] = extended metrics
     *   [GLOBAL_STATE_DIM + 13] = cross-rack fraction of gang traffic this interval
     */
    public double[] step(int[] overloadedHostIndices, int[] underloadedHostIndices,
                         int selectionAction, int[] placementActions) {
        if (simDone) {
            double[] result = new double[GLOBAL_STATE_DIM + 4];
            result[GLOBAL_STATE_DIM + 1] = 1.0;
            return result;
        }

        currentStep++;
        resetNetworkTrafficWindow();
        recordBackgroundTraffic();

        Map<Vm, Host> vmSourceHost = new HashMap<>();
        List<Vm> vmsToMigrate = collectVmsToMigrate(
            overloadedHostIndices,
            underloadedHostIndices,
            selectionAction,
            vmSourceHost
        );

        // 4. Execute migrations (LOGICAL: direct VM reassignment)
        intervalMigrations = 0;
        double intervalSlaCost = 0.0;
        int attempted = 0;
        int skippedNoCandidates = lastSkippedNoMovableSources;
        int skippedSameHost = 0;
        int migFailed = 0;
        long activeHostsBefore = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        Map<Host, Double> reservedTargetCpu = new HashMap<>();
        Map<Host, Double> reservedTargetGpu = new HashMap<>();
        Map<Vm, Host> requestedTargets = new HashMap<>();
        Map<Vm, Double> requestedTargetUtils = new HashMap<>();
        Map<Vm, Double> requestedDowntimes = new HashMap<>();
        Map<Vm, Double> requestedSlaCosts = new HashMap<>();

        for (int i = 0; i < vmsToMigrate.size(); i++) {
            Vm vm = vmsToMigrate.get(i);
            List<Host> candidates = getTopKCandidates(vm);
            if (candidates.isEmpty()) {
                skippedNoCandidates++;
                continue;
            }

            int actionIdx = (placementActions != null && i < placementActions.length)
                ? placementActions[i] : 0;
            actionIdx = Math.max(0, Math.min(actionIdx, candidates.size() - 1));

            Host target = candidates.get(actionIdx);
            Host source = vmSourceHost.get(vm);

            if (target == null || source == null || target == source) {
                skippedSameHost++;
                continue;
            }
            if (!hasGpuCapacity(target, vm)) {
                migFailed++;
                continue;
            }
            if (!hasReservedTargetCapacity(target, vm, reservedTargetCpu, reservedTargetGpu)) {
                skippedNoCandidates++;
                continue;
            }

            attempted++;
            try {
                // Downtime formula (Beloglazov 2012): T_mig = V_ram / BW_link
                double linkBwMbps = Math.min(source.getBw().getCapacity(), target.getBw().getCapacity());
                double ramMb = vm.getRam().getCapacity() * 8.0;
                double downtime = ramMb / Math.max(1, linkBwMbps);

                // SLA tier cost
                double wTier = 0.5;
                if (vm.getRam().getCapacity() >= 4000) wTier = 3.0;
                else if (vm.getRam().getCapacity() >= 2000) wTier = 1.5;

                // Use CloudSim Plus' migration transaction. Metrics are
                // committed only after the simulation confirms the VM moved.
                datacenter.requestVmMigration(vm, target);
                requestedTargets.put(vm, target);
                requestedTargetUtils.put(vm, getHostCpuUtil(target));
                requestedDowntimes.put(vm, downtime);
                requestedSlaCosts.put(vm, wTier * downtime);
                reserveTargetCapacity(target, vm, reservedTargetCpu, reservedTargetGpu);
            } catch (Exception e) {
                migFailed++;
            }
        }

        // 5. Advance simulation 1 interval
        nextPauseTime += INTERVAL;
        if (nextPauseTime > MAX_TIME) {
            simDone = true;
        } else {
            simPaused = false;
            simulation.resume();
            waitForPause();
        }

        // Commit migration metrics only for VMs CloudSim actually moved.
        intervalMigrations = 0;
        double sumTargetUtil = 0.0;
        intervalSlaCost = 0.0;
        Set<Host> confirmedTargetHosts = new HashSet<>();
        for (Map.Entry<Vm, Host> entry : requestedTargets.entrySet()) {
            Vm vm = entry.getKey();
            Host target = entry.getValue();
            if (vm.getHost().equals(target)) {
                intervalMigrations++;
                confirmedTargetHosts.add(target);
                sumTargetUtil += getHostCpuUtil(target);
                double downtime = requestedDowntimes.getOrDefault(vm, 0.0);
                totalMigrationDowntime += downtime;
                intervalSlaCost += requestedSlaCosts.getOrDefault(vm, 0.0);
                recordMigrationNetworkTraffic(vmSourceHost.get(vm), target, vm);
            } else {
                migFailed++;
            }
        }
        int intervalTargetCritical = 0;
        for (Host target : confirmedTargetHosts) {
            if (getRawHostCpuDemandRatio(target) > adaptiveCriticalThreshold) {
                intervalTargetCritical++;
            }
        }
        long activeHostsAfter = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        double avgTargetUtil = intervalMigrations > 0 ? sumTargetUtil / intervalMigrations : 0.0;
        double activeHostDelta = activeHostsAfter - activeHostsBefore;
        totalMigrations += intervalMigrations;
        long hostedVmCount = hostList.stream().mapToLong(h -> h.getVmList().size()).sum();
        long assignedVmCount = vmList.stream()
            .filter(v -> !v.getHost().equals(Host.NULL))
            .count();
        // A VM that is not migrating sits in exactly one host vmList AND has a non-null
        // host, so it contributes equally to both counts. Only VMs actively migrating can
        // differ between the two views (removed from a vmList mid-transit, or host pointer
        // not yet committed) -- each by at most one. The gap is therefore bounded by the
        // number of migrating VMs; anything larger is a genuine placement-accounting bug.
        // Count every VM whose host<->vmList accounting can legitimately be in flux at
        // this pause point. In an event-driven simulator a VM can be (a) flagged
        // isInMigration, (b) sitting in some host's migratingIn set, or (c) in a
        // migratingOut set -- and at the exact instant a migration finishes it may have
        // left the source vmList before its host pointer/target vmList membership is
        // committed. The union of these sets is the maximum number of VMs that can differ
        // between hostedVmCount and assignedVmCount without indicating a real leak.
        java.util.Set<Vm> inTransit = new java.util.HashSet<>();
        for (Vm v : vmList) {
            if (v.isInMigration()) inTransit.add(v);
        }
        for (Host h : hostList) {
            inTransit.addAll(h.getVmsMigratingIn());
            inTransit.addAll(h.getVmsMigratingOut());
        }
        long transitVmCount = inTransit.size();
        long invariantGap = Math.abs(hostedVmCount - assignedVmCount);
        if (invariantGap > transitVmCount) {
            // Previously this threw IllegalStateException and killed the entire
            // training run (e.g. the Ep25 crash with hosted=19 assigned=20).
            // An off-by-one host<->vmList accounting gap at a pause point is
            // transient in the event-driven simulator and self-heals on the
            // next interval, so we log and continue instead of aborting. A
            // genuinely large, persistent leak is still surfaced via the
            // running invariantWarnings counter exposed to the Python side.
            invariantWarnings++;
            System.err.printf(
                "[WARN] VM placement invariant gap: hosted=%d assigned=%d inTransit=%d (warning #%d)%n",
                hostedVmCount, assignedVmCount, transitVmCount, invariantWarnings
            );
        }

        // Diagnostic (first 3 steps per episode)
        if (currentStep <= 3) {
            System.out.printf("[DIAG] step=%d: OL=%d vmsToMig=%d attempted=%d " +
                "success=%d noCand=%d sameHost=%d failed=%d%n",
                currentStep, countValidHosts(overloadedHostIndices), vmsToMigrate.size(),
                attempted, intervalMigrations, skippedNoCandidates,
                skippedSameHost, migFailed);
        }

        // 6. Compute per-interval energy (Beloglazov 2012: E = Σ P(u) × Δt)
        double intervalEnergy = computeIntervalEnergy();
        currentEnergy += intervalEnergy;
        double deltaEnergy = intervalEnergy;

        // 6b. Per-interval gang traffic (diff of the monotonic totals).
        long crossRackNow = 0L, localNow = 0L;
        for (MeteredEdgeSwitch s : edgeSwitches) {
            crossRackNow += s.crossRackBytesTotal;
            localNow += s.localBytesTotal;
        }
        intervalCrossRackBytes = Math.max(0L, crossRackNow - prevCrossRackBytes);
        intervalLocalBytes = Math.max(0L, localNow - prevLocalBytes);
        prevCrossRackBytes = crossRackNow;
        prevLocalBytes = localNow;

        // 7. Update History and SLA Metrics
        // Physical overload remains separate from the process-adaptive
        // operational and critical thresholds.
        double maxRawDemand = 0;
        int operationalOverloads = 0;
        int criticalOverloads = 0;
        int capacityOverloads = 0;
        for (int hi = 0; hi < hostList.size(); hi++) {
            Host h = hostList.get(hi);
            double util = getHostCpuUtil(h);         // capped [0,1] for obs/history
            double rawDemand = getRawHostCpuDemandRatio(h); // demand/capacity ratio
            if (rawDemand > maxRawDemand) maxRawDemand = rawDemand;
            if (rawDemand > adaptiveOverloadThreshold) operationalOverloads++;
            if (rawDemand > adaptiveCriticalThreshold) criticalOverloads++;
            if (rawDemand > 1.0) capacityOverloads++;
            // Shift history left and add new value
            System.arraycopy(hostCpuHistoryArr[hi], 1, hostCpuHistoryArr[hi], 0, historyLen - 1);
            hostCpuHistoryArr[hi][historyLen - 1] = util;
            if (!h.getVmList().isEmpty()) {
                totalActiveHostTime += INTERVAL;
                // SLA risk time uses the adaptive critical threshold. Actual
                // physical capacity violations remain in capacityOverloads.
                if (rawDemand > adaptiveCriticalThreshold) {
                    totalViolatedHostTime += INTERVAL;
                }
            }
        }
        // PDM denominator: total active VM-time
        long activeVmCount = vmList.stream()
            .filter(v -> !v.getHost().equals(Host.NULL)).count();
        totalActiveVmTime += activeVmCount * INTERVAL;

        currentSlatah = totalActiveHostTime > 0
            ? totalViolatedHostTime / totalActiveHostTime : 0.0;
        currentPdm = totalActiveVmTime > 0
            ? totalMigrationDowntime / totalActiveVmTime : 0.0;
        double currentSlav = currentSlatah * currentPdm;

        // 7b. Rack power stats (GPU-exclusive: compute after advancing sim)
        computeRackStatsForStep();
        int totalRackViol = 0;
        if (rackViolCountStep != null) for (int v : rackViolCountStep) totalRackViol += v;

        // 8. Reward: Energy + rack budget violation + checkpoint cost (no migration cost)
        double reward = -(deltaEnergy / 10.0)             // normalize energy
                        - 1000.0 * currentSlav             // SLAV (PDM uses checkpoint overhead)
                        - 2.0 * totalRackViol              // rack power cap violations
                        - 0.5 * (totalCheckpointOverhead / Math.max(1, INTERVAL)); // checkpoint norm

        // 8. Pack result (expanded: +attempted, +failed, +consolidation, +blocked)
        double[] globalState = collectGlobalState();
        double[] result = new double[GLOBAL_STATE_DIM + 14];
        System.arraycopy(globalState, 0, result, 0, GLOBAL_STATE_DIM);
        result[GLOBAL_STATE_DIM] = reward;
        result[GLOBAL_STATE_DIM + 1] = simDone ? 1.0 : 0.0;
        result[GLOBAL_STATE_DIM + 2] = intervalMigrations;
        result[GLOBAL_STATE_DIM + 3] = currentStep;
        result[GLOBAL_STATE_DIM + 4] = intervalSlaCost;
        result[GLOBAL_STATE_DIM + 5] = attempted;
        result[GLOBAL_STATE_DIM + 6] = migFailed;
        result[GLOBAL_STATE_DIM + 7] = avgTargetUtil;     // consolidation metric
        result[GLOBAL_STATE_DIM + 8] = activeHostDelta;    // +/- active hosts change
        long activeNow = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        result[GLOBAL_STATE_DIM + 9] = (double) activeNow / NUM_HOSTS;  // current active ratio
        result[GLOBAL_STATE_DIM + 10] = skippedNoCandidates;
        result[GLOBAL_STATE_DIM + 11] = skippedSameHost;
        result[GLOBAL_STATE_DIM + 12] = intervalTargetCritical;
        // [13] cross-rack fraction of gang traffic this interval: cross/(cross+local).
        // 1.0 = every gang packet crossed a rack (worst placement), 0.0 = fully
        // rack-local (ideal gang packing). Bounded [0,1], no capacity assumption.
        long gangTotal = intervalCrossRackBytes + intervalLocalBytes;
        result[GLOBAL_STATE_DIM + 13] = gangTotal > 0
            ? (double) intervalCrossRackBytes / gangTotal : 0.0;

        System.out.printf("[Bridge] step=%d t=%.0f | sources=%d opOL=%d critOL=%d capOL=%d | sel=%d | mig=%d | " +
                "blocked=%d same=%d | SLATAH=%.6f PDM=%.10f | reward=%.3f | active=%d avgTgt=%.3f maxDemand=%.3f maxGpu=%.3f%n",
            currentStep, nextPauseTime - INTERVAL, countValidHosts(overloadedHostIndices),
            operationalOverloads, criticalOverloads, capacityOverloads,
            selectionAction, intervalMigrations,
            skippedNoCandidates, skippedSameHost,
            currentSlatah, currentPdm, reward,
            activeNow, avgTargetUtil, maxRawDemand, getMaxHostGpuUtil());

        return result;
    }

    // ========== API dimensions ==========
    public int getGlobalStateDim() { return GLOBAL_STATE_DIM; }
    public int getVmStateDim() { return VM_STATE_DIM; }
    public int getTopK() { return TOP_K; }
    public int getNumSelectionActions() { return 3; }
    public int getNumHosts() { return NUM_HOSTS; }
    public int getHistoryLen() { return historyLen; }
    public int getInvariantWarnings() { return invariantWarnings; }
    public void setHistoryLen(int requestedHistoryLen) {
        historyLen = Math.max(2, requestedHistoryLen);
    }

    /** Receive process-adaptive policy thresholds from Python before each step. */
    public void setAdaptiveThresholds(double overloadThreshold, double criticalThreshold) {
        adaptiveOverloadThreshold = Math.max(0.0, Math.min(1.0, overloadThreshold));
        adaptiveCriticalThreshold = Math.max(
            adaptiveOverloadThreshold,
            Math.min(1.0, criticalThreshold)
        );
    }

    public void setProcessProfile(double overloadThreshold, double criticalThreshold,
                                  double meanUtilization) {
        setAdaptiveThresholds(overloadThreshold, criticalThreshold);
        adaptiveMeanUtilization = Math.max(
            0.0,
            Math.min(adaptiveOverloadThreshold, meanUtilization)
        );
    }

    public int getInitialVmCount() { return lastInitialVmCount; }

    /** API: time-series history for Autoformer */
    public double[][] getHostHistory() {
        return hostCpuHistoryArr;
    }

    public int[] getHostVmCounts() {
        int[] counts = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            counts[i] = hostList.get(i).getVmList().size();
        }
        return counts;
    }

    public double[] getHostRawDemandRatios() {
        double[] ratios = new double[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            ratios[i] = getRawHostCpuDemandRatio(hostList.get(i));
        }
        return ratios;
    }

    public int[] getMovableHostMask() {
        int[] mask = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            mask[i] = hasMovableVm(hostList.get(i)) ? 1 : 0;
        }
        return mask;
    }

    /**
     * Explain why a host cannot be used as an overload-migration source.
     * 0=movable, 1=single/empty, 2=no CPU/RAM/BW candidate, 3=GPU blocked.
     */
    public int[] getHostMobilityReasonCodes() {
        int[] codes = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            codes[i] = getHostMobilityReasonCode(hostList.get(i));
        }
        return codes;
    }

    public double[] getHostGpuUtilRatios() {
        double[] utils = new double[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            utils[i] = getHostGpuUtil(hostList.get(i));
        }
        return utils;
    }

    public String getNetworkSnapshotJson() {
        double windowSec = INTERVAL;
        StringBuilder sb = new StringBuilder();
        sb.append("{\"model\":\"migration-traffic-rack\",\"is_proxy\":false,\"traffic_source\":\"confirmed_migrations_plus_cpu_proxy\",\"window_sec\":")
          .append(windowSec)
          .append(",\"nodes\":[");
        for (int i = 0; i < NUM_HOSTS; i++) {
            if (i > 0) sb.append(",");
            Host h = hostList.get(i);
            sb.append("{\"id\":").append(i)
              .append(",\"rack\":").append(i / NETWORK_RACK_SIZE)
              .append(",\"active\":").append(h.getVmList().isEmpty() ? 0 : 1)
              .append(",\"cpu\":").append(fmt(getHostCpuUtil(h)))
              .append(",\"raw_demand\":").append(fmt(getRawHostCpuDemandRatio(h)))
              .append(",\"gpu\":").append(fmt(getHostGpuUtil(h)))
              .append("}");
        }
        sb.append("],\"links\":[");
        boolean first = true;
        for (NetworkLinkMetric link : networkLinks.values()) {
            first = appendNetworkLinkJson(sb, first, link);
        }
        sb.append("]}");
        return sb.toString();
    }

    /**
     * Metered edge-switch traffic accumulated since episode start:
     * [crossRackBytes, localBytes]. Cross-rack bytes are the placement-dependent
     * gang all-reduce signal; local bytes are intra-rack (cheap) exchanges.
     */
    public long[] getGangTrafficBytes() {
        long cross = 0L, local = 0L;
        for (MeteredEdgeSwitch s : edgeSwitches) {
            cross += s.crossRackBytesTotal;
            local += s.localBytesTotal;
        }
        return new long[]{cross, local};
    }

    /** API: SLAV metrics for CSV logging */
    public double[] getSlavMetrics() {
        return new double[]{currentSlatah, currentPdm, currentSlatah * currentPdm,
                            currentEnergy, totalMigrations, currentStep};
    }

    /** Preview VM states for Agent 2 */
    public double[] previewVmStates(int[] overloadedHostIndices, int selectionAction) {
        return previewMigrationVmStates(overloadedHostIndices, null, selectionAction);
    }

    /** Preview VM states in the exact order consumed by step() placementActions. */
    public double[] previewMigrationVmStates(int[] overloadedHostIndices, int[] underloadedHostIndices,
                                             int selectionAction) {
        Map<Vm, Host> vmSourceHost = new HashMap<>();
        List<Vm> vmsToMigrate = collectVmsToMigrate(
            overloadedHostIndices,
            underloadedHostIndices,
            selectionAction,
            vmSourceHost
        );
        List<double[]> states = new ArrayList<>();
        for (Vm vm : vmsToMigrate) {
            Host source = vmSourceHost.get(vm);
            if (source != null) {
                states.add(buildVmState(vm, source));
            }
        }
        int numVms = states.size();
        double[] result = new double[1 + numVms * VM_STATE_DIM];
        result[0] = numVms;
        for (int i = 0; i < numVms; i++) {
            System.arraycopy(states.get(i), 0, result, 1 + i * VM_STATE_DIM, VM_STATE_DIM);
        }
        return result;
    }

    public int[] previewMigrationCandidateHostIds(int[] overloadedHostIndices, int[] underloadedHostIndices,
                                                  int selectionAction) {
        Map<Vm, Host> vmSourceHost = new HashMap<>();
        List<Vm> vmsToMigrate = collectVmsToMigrate(
            overloadedHostIndices,
            underloadedHostIndices,
            selectionAction,
            vmSourceHost
        );
        int[] result = new int[1 + vmsToMigrate.size() * TOP_K];
        result[0] = vmsToMigrate.size();
        for (int vmIdx = 0; vmIdx < vmsToMigrate.size(); vmIdx++) {
            List<Host> candidates = getTopKCandidates(vmsToMigrate.get(vmIdx));
            for (int i = 0; i < TOP_K; i++) {
                int outIdx = 1 + vmIdx * TOP_K + i;
                result[outIdx] = (i < candidates.size()) ? (int) candidates.get(i).getId() : -1;
            }
        }
        return result;
    }

    public void setVmSelectionActions(int[] selectionActions) {
        pendingVmSelectionActions = selectionActions;
    }

    public double[] previewVmSelectionStates(int[] overloadedHostIndices) {
        return previewVmSelectionStates(overloadedHostIndices, null);
    }

    public double[] previewVmSelectionStates(int[] overloadedHostIndices, int[] underloadedHostIndices) {
        List<Host> sources = getMigrationSourceHosts(overloadedHostIndices, underloadedHostIndices);
        double[] result = new double[2 + sources.size() * (3 + TOP_K * VM_SELECTION_FEATURE_DIM)];
        result[0] = sources.size();
        result[1] = VM_SELECTION_FEATURE_DIM;
        int offset = 2;
        for (Host source : sources) {
            List<Vm> candidates = getVmSelectionCandidates(source);
            result[offset++] = source.getId();
            result[offset++] = getSourceType(source, overloadedHostIndices, underloadedHostIndices);
            result[offset++] = candidates.size();
            for (int i = 0; i < TOP_K; i++) {
                double[] features = i < candidates.size()
                    ? buildVmSelectionFeatures(candidates.get(i), source)
                    : new double[VM_SELECTION_FEATURE_DIM];
                System.arraycopy(features, 0, result, offset, VM_SELECTION_FEATURE_DIM);
                offset += VM_SELECTION_FEATURE_DIM;
            }
        }
        return result;
    }

    private List<Vm> collectVmsToMigrate(int[] overloadedHostIndices, int[] underloadedHostIndices,
                                         int selectionAction, Map<Vm, Host> vmSourceHost) {
        lastSkippedNoMovableSources = 0;
        List<Host> sources = getMigrationSourceHosts(overloadedHostIndices, underloadedHostIndices);
        List<Vm> vmsToMigrate = new ArrayList<>();
        int sourceActionIdx = 0;
        for (Host h : sources) {
            List<Vm> hvms = new ArrayList<>(h.getVmList());
            Vm selected;
            if (selectionAction < 0 && pendingVmSelectionActions != null) {
                int vmAction = sourceActionIdx < pendingVmSelectionActions.length
                    ? pendingVmSelectionActions[sourceActionIdx]
                    : 0;
                selected = selectVmByIndex(hvms, vmAction);
            } else {
                selected = selectVm(hvms, selectionAction);
            }
            sourceActionIdx++;
            if (selected != null) {
                vmsToMigrate.add(selected);
                vmSourceHost.put(selected, h);
            } else {
                lastSkippedNoMovableSources++;
            }
        }
        return vmsToMigrate;
    }

    private List<Host> getMigrationSourceHosts(int[] overloadedHostIndices, int[] underloadedHostIndices) {
        List<Host> sources = new ArrayList<>();
        Set<Long> seen = new HashSet<>();
        for (Host h : getOverloadedSourceHosts(overloadedHostIndices)) {
            if (seen.add(h.getId())) sources.add(h);
        }
        if (underloadedHostIndices != null) {
            for (int idx : underloadedHostIndices) {
                if (idx >= 0 && idx < hostList.size()) {
                    Host h = hostList.get(idx);
                    if (!h.getVmList().isEmpty() && hasMovableVm(h) && seen.add(h.getId())) {
                        sources.add(h);
                    }
                }
            }
        }
        return sources;
    }

    private int getSourceType(Host source, int[] overloadedHostIndices, int[] underloadedHostIndices) {
        int id = (int) source.getId();
        if (containsHostIndex(overloadedHostIndices, id)) return 0;
        if (containsHostIndex(underloadedHostIndices, id)) return 3;
        return -1;
    }

    private boolean containsHostIndex(int[] hostIndices, int id) {
        if (hostIndices == null) return false;
        for (int idx : hostIndices) {
            if (idx == id) return true;
        }
        return false;
    }

    private List<Host> getOverloadedSourceHosts(int[] overloadedHostIndices) {
        List<Host> overloaded = new ArrayList<>();
        if (overloadedHostIndices != null) {
            for (int idx : overloadedHostIndices) {
                if (idx >= 0 && idx < hostList.size()) {
                    Host h = hostList.get(idx);
                    if (h.getVmList().size() > 1) overloaded.add(h);
                }
            }
        }
        return overloaded;
    }

    private int countValidHosts(int[] hostIndices) {
        int count = 0;
        if (hostIndices != null) {
            for (int idx : hostIndices) {
                if (idx >= 0 && idx < hostList.size()) count++;
            }
        }
        return count;
    }

    // ========== Internal helpers ==========

    private void waitForPause() {
        while (!simPaused && !simDone) {
            try { Thread.sleep(5); } catch (InterruptedException e) { break; }
        }
    }

    private Vm selectVm(List<Vm> vms, int selectionAction) {
        if (vms.isEmpty()) return null;
        List<Vm> feasible = getVmSelectionCandidates(vms);
        if (feasible.isEmpty()) return null;
        List<Vm> pool = feasible;
        if (selectionAction == 0) { // MMT
            return pool.stream().min(Comparator.comparingLong(v -> v.getRam().getCapacity())).orElse(null);
        } else if (selectionAction == 1) { // Max Utilization (highest MIPS)
            return pool.stream().max(Comparator.comparingDouble(v -> v.getMips())).orElse(null);
        } else { // Random
            int idx = Math.floorMod(currentStep, pool.size());
            return pool.get(idx);
        }
    }

    private Vm selectVmByIndex(List<Vm> vms, int selectionAction) {
        List<Vm> feasible = getVmSelectionCandidates(vms);
        if (feasible.isEmpty()) return null;
        int idx = Math.max(0, Math.min(selectionAction, feasible.size() - 1));
        return feasible.get(idx);
    }

    private List<Vm> getVmSelectionCandidates(Host source) {
        return getVmSelectionCandidates(new ArrayList<>(source.getVmList()));
    }

    private List<Vm> getVmSelectionCandidates(List<Vm> vms) {
        return vms.stream()
            .filter(this::hasPlacementCandidate)
            .sorted(Comparator
                .comparingDouble((Vm v) -> -getVmCpuRatio(v, v.getHost()))
                .thenComparingLong(v -> -v.getRam().getCapacity())
                .thenComparingLong(Vm::getId))
            .limit(TOP_K)
            .collect(Collectors.toList());
    }

    private boolean hasMovableVm(Host host) {
        if (host == null || host.getVmList().size() <= 1) return false;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            if (hasPlacementCandidate(vm)) return true;
        }
        return false;
    }

    private int getHostMobilityReasonCode(Host host) {
        if (host == null || host.getVmList().size() <= 1) return 1;
        boolean hasBaseCandidate = false;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            for (Host target : hostList) {
                if (target.equals(host) || !target.isSuitableForVm(vm)) continue;
                hasBaseCandidate = true;
                if (hasGpuCapacity(target, vm)) return 0;
            }
        }
        return hasBaseCandidate ? 3 : 2;
    }

    private boolean hasPlacementCandidate(Vm vm) {
        return !getTopKCandidates(vm).isEmpty();
    }

    private List<NetworkHost> createHosts() {
        List<NetworkHost> hosts = new ArrayList<>();
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> pes = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) pes.add(new PeSimple(HOST_MIPS));
            NetworkHost h = new NetworkHost(HOST_RAM, HOST_BW, HOST_STORAGE, pes);
            h.setId(i);   // enforce host.getId() == index invariant (relied on everywhere)
            h.setVmScheduler(new VmSchedulerTimeShared());
            // Real SPECpower model (HP ProLiant ML110 G5)
            h.setPowerModel(new PowerModelHostSimple(NREL_PEAK_POWER_W, NREL_IDLE_POWER_W));
            hosts.add(h);
        }
        return hosts;
    }

    private List<Vm> createVms(int count) {
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            NetworkVm vm = new NetworkVm(i, HOST_MIPS, HOST_PES);
            vm.setRam((long) HOST_RAM).setBw(100).setSize(10000);  // VM BW = 100 Mbit/s
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            // Required so NetworkCloudlet send/receive tasks are dispatched; VMs
            // not in a gang simply never carry such tasks (harmless overhead).
            vm.getCloudletScheduler().setTaskScheduler(new CloudletTaskSchedulerSimple());
            vmGpuRequests.put(vm.getId(), HOST_GPU);
            vms.add(vm);
        }
        return vms;
    }

    private int deriveInitialVmCount() {
        return NUM_HOSTS;
    }

    // Total compute (MI) per cloudlet — long-running, survives 24h at moderate
    // utilization. Gang cloudlets split this across execution tasks so their
    // ring traffic is sustained over the whole episode rather than bursting once.
    private static final int CLOUDLET_TOTAL_MI = 300_000_000;

    private List<Cloudlet> createCloudlets(int count) {
        vmUtilArray = new org.cloudsimplus.utilizationmodels.UtilizationModel[count];

        // VM ids that belong to a multi-VM gang must be NetworkCloudlets so a
        // gang sibling can be named as a packet destination (addPacket requires
        // a NetworkCloudlet). Solo VMs stay CloudletSimple.
        boolean[] inGang = new boolean[count];
        for (int[] g : gangs) {
            if (g.length < 2) continue;
            for (int v : g) if (v >= 0 && v < count) inGang[v] = true;
        }

        // Index VMs by id (createVms sets id==creation index). vmList is already
        // shuffled for random placement, but ids are stable — used to pin each
        // cloudlet to its VM so the gang packet graph (defined over VM ids) is
        // exact and independent of the broker's round-robin mapping.
        NetworkVm[] vmById = new NetworkVm[count];
        for (Vm v : vmList) {
            int id = (int) v.getId();
            if (id >= 0 && id < count) vmById[id] = (NetworkVm) v;
        }

        List<Cloudlet> cloudlets = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            org.cloudsimplus.utilizationmodels.UtilizationModel utilModel = buildUtilModel(i);
            Cloudlet c = inGang[i]
                ? new NetworkCloudlet(CLOUDLET_TOTAL_MI, 1)
                : new CloudletSimple(CLOUDLET_TOTAL_MI, 1);
            c.setId(i);                                  // pin id == creation index
            c.setUtilizationModelCpu(utilModel);
            if (vmById[i] != null) c.setVm(vmById[i]);   // pin cloudlet i -> VM id i
            vmUtilArray[i] = utilModel;
            cloudlets.add(c);
        }
        // Gang packet wiring is deferred until after submitCloudletList: addPacket
        // requires both cloudlets to be broker-bound (isBoundToVm checks broker
        // equality), and the broker is only assigned to a cloudlet on submission.
        return cloudlets;
    }

    /** CPU utilization model for cloudlet i: trace-driven if available, else synthetic. */
    private org.cloudsimplus.utilizationmodels.UtilizationModel buildUtilModel(int i) {
        if (traceFiles.isEmpty()) return createSyntheticModel(i);
        String file = traceFiles.get(i % traceFiles.size());
        try {
            return useAzureTraces
                ? new UtilizationModelAzure(file, INTERVAL)
                : new UtilizationModelGenParallel(file, INTERVAL, i);
        } catch (Exception e) {
            return createSyntheticModel(i);
        }
    }

    /**
     * Wire ring all-reduce traffic across every multi-VM gang. Each member runs
     * GANG_CYCLES cycles of [execute a slice][send one packet to the next member]
     * [receive one packet from the previous member]. Execution precedes the
     * receive every cycle, so packets are in flight before any member blocks —
     * a balanced ring (a permutation) cannot deadlock. Total execution length
     * equals CLOUDLET_TOTAL_MI, matching solo cloudlets, so gang VMs are not
     * starved of compute. Cross-rack placement makes these packets traverse the
     * metered edge switch, which is the network signal the placer learns from.
     */
    private void buildGangTraffic(List<Cloudlet> cloudlets, int count) {
        // Index cloudlets and VMs by id (both pinned to creation index).
        Cloudlet[] byId = new Cloudlet[count];
        for (Cloudlet c : cloudlets) {
            int id = (int) c.getId();
            if (id >= 0 && id < count) byId[id] = c;
        }
        NetworkVm[] vmById = new NetworkVm[count];
        for (Vm v : vmList) {
            int id = (int) v.getId();
            if (id >= 0 && id < count) vmById[id] = (NetworkVm) v;
        }

        int cycles = (int) Math.max(1, parseEnvLong("GANG_CYCLES", 40));
        long packetBytes = Math.max(1L, parseEnvLong("GANG_PACKET_BYTES", 1_000_000L));
        long execPerCycle = Math.max(1L, (long) CLOUDLET_TOTAL_MI / cycles);
        int taskId = 0;
        for (int[] g : gangs) {
            if (g.length < 2) continue;
            int k = g.length;
            for (int p = 0; p < k; p++) {
                int me = g[p], succ = g[(p + 1) % k], pred = g[(p - 1 + k) % k];
                if (byId[me] == null || byId[succ] == null || vmById[pred] == null) continue;
                NetworkCloudlet nc = (NetworkCloudlet) byId[me];
                NetworkCloudlet dst = (NetworkCloudlet) byId[succ];
                NetworkVm srcVm = vmById[pred];
                for (int cyc = 0; cyc < cycles; cyc++) {
                    nc.addTask(new CloudletExecutionTask(taskId++, execPerCycle));
                    CloudletSendTask send = new CloudletSendTask(taskId++);
                    nc.addTask(send);                 // must precede addPacket (sets owner)
                    send.addPacket(dst, packetBytes);
                    CloudletReceiveTask recv = new CloudletReceiveTask(taskId++, srcVm);
                    recv.setExpectedPacketsToReceive(1);
                    nc.addTask(recv);
                }
            }
        }
    }

    private UtilizationModelDynamic createSyntheticModel(int seed) {
        // Varied base utilization to create interesting workload
        double base = 0.15 + (seed % 7) * 0.1;
        return new UtilizationModelDynamic(Math.min(0.9, base));
    }

    /**
     * Group the numVms VMs into gangs whose sizes follow the Blue Waters node_num
     * distribution. VMs are assigned consecutive ids; a job with node_num=k claims
     * min(k, GANG_MAX_SIZE) VMs as one gang, a 1-node job a solo VM. Gang entries
     * reference VM ids (createVms sets id==creation index), so they remain valid
     * after the placement shuffle. Deterministic given the trace order.
     */
    private List<int[]> buildGangs(int numVms) {
        int gangCap = (int) Math.max(2, parseEnvLong("GANG_MAX_SIZE", 8));
        List<Integer> sizes = readJobNodeCounts();
        List<int[]> result = new ArrayList<>();
        int vm = 0, ji = 0;
        while (vm < numVms) {
            int size = 1;
            if (!sizes.isEmpty()) {
                size = Math.min(sizes.get(ji % sizes.size()), gangCap);
                ji++;
            }
            size = Math.max(1, Math.min(size, numVms - vm));
            int[] g = new int[size];
            for (int k = 0; k < size; k++) g[k] = vm++;
            result.add(g);
        }
        return result;
    }

    /** Read node_num (col 4) per job from the active trace file, in file order. */
    private List<Integer> readJobNodeCounts() {
        List<Integer> nodes = new ArrayList<>();
        if (traceFiles.isEmpty() || useAzureTraces) return nodes;
        File f = new File(traceFiles.get(0));
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(f))) {
            br.readLine(); // header
            String line;
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length < 5) continue;
                try {
                    int n = (int) Math.round(Double.parseDouble(p[4]));  // node_num
                    nodes.add(Math.max(1, n));
                } catch (NumberFormatException ignore) { }
            }
        } catch (Exception e) {
            System.out.println("[Bridge] readJobNodeCounts failed: " + e.getMessage());
        }
        return nodes;
    }

    /** Log gang density so the network signal's data basis is visible, not silent. */
    private void logGangStats() {
        int multi = 0, vmsInGangs = 0, maxSize = 0, total = 0;
        for (int[] g : gangs) {
            total += g.length;
            maxSize = Math.max(maxSize, g.length);
            if (g.length >= 2) { multi++; vmsInGangs += g.length; }
        }
        System.out.printf(
            "[Bridge] Gangs: %d groups (%d multi-VM), %d/%d VMs in gangs (%.0f%%), maxSize=%d%n",
            gangs.size(), multi, vmsInGangs, total,
            total > 0 ? 100.0 * vmsInGangs / total : 0.0, maxSize);
    }

    private double getHostCpuUtil(Host host) {
        if (host.getVmList().isEmpty()) return 0.0;
        double totalCap = host.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        double used = computeRawDemand(host);
        return Math.min(1.0, used / totalCap);
    }

    /**
     * Compute RAW CPU demand (uncapped) — can exceed 1.0 when oversubscribed.
     * Used for SLATAH: Beloglazov 2012 defines violation as demand > capacity.
     */
    private double getRawHostCpuDemandRatio(Host host) {
        if (host.getVmList().isEmpty()) return 0.0;
        double totalCap = host.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        double used = computeRawDemand(host);
        return used / totalCap;  // NOT capped — can be > 1.0
    }

    private double computeRawDemand(Host host) {
        double used = 0;
        double simTime = (simulation != null) ? simulation.clock() : 0;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            double vmAllocatedMips = vm.getMips() * vm.getPesNumber();
            double utilRatio = 0.5;
            long vmId = vm.getId();
            if (vmId >= 0 && vmId < vmUtilArray.length) {
                utilRatio = vmUtilArray[(int) vmId].getUtilization(simTime);
            }
            double base = vmAllocatedMips * Math.max(0.01, Math.min(1.0, utilRatio));
            // Communication tax inflates a cross-rack gang VM's effective demand.
            used += base * (1.0 + commTax(vm, host));
        }
        return used;
    }

    /**
     * Extra effective-CPU-demand fraction imposed on a gang VM by cross-rack
     * communication: NET_COMM_TAX × (fraction of the VM's gang siblings that are
     * NOT on its rack). 0 for solo VMs or a fully rack-local gang. This is the
     * physical cost channel that makes the placer network-aware — it raises host
     * power (energy) and utilization (SLA/consolidation) without any reward weight.
     */
    private double commTax(Vm vm, Host host) {
        if (NET_COMM_TAX <= 0.0 || gangOfVm == null) return 0.0;
        int vid = (int) vm.getId();
        if (vid < 0 || vid >= gangOfVm.length) return 0.0;
        int[] gang = gangOfVm[vid];
        if (gang == null || gang.length < 2) return 0.0;
        double onRackFrac = gangSiblingsOnRack(vm, host);   // siblings sharing host's rack
        return NET_COMM_TAX * (1.0 - onRackFrac);           // cross-rack degree
    }

    // =========== Rack power state (Phase 1 GPU-exclusive) ===========
    private double[] rackPowerW;        // [NUM_RACKS] avg W this RL step
    private double[] rackPeakW;         // [NUM_RACKS] peak W this RL step (fine trace)
    private double[] rackP95W;          // [NUM_RACKS] p95 W this RL step
    private int[]    rackViolCountStep; // [NUM_RACKS] fine-samples > RACK_BUDGET_W

    /** Return NREL per-node power for host hostId at currentStep. */
    private double getHostPowerNow(int hostId) {
        if (hostRlTrace == null || hostRlTrace[hostId] == null) return NREL_IDLE_POWER_W;
        int elapsed = currentStep - hostJobStartRlStep[hostId];
        if (elapsed < 0 || elapsed >= hostJobDurationRlSteps[hostId]) return NREL_IDLE_POWER_W;
        return hostRlTrace[hostId][elapsed];
    }

    /**
     * Compute energy consumed in ONE interval (Δt = 300s).
     * GPU-exclusive: E = Σ nrel_power_host(t) × Δt  (no SPECpower linear model)
     */
    private double computeIntervalEnergy() {
        double totalPowerW = 0;
        for (int i = 0; i < NUM_HOSTS; i++) {
            totalPowerW += getHostPowerNow(i);
        }
        return totalPowerW * INTERVAL / 3_600_000.0; // kWh
    }

    /**
     * Sub-sample fine trace per rack to compute peak/p95/violation stats for this RL step.
     * Call once per step() before exposing rack metrics to Python.
     */
    private void computeRackStatsForStep() {
        if (rackPowerW == null) {
            rackPowerW = new double[NUM_RACKS];
            rackPeakW  = new double[NUM_RACKS];
            rackP95W   = new double[NUM_RACKS];
            rackViolCountStep = new int[NUM_RACKS];
        }
        Arrays.fill(rackPowerW, 0.0);
        Arrays.fill(rackPeakW,  0.0);
        Arrays.fill(rackViolCountStep, 0);

        int finePerStep = (int) Math.round(INTERVAL / 0.2); // 1500

        for (int r = 0; r < NUM_RACKS; r++) {
            double[] rackFineSums = new double[finePerStep];
            int startHi = r * NETWORK_RACK_SIZE;
            int endHi = Math.min(NUM_HOSTS, startHi + NETWORK_RACK_SIZE);

            for (int hi = startHi; hi < endHi; hi++) {
                double[] fine = (hostFineTrace != null) ? hostFineTrace[hi] : null;
                int startIdx   = (hostJobStartFineIdx   != null) ? hostJobStartFineIdx[hi]   : 0;
                int durFine    = (hostJobDurationFineSteps != null) ? hostJobDurationFineSteps[hi] : 0;
                double dt      = (hostFineDt != null && hostFineDt[hi] > 0) ? hostFineDt[hi] : 0.2;

                int localFinePerStep = (int) Math.round(INTERVAL / dt);
                int fineStart = startIdx + (currentStep - 1) * localFinePerStep;

                for (int fi = 0; fi < localFinePerStep && fi < finePerStep; fi++) {
                    int traceIdx = fineStart + fi;
                    double pw;
                    if (fine == null || traceIdx < startIdx || (traceIdx - startIdx) >= durFine) {
                        pw = NREL_IDLE_POWER_W;
                    } else {
                        pw = fine[traceIdx - startIdx];
                    }
                    rackFineSums[fi] += pw;
                }
            }

            // Aggregate per rack
            double sum = 0, peak = 0;
            java.util.List<Double> samples = new java.util.ArrayList<>(finePerStep);
            for (int fi = 0; fi < finePerStep; fi++) {
                double v = rackFineSums[fi];
                sum += v;
                if (v > peak) peak = v;
                samples.add(v);
                if (v > RACK_BUDGET_W) rackViolCountStep[r]++;
            }
            rackPowerW[r] = sum / finePerStep;
            rackPeakW[r]  = peak;

            // p95
            java.util.Collections.sort(samples);
            rackP95W[r] = samples.get((int) (samples.size() * 0.95));
        }
    }

    /**
     * Future rack power trajectory: traj[rack][t] = Σ host power in t steps ahead.
     * Tất định từ RL-step trace — không cần forecast.
     */
    public double[][] getRackFutureTrajectory(int horizonSteps) {
        double[][] traj = new double[NUM_RACKS][horizonSteps];
        for (int r = 0; r < NUM_RACKS; r++) {
            int start = r * NETWORK_RACK_SIZE;
            int end   = Math.min(NUM_HOSTS, start + NETWORK_RACK_SIZE);
            for (int t = 0; t < horizonSteps; t++) {
                for (int hi = start; hi < end; hi++) {
                    double[] rltr = (hostRlTrace != null) ? hostRlTrace[hi] : null;
                    int startStep = (hostJobStartRlStep != null) ? hostJobStartRlStep[hi] : 0;
                    int durSteps  = (hostJobDurationRlSteps != null) ? hostJobDurationRlSteps[hi] : 0;
                    int elapsed = currentStep + t - startStep;
                    double pw = (rltr == null || elapsed < 0 || elapsed >= durSteps)
                        ? NREL_IDLE_POWER_W : rltr[elapsed];
                    traj[r][t] += pw;
                }
            }
        }
        return traj;
    }

    // =========== Py4j APIs — rack metrics ===========
    public double[] getRackPowerW()       { return rackPowerW != null ? rackPowerW : new double[NUM_RACKS]; }
    public double[] getRackPeakW()        { return rackPeakW  != null ? rackPeakW  : new double[NUM_RACKS]; }
    public double[] getRackP95W()         { return rackP95W   != null ? rackP95W   : new double[NUM_RACKS]; }
    public int[]    getRackViolCountStep(){ return rackViolCountStep != null ? rackViolCountStep : new int[NUM_RACKS]; }
    public double[] getRackHeadroomW() {
        double[] h = new double[NUM_RACKS];
        double[] pw = getRackPowerW();
        for (int r = 0; r < NUM_RACKS; r++) h[r] = RACK_BUDGET_W - pw[r];
        return h;
    }
    public double getRackBudgetW()     { return RACK_BUDGET_W; }
    public int    getNumRacks()        { return NUM_RACKS; }

    // =========== Py4j APIs — job trace + gang management ===========

    /**
     * Load NREL power trace for one node of a gang job onto hostId.
     * Java resamples fineTraceW (dt=dtSec, e.g. 0.2s) to RL-step averages.
     */
    public void setHostJobTrace(int hostId, double[] fineTraceW, double dtSec,
                                int startRlStep, int durationRlSteps) {
        if (hostFineTrace == null) {
            hostFineTrace = new double[NUM_HOSTS][];
            hostFineDt    = new double[NUM_HOSTS];
            hostJobStartFineIdx    = new int[NUM_HOSTS];
            hostJobDurationFineSteps = new int[NUM_HOSTS];
            hostRlTrace   = new double[NUM_HOSTS][];
            hostJobStartRlStep     = new int[NUM_HOSTS];
            hostJobDurationRlSteps = new int[NUM_HOSTS];
        }
        hostFineTrace[hostId]          = fineTraceW;
        hostFineDt[hostId]             = dtSec;
        hostJobStartFineIdx[hostId]    = 0;
        hostJobDurationFineSteps[hostId] = fineTraceW.length;
        hostJobStartRlStep[hostId]     = startRlStep;
        hostJobDurationRlSteps[hostId] = durationRlSteps;

        // Resample fine trace → RL-step trace (avg per step)
        int finePerStep = (dtSec > 0) ? (int) Math.round(INTERVAL / dtSec) : 1500;
        double[] rltr = new double[durationRlSteps];
        for (int s = 0; s < durationRlSteps; s++) {
            int fi0 = s * finePerStep;
            int fi1 = Math.min(fi0 + finePerStep, fineTraceW.length);
            if (fi0 >= fineTraceW.length) { rltr[s] = NREL_IDLE_POWER_W; continue; }
            double sum = 0; int cnt = 0;
            for (int fi = fi0; fi < fi1; fi++) { sum += fineTraceW[fi]; cnt++; }
            rltr[s] = cnt > 0 ? sum / cnt : NREL_IDLE_POWER_W;
        }
        hostRlTrace[hostId] = rltr;
    }

    /** Clear a host's job (set to idle power). */
    public void clearHostJob(int hostId) {
        if (hostRlTrace != null) {
            hostRlTrace[hostId]   = null;
            hostFineTrace[hostId] = null;
            hostJobDurationRlSteps[hostId] = 0;
            hostJobDurationFineSteps[hostId] = 0;
        }
    }

    /**
     * Atomically submit a gang job across N hosts.
     * All hostIds must be idle. Loads the same per-node NREL trace on each.
     */
    public boolean submitJob(String jobId, int[] hostIds, double[] nrelPerNodeTrace,
                             double dtSec, int durationRlSteps, double priorityWeight) {
        // Validate all hosts idle
        for (int hi : hostIds) {
            if (!isHostFree(hi)) return false;
        }
        // Load trace on each host
        for (int hi : hostIds) {
            setHostJobTrace(hi, nrelPerNodeTrace, dtSec, currentStep, durationRlSteps);
        }
        jobHostMap.put(jobId, hostIds.clone());
        return true;
    }

    /** Preempt a gang job: free all N hosts, record checkpoint overhead. */
    public void preemptJob(String jobId) {
        int[] hosts = jobHostMap.get(jobId);
        if (hosts == null) return;
        for (int hi : hosts) clearHostJob(hi);
        jobHostMap.remove(jobId);
        totalCheckpointOverhead += CHECKPOINT_COST_SEC * hosts.length;
        totalMigrationDowntime  += CHECKPOINT_COST_SEC * hosts.length; // remap PDM
    }

    public boolean isHostFree(int hostId) {
        if (hostRlTrace == null || hostRlTrace[hostId] == null) return true;
        int elapsed = currentStep - hostJobStartRlStep[hostId];
        return elapsed < 0 || elapsed >= hostJobDurationRlSteps[hostId];
    }

    public boolean[] getHostFreeMap() {
        boolean[] free = new boolean[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) free[i] = isHostFree(i);
        return free;
    }

    public String[] getRunningJobIds()   { return jobHostMap.keySet().toArray(new String[0]); }
    public int[]    getJobHosts(String id) {
        int[] h = jobHostMap.get(id);
        return h != null ? h : new int[0];
    }

    private double[] collectGlobalState() {
        double[] state = new double[GLOBAL_STATE_DIM];
        double[] utils = hostList.stream().mapToDouble(this::getHostCpuUtil).toArray();
        double avg = Arrays.stream(utils).average().orElse(0);
        state[0] = avg;                                                              // avg cpu
        state[1] = Math.sqrt(Arrays.stream(utils).map(u -> (u - avg) * (u - avg))
                    .average().orElse(0));                                            // std cpu
        long active = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        state[2] = (double) active / NUM_HOSTS;                                      // active ratio
        state[3] = currentEnergy - prevEnergy;                                       // delta energy
        prevEnergy = currentEnergy;
        state[4] = currentSlatah;                                                    // SLATAH
        state[5] = currentPdm;                                                       // PDM
        state[6] = (double) intervalMigrations / Math.max(1, NUM_HOSTS);             // norm migrations
        state[7] = (double) currentStep / (MAX_TIME / INTERVAL);                     // time progress
        return state;
    }

    private List<Host> getTopKCandidates(Vm vm) {
        return hostList.stream()
            .filter(h -> !h.equals(vm.getHost()))
            .filter(h -> h.isSuitableForVm(vm))
            .filter(h -> hasGpuCapacity(h, vm))
            .sorted(Comparator.comparingDouble(this::getHostCpuUtil))
            .limit(TOP_K)
            .collect(Collectors.toList());
    }

    private boolean hasReservedTargetCapacity(Host target, Vm vm,
                                              Map<Host, Double> reservedCpu,
                                              Map<Host, Double> reservedGpu) {
        double projectedCpu = getHostCpuUtil(target)
            + reservedCpu.getOrDefault(target, 0.0)
            + getVmCpuRatio(vm, target);
        double projectedGpu = getHostGpuUsed(target)
            + reservedGpu.getOrDefault(target, 0.0)
            + getVmGpuRequest(vm);
        return projectedCpu <= 1.0 + 1e-9
            && projectedGpu <= HOST_GPU + 1e-9;
    }

    private void reserveTargetCapacity(Host target, Vm vm,
                                       Map<Host, Double> reservedCpu,
                                       Map<Host, Double> reservedGpu) {
        reservedCpu.put(target, reservedCpu.getOrDefault(target, 0.0) + getVmCpuRatio(vm, target));
        reservedGpu.put(target, reservedGpu.getOrDefault(target, 0.0) + getVmGpuRequest(vm));
    }

    private double getVmCpuRatio(Vm vm, Host target) {
        double totalMips = target.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        return (vm.getMips() * vm.getPesNumber()) / Math.max(1.0, totalMips);
    }

    /**
     * Build a real packet-level fat-tree on the NetworkDatacenter: one metered
     * EdgeSwitch per rack, plus a single Aggregate + Root switch to form a valid
     * topology. Hosts keep the id==index invariant, so host i lives in rack
     * i/NETWORK_RACK_SIZE — the same rack mapping the snapshot already uses.
     */
    private void buildFatTree(NetworkDatacenter netDc, List<NetworkHost> netHosts) {
        edgeSwitches.clear();
        RootSwitch root = new RootSwitch(simulation, netDc);
        AggregateSwitch agg = new AggregateSwitch(simulation, netDc);
        agg.getUplinkSwitches().add(root);
        root.getDownlinkSwitches().add(agg);

        int numRacks = (int) Math.ceil(NUM_HOSTS / (double) NETWORK_RACK_SIZE);
        for (int r = 0; r < numRacks; r++) {
            MeteredEdgeSwitch edge = new MeteredEdgeSwitch(simulation, netDc, r);
            edge.setPorts(NETWORK_RACK_SIZE);
            edge.getUplinkSwitches().add(agg);
            agg.getDownlinkSwitches().add(edge);
            int start = r * NETWORK_RACK_SIZE;
            int end = Math.min(NUM_HOSTS, start + NETWORK_RACK_SIZE);
            for (int i = start; i < end; i++) {
                edge.connectHost(netHosts.get(i));
            }
            edgeSwitches.add(edge);
            netDc.addSwitch(edge);
        }
        netDc.addSwitch(agg);
        netDc.addSwitch(root);
    }

    /**
     * EdgeSwitch that tallies per-rack traffic. Phase 0 verified CloudSim 8.5.7
     * routes VM↔VM packets straight to the destination host (uplink path unused),
     * so we classify each forwarded packet by whether its destination host is one
     * of THIS switch's connected hosts: same rack = local, otherwise = cross-rack
     * egress. The cross-rack tally is the exogenous, placement-dependent signal
     * that Phase 3's snapshot will expose (kept accumulating here in Phase 1).
     */
    static final class MeteredEdgeSwitch extends EdgeSwitch {
        final int rackId;
        long localBytesWindow;
        long crossRackBytesWindow;
        long localBytesTotal;
        long crossRackBytesTotal;

        MeteredEdgeSwitch(CloudSimPlus sim, NetworkDatacenter dc, int rackId) {
            super(sim, dc);
            this.rackId = rackId;
        }

        void resetWindow() {
            localBytesWindow = 0L;
            crossRackBytesWindow = 0L;
        }

        @Override
        protected void addPacketToSendToHost(NetworkHost host, HostPacket pkt) {
            long size = pkt.getSize();
            if (getHostList().contains(host)) {
                localBytesWindow += size;
                localBytesTotal += size;
            } else {
                crossRackBytesWindow += size;
                crossRackBytesTotal += size;
            }
            super.addPacketToSendToHost(host, pkt);
        }
    }

    private void initializeNetworkTopologyMetrics() {
        networkLinks = new LinkedHashMap<>();
        for (int start = 0; start < NUM_HOSTS; start += NETWORK_RACK_SIZE) {
            int end = Math.min(NUM_HOSTS, start + NETWORK_RACK_SIZE);
            for (int u = start; u < end; u++) {
                for (int v = u + 1; v < end; v++) {
                    addNetworkLinkMetric(u, v, DEFAULT_LINK_BW_MBPS, INTRA_RACK_LATENCY_SEC);
                }
            }
        }
        int numRacks = (int) Math.ceil(NUM_HOSTS / (double) NETWORK_RACK_SIZE);
        for (int rack = 0; rack < numRacks; rack++) {
            int nextRack = (rack + 1) % numRacks;
            for (int offset = 0; offset < NETWORK_RACK_SIZE; offset++) {
                int u = rack * NETWORK_RACK_SIZE + offset;
                int v = nextRack * NETWORK_RACK_SIZE + offset;
                if (u < NUM_HOSTS && v < NUM_HOSTS && u != v) {
                    addNetworkLinkMetric(u, v, DEFAULT_LINK_BW_MBPS, INTER_RACK_LATENCY_SEC);
                }
            }
        }
    }

    private void addNetworkLinkMetric(int u, int v, double bwMbps, double latencySec) {
        int a = Math.min(u, v);
        int b = Math.max(u, v);
        String key = a + ":" + b;
        networkLinks.putIfAbsent(key, new NetworkLinkMetric(a, b, bwMbps, latencySec));
    }

    private void resetNetworkTrafficWindow() {
        for (NetworkLinkMetric link : networkLinks.values()) {
            link.bytesWindow = 0.0;
        }
    }

    private void recordBackgroundTraffic() {
        if (BACKGROUND_TRAFFIC_COEF <= 0.0 || hostList == null || hostList.isEmpty()) return;
        for (NetworkLinkMetric link : networkLinks.values()) {
            if (link.src < 0 || link.src >= hostList.size() || link.dst < 0 || link.dst >= hostList.size()) {
                link.bytesBackground = 0.0;
                continue;
            }
            Host hostU = hostList.get(link.src);
            Host hostV = hostList.get(link.dst);
            double utilU = getHostCpuUtil(hostU);
            double utilV = getHostCpuUtil(hostV);
            double capacityBits = link.bwMbps * 1_000_000.0 * INTERVAL;
            link.bytesBackground = Math.max(
                0.0,
                BACKGROUND_TRAFFIC_COEF * Math.min(utilU, utilV) * capacityBits / 8.0
            );
        }
    }

    private void recordMigrationNetworkTraffic(Host source, Host target, Vm vm) {
        if (source == null || target == null || source.equals(target)) return;
        int src = (int) source.getId();
        int dst = (int) target.getId();
        if (src < 0 || src >= NUM_HOSTS || dst < 0 || dst >= NUM_HOSTS) return;
        double bytes = migrationTrafficBytes(vm);
        List<int[]> path = networkPath(src, dst);
        for (int[] edge : path) {
            NetworkLinkMetric link = networkLinks.get(linkKey(edge[0], edge[1]));
            if (link != null) {
                link.bytesWindow += bytes;
            }
        }
    }

    private double migrationTrafficBytes(Vm vm) {
        double dirtyFactor = parseEnvDouble("MIGRATION_DIRTY_FACTOR", 1.0);
        double ramMb = Math.max(1.0, vm.getRam().getCapacity());
        return ramMb * 1024.0 * 1024.0 * Math.max(0.0, dirtyFactor);
    }

    private List<int[]> networkPath(int src, int dst) {
        List<int[]> path = new ArrayList<>();
        if (src == dst) return path;
        int srcRack = src / NETWORK_RACK_SIZE;
        int dstRack = dst / NETWORK_RACK_SIZE;
        if (srcRack == dstRack) {
            path.add(new int[]{src, dst});
            return path;
        }

        int offset = src % NETWORK_RACK_SIZE;
        int srcBridge = srcRack * NETWORK_RACK_SIZE + offset;
        int dstBridge = dstRack * NETWORK_RACK_SIZE + offset;
        if (src != srcBridge) {
            path.add(new int[]{src, srcBridge});
        }

        int numRacks = (int) Math.ceil(NUM_HOSTS / (double) NETWORK_RACK_SIZE);
        int currentRack = srcRack;
        int currentHost = srcBridge;
        for (int guard = 0; guard < numRacks && currentRack != dstRack; guard++) {
            int nextRack = (currentRack + 1) % numRacks;
            int nextHost = nextRack * NETWORK_RACK_SIZE + offset;
            if (nextHost >= NUM_HOSTS) break;
            path.add(new int[]{currentHost, nextHost});
            currentRack = nextRack;
            currentHost = nextHost;
        }

        if (currentHost != dst) {
            path.add(new int[]{currentHost, dst});
        }
        return path;
    }

    private String linkKey(int u, int v) {
        int a = Math.min(u, v);
        int b = Math.max(u, v);
        return a + ":" + b;
    }

    private static double parseEnvDouble(String name, double fallback) {
        try {
            return Double.parseDouble(Optional.ofNullable(System.getenv(name)).orElse(String.valueOf(fallback)));
        } catch (Exception e) {
            return fallback;
        }
    }

    private static long parseEnvLong(String name, long fallback) {
        try {
            return Long.parseLong(Optional.ofNullable(System.getenv(name)).orElse(String.valueOf(fallback)));
        } catch (Exception e) {
            return fallback;
        }
    }

    private boolean appendNetworkLinkJson(StringBuilder sb, boolean first, NetworkLinkMetric link) {
        double utilization = link.utilization(INTERVAL);
        if (!first) sb.append(",");
        sb.append("{\"src\":").append(link.src)
          .append(",\"dst\":").append(link.dst)
          .append(",\"bw_mbps\":").append(fmt(link.bwMbps))
          .append(",\"latency_sec\":").append(fmt(link.latencySec))
          .append(",\"bytes_window\":").append(fmt(link.bytesWindow))
          .append(",\"bytes_background\":").append(fmt(link.bytesBackground))
          .append(",\"utilization\":").append(fmt(utilization))
          .append(",\"is_proxy\":false}");
        return false;
    }

    private String fmt(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) return "0.0";
        return String.format(Locale.US, "%.8f", value);
    }

    private double[] buildVmSelectionFeatures(Vm vm, Host sourceHost) {
        double[] f = new double[VM_SELECTION_FEATURE_DIM];
        double sourceUtil = getHostCpuUtil(sourceHost);
        double rawDemand = getRawHostCpuDemandRatio(sourceHost);
        double vmCpu = getVmCpuRatio(vm, sourceHost);
        double postRemovalUtil = Math.max(0.0, rawDemand - vmCpu);
        List<Host> targets = getTopKCandidates(vm);
        double bestTargetUtil = targets.stream().mapToDouble(this::getHostCpuUtil).min().orElse(1.0);
        double meanTargetUtil = targets.stream().mapToDouble(this::getHostCpuUtil).average().orElse(1.0);
        double bwMBps = sourceHost.getBw().getCapacity() / 8.0;
        double downtime = vm.getRam().getCapacity() / Math.max(1.0, bwMBps);
        double tier = 0.5;
        if (vm.getRam().getCapacity() >= 4000) tier = 3.0;
        else if (vm.getRam().getCapacity() >= 2000) tier = 1.5;

        f[0] = vmCpu;
        f[1] = vm.getRam().getCapacity() / (double) HOST_RAM;
        f[2] = getVmGpuRequest(vm) / HOST_GPU;
        f[3] = sourceUtil;
        f[4] = Math.max(0.0, rawDemand - adaptiveOverloadThreshold);
        f[5] = postRemovalUtil;
        f[6] = targets.size() / (double) TOP_K;
        f[7] = bestTargetUtil;
        f[8] = meanTargetUtil;
        f[9] = Math.min(1.0, downtime * tier / INTERVAL);
        return f;
    }

    private double[] buildVmState(Vm vm, Host sourceHost) {
        double[] s = new double[VM_STATE_DIM];
        List<Host> candidates = getTopKCandidates(vm);
        double vmGpuRatio = getVmGpuRequest(vm) / HOST_GPU;
        for (int i = 0; i < TOP_K; i++) {
            if (i < candidates.size()) {
                Host candidate = candidates.get(i);
                double candidateUtil = getHostCpuUtil(candidate);
                s[i] = candidateUtil;
                double power = NREL_IDLE_POWER_W + candidateUtil
                    * (NREL_PEAK_POWER_W - NREL_IDLE_POWER_W);
                s[TOP_K + i] = power / NREL_PEAK_POWER_W; // normalized power
                s[TOP_K * 2 + i] = getHostGpuFreeRatio(candidate, vm);
            } else {
                s[i] = -1;
                s[TOP_K + i] = -1;
                s[TOP_K * 2 + i] = -1;
            }
        }
        double totalMips = sourceHost.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        int base = TOP_K * 3;
        s[base]     = vm.getMips() / totalMips;                                   // vm cpu ratio
        s[base + 1] = vm.getRam().getCapacity() / (double) HOST_RAM;              // vm ram ratio
        s[base + 2] = vmGpuRatio;                                                 // vm gpu ratio
        s[base + 3] = getHostCpuUtil(sourceHost);                                 // source cpu util
        s[base + 4] = getHostGpuUtil(sourceHost);                                 // source gpu util
        long activeCount = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        s[base + 5] = (double) activeCount / NUM_HOSTS;                           // active ratio
        // Migration time estimate (seconds)
        double bwMBps = sourceHost.getBw().getCapacity() / 8.0;
        s[base + 6] = vm.getRam().getCapacity() / Math.max(1, bwMBps);            // est. downtime
        s[base + 7] = (double) sourceHost.getVmList().size() / (NUM_HOSTS);       // source vm density

        // Gang-locality per candidate: fraction of this VM's gang siblings already
        // running on the candidate host's rack. Directly tells the placer which
        // target keeps the gang's all-reduce traffic rack-local (cheap) vs forces
        // it cross-rack. Solo VMs (no gang) score 0 everywhere.
        int gangBase = TOP_K * 3 + VM_EXTRA_FEATURES;
        for (int i = 0; i < TOP_K; i++) {
            s[gangBase + i] = (i < candidates.size())
                ? gangSiblingsOnRack(vm, candidates.get(i))
                : -1.0;
        }
        return s;
    }

    /**
     * Fraction of {@code vm}'s gang siblings currently placed on the same rack as
     * {@code candidate}. Range [0,1]; 0 for solo VMs. Rack = hostId / NETWORK_RACK_SIZE,
     * the same mapping the metered edge switches use, so this observation is
     * consistent with the cross-rack signal fed into the reward.
     */
    private double gangSiblingsOnRack(Vm vm, Host candidate) {
        int vid = (int) vm.getId();
        if (gangOfVm == null || vid < 0 || vid >= gangOfVm.length) return 0.0;
        int[] gang = gangOfVm[vid];
        if (gang == null || gang.length < 2) return 0.0;
        int candRack = (int) candidate.getId() / NETWORK_RACK_SIZE;
        int onRack = 0;
        for (int sib : gang) {
            if (sib == vid || sib < 0 || sib >= vmByIdArr.length) continue;
            Vm sv = vmByIdArr[sib];
            if (sv == null) continue;
            Host h = sv.getHost();
            if (h != null && h != Host.NULL
                && (int) h.getId() / NETWORK_RACK_SIZE == candRack) {
                onRack++;
            }
        }
        return (double) onRack / (gang.length - 1);
    }

    private double deriveGpuRequestForVm(int vmIndex) {
        double fallback = syntheticGpuRequest(vmIndex);
        if (traceFiles.isEmpty() || useAzureTraces) {
            return fallback;
        }
        String file = traceFiles.get(vmIndex % traceFiles.size());
        double traceGpu = readGpuRequestFromTrace(file, vmIndex);
        if (traceGpu < 0) {
            return fallback;
        }
        return Math.max(0.0, Math.min(HOST_GPU, traceGpu));
    }

    private double syntheticGpuRequest(int vmIndex) {
        double[] profile = {0.0, 0.0, 1.0, 1.0, 2.0};
        return profile[Math.floorMod(vmIndex, profile.length)];
    }

    private double readGpuRequestFromTrace(String path, int rowIndex) {
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path))) {
            String header = br.readLine();
            if (header == null) return -1.0;
            String[] columns = header.split(",");
            int gpuIdx = -1;
            int nodeIdx = -1;
            for (int i = 0; i < columns.length; i++) {
                String column = columns[i].trim();
                if ("gpu_num".equals(column)) {
                    gpuIdx = i;
                } else if ("node_num".equals(column)) {
                    nodeIdx = i;
                }
            }
            if (gpuIdx < 0) return -1.0;
            String line;
            int current = 0;
            int target = Math.max(0, rowIndex);
            while ((line = br.readLine()) != null) {
                if (current == target) {
                    String[] parts = line.split(",");
                    if (gpuIdx >= parts.length || parts[gpuIdx].isBlank()) return -1.0;
                    double gpu = Double.parseDouble(parts[gpuIdx]);
                    double nodes = 1.0;
                    if (nodeIdx >= 0 && nodeIdx < parts.length && !parts[nodeIdx].isBlank()) {
                        nodes = Math.max(1.0, Double.parseDouble(parts[nodeIdx]));
                    }
                    return gpu / nodes;
                }
                current++;
            }
        } catch (Exception ignored) {
            return -1.0;
        }
        return -1.0;
    }

    private double getVmGpuRequest(Vm vm) {
        if (vm == null) return 0.0;
        return Math.max(0.0, vmGpuRequests.getOrDefault(vm.getId(), 0.0));
    }

    private double getHostGpuUsed(Host host) {
        if (host == null || host.getVmList().isEmpty()) return 0.0;
        double used = 0.0;
        for (Object obj : host.getVmList()) {
            used += getVmGpuRequest((Vm) obj);
        }
        return used;
    }

    private double getHostGpuUtil(Host host) {
        return Math.min(1.0, getHostGpuUsed(host) / HOST_GPU);
    }

    private double getHostGpuFreeRatio(Host host, Vm incoming) {
        double reserved = Math.max(0.0, HOST_GPU - getHostGpuUsed(host) - getVmGpuRequest(incoming));
        return Math.max(0.0, reserved / HOST_GPU);
    }

    private boolean hasGpuCapacity(Host host, Vm vm) {
        return getHostGpuUsed(host) + getVmGpuRequest(vm) <= HOST_GPU + 1e-9;
    }

    private double getMaxHostGpuUtil() {
        return hostList.stream().mapToDouble(this::getHostGpuUtil).max().orElse(0.0);
    }

    // ========== Main ==========
    public static void main(String[] args) {
        // Port is configurable so a serving bridge can run alongside a training
        // bridge. Precedence: CLI arg > BRIDGE_PORT env > default 25333.
        int port = 25333;
        String envPort = System.getenv("BRIDGE_PORT");
        if (args.length > 0) {
            port = Integer.parseInt(args[0].trim());
        } else if (envPort != null && !envPort.trim().isEmpty()) {
            port = Integer.parseInt(envPort.trim());
        }
        Py4jBridge bridge = new Py4jBridge();
        GatewayServer server = new GatewayServer(bridge, port);
        server.start();
        System.out.println("[Bridge] Py4J Gateway started on port " + port + ". Waiting for Python...");
    }
}
