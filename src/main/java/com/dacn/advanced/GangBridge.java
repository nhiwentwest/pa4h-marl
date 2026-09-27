package com.dacn.advanced;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Base64;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import py4j.GatewayServer;

/**
 * Reconstructed GPU gang replay bridge. It keeps the gang simulator separate
 * from the older VM-migration CloudSim bridge. All power traces are measured
 * inputs; this class supplies placement, replay, and a documented network proxy.
 */
public final class GangBridge {
    private static final int HOSTS = Integer.parseInt(System.getenv().getOrDefault("NUM_HOSTS", "64"));
    private static final int RACK_SIZE = 4;
    private static final int RACKS = (HOSTS + RACK_SIZE - 1) / RACK_SIZE;
    private static final int GLOBAL_DIM = 8;
    private static final double INTERVAL = 300.0;
    private static final double IDLE_W = 612.0;
    private static final double BUDGET_W = 12248.0 / envDouble("RACK_OVERSUB", 1.2);
    private static final double NETWORK_TAX = envDouble("NET_COMM_TAX", 0.30);
    private static final double LINK_MBPS = envDouble("GANG_LINK_MBPS", 1000.0);
    private static final double PACKET_BYTES = envDouble("GANG_PACKET_BYTES", 1_000_000.0);
    private static final int CYCLES = Integer.parseInt(System.getenv().getOrDefault("GANG_CYCLES", "40"));
    private static final int MAX_STEPS = Integer.parseInt(System.getenv().getOrDefault("GANG_MAX_STEPS", "120"));
    private static final double BACKGROUND_UTIL = envDouble("NETWORK_BACKGROUND_UTIL", 0.0);

    private static double envDouble(String key, double fallback) {
        String value = System.getenv(key);
        return value == null ? fallback : Double.parseDouble(value);
    }

    private static final class Job {
        final int[] hosts;
        final double[] trace;
        final double dt;
        int start;
        int traceOffsetSteps;
        int duration;
        double delayedSeconds;
        int delaySteps;
        final double crossFraction;

        Job(int[] hosts, double[] trace, double dt, int start, int duration) {
            this.hosts = hosts.clone();
            this.trace = trace.clone();
            this.dt = dt;
            this.start = start;
            this.duration = duration;
            int cross = 0;
            for (int i = 0; i < hosts.length; i++) {
                if (hosts[i] / RACK_SIZE != hosts[(i + 1) % hosts.length] / RACK_SIZE) cross++;
            }
            this.crossFraction = hosts.length < 2 ? 0.0 : (double) cross / hosts.length;
        }
    }

    private static final class Link {
        final double latency;
        double bytes;

        Link(double latency) { this.latency = latency; }

        double utilization() {
            double traffic = bytes * 8.0 / (LINK_MBPS * 1_000_000.0 * INTERVAL);
            return Math.min(1.0, Math.max(0.0, traffic + BACKGROUND_UTIL));
        }
    }

    private final Map<String, Job> jobs = new LinkedHashMap<>();
    private final Map<String, Job> checkpoints = new LinkedHashMap<>();
    private final Map<String, Link> links = new LinkedHashMap<>();
    private int step;
    private double energyKwh;
    private double[] rackPower = new double[RACKS];
    private double[] rackPeak = new double[RACKS];
    private int[] rackViolations = new int[RACKS];
    private double[] network = new double[12];
    private double crossRackFraction;

    public GangBridge() { reset(); }

    public double[] reset() {
        jobs.clear();
        checkpoints.clear();
        links.clear();
        step = 0;
        energyKwh = 0.0;
        network = new double[12];
        crossRackFraction = 0.0;
        Arrays.fill(rackPower, IDLE_W * RACK_SIZE);
        Arrays.fill(rackPeak, IDLE_W * RACK_SIZE);
        Arrays.fill(rackViolations, 0);
        return new double[GLOBAL_DIM];
    }

    public int getNumHosts() { return HOSTS; }
    public int getNumRacks() { return RACKS; }
    public int getGlobalStateDim() { return GLOBAL_DIM; }
    public double getRackBudgetW() { return BUDGET_W; }
    public double getEnergyKwh() { return energyKwh; }
    public double getGangCrossRackFraction() { return crossRackFraction; }
    public double[] getGangNetworkMetrics() { return network.clone(); }
    public double[] getRackPowerW() { return rackPower.clone(); }
    public double[] getRackPeakW() { return rackPeak.clone(); }
    public int[] getRackViolCountStep() { return rackViolations.clone(); }

    public boolean[] getHostFreeMap() {
        boolean[] free = new boolean[HOSTS];
        Arrays.fill(free, true);
        for (Job job : jobs.values()) {
            for (int host : job.hosts) free[host] = false;
        }
        return free;
    }

    public boolean submitJob(String id, int[] hosts, double[] trace,
                             double dtSec, int durationSteps, double priority) {
        if (id == null || hosts == null || trace == null || hosts.length == 0
                || trace.length == 0 || dtSec <= 0 || durationSteps <= 0 || jobs.containsKey(id)) return false;
        boolean[] free = getHostFreeMap();
        boolean[] seen = new boolean[HOSTS];
        for (int host : hosts) {
            if (host < 0 || host >= HOSTS || !free[host] || seen[host]) return false;
            seen[host] = true;
        }
        Job saved = checkpoints.get(id);
        if (saved != null && saved.duration <= 0) return false;
        Job job = saved == null ? new Job(hosts, trace, dtSec, step, durationSteps)
                : new Job(hosts, saved.trace, saved.dt, step, saved.duration);
        if (saved != null) {
            job.traceOffsetSteps = saved.traceOffsetSteps;
            job.delayedSeconds = saved.delayedSeconds;
            job.delaySteps = saved.delaySteps;
            checkpoints.remove(id);
        }
        jobs.put(id, job);
        return true;
    }

    /** Transfer a measured float64 trace in one Py4J call (little-endian Base64). */
    public boolean submitJobPacked(String id, int[] hosts, String traceBase64,
                                   double dtSec, int durationSteps, double priority) {
        if (traceBase64 == null) return false;
        final byte[] bytes;
        try {
            bytes = Base64.getDecoder().decode(traceBase64);
        } catch (IllegalArgumentException invalid) {
            return false;
        }
        if (bytes.length == 0 || bytes.length % Double.BYTES != 0) return false;
        double[] trace = new double[bytes.length / Double.BYTES];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < trace.length; i++) {
            trace[i] = buffer.getDouble();
            if (!Double.isFinite(trace[i]) || trace[i] < 0.0) return false;
        }
        return submitJob(id, hosts, trace, dtSec, durationSteps, priority);
    }

    public void preemptJob(String id) {
        Job job = jobs.remove(id);
        if (job == null) return;
        int elapsed = Math.max(0, step - job.start);
        job.traceOffsetSteps += elapsed;
        job.duration = Math.max(0, job.duration - elapsed);
        job.start = step;
        checkpoints.put(id, job);
    }

    /** Replay progress and remaining simulated steps, including network extension. */
    public int[] getJobProgress(String id) {
        Job job = jobs.get(id);
        if (job != null) {
            int elapsed = Math.max(0, step - job.start);
            return new int[]{job.traceOffsetSteps + elapsed, Math.max(0, job.duration - elapsed)};
        }
        job = checkpoints.get(id);
        return job == null ? new int[]{-1, -1} : new int[]{job.traceOffsetSteps, job.duration};
    }

    public int reapFinishedJobs() {
        int before = jobs.size();
        jobs.entrySet().removeIf(e -> step >= e.getValue().start + e.getValue().duration);
        return before - jobs.size();
    }

    private double hostPower(int host, int atStep, int fineIndex) {
        for (Job job : jobs.values()) {
            if (atStep < job.start || atStep >= job.start + job.duration) continue;
            for (int occupied : job.hosts) {
                if (occupied != host) continue;
                long sample = Math.round((job.traceOffsetSteps + atStep - job.start) * INTERVAL / job.dt) + fineIndex;
                int index = (int) (sample % job.trace.length);
                return job.trace[index] * (1.0 + NETWORK_TAX * job.crossFraction);
            }
        }
        return IDLE_W;
    }

    public double[][] getRackFutureTrajectory(int horizon) {
        double[][] values = new double[RACKS][horizon];
        for (int r = 0; r < RACKS; r++) {
            for (int t = 0; t < horizon; t++) {
                for (int h = r * RACK_SIZE; h < Math.min(HOSTS, (r + 1) * RACK_SIZE); h++) {
                    values[r][t] += hostPower(h, step + t, 0);
                }
            }
        }
        return values;
    }

    private String linkKey(int from, int to) { return Math.min(from, to) + ":" + Math.max(from, to); }

    private List<String> path(int source, int target) {
        List<String> route = new ArrayList<>();
        int sourceRack = source / RACK_SIZE;
        int targetRack = target / RACK_SIZE;
        int sourceSwitch = HOSTS + sourceRack;
        int targetSwitch = HOSTS + targetRack;
        int aggregate = HOSTS + RACKS;
        route.add(linkKey(source, sourceSwitch));
        if (sourceRack != targetRack) {
            route.add(linkKey(sourceSwitch, aggregate));
            route.add(linkKey(aggregate, targetSwitch));
        }
        route.add(linkKey(targetSwitch, target));
        return route;
    }

    private double[] computeNetwork() {
        links.clear();
        double crossBytes = 0.0;
        double localBytes = 0.0;
        double totalDelay = 0.0;
        double maxJobDelay = 0.0;
        int addedSteps = 0;
        final double flowBytes = PACKET_BYTES * CYCLES;
        for (Job job : jobs.values()) {
            if (job.hosts.length < 2) continue;
            for (int i = 0; i < job.hosts.length; i++) {
                int source = job.hosts[i];
                int target = job.hosts[(i + 1) % job.hosts.length];
                boolean cross = source / RACK_SIZE != target / RACK_SIZE;
                if (cross) crossBytes += flowBytes; else localBytes += flowBytes;
                for (String key : path(source, target)) {
                    links.computeIfAbsent(key, k -> new Link(
                            k.contains(":" + (HOSTS + RACKS)) ? 0.002 : 0.0005)).bytes += flowBytes;
                }
            }
        }
        double meanUtil = 0.0;
        double maxUtil = 0.0;
        for (Link link : links.values()) {
            meanUtil += link.utilization();
            maxUtil = Math.max(maxUtil, link.utilization());
        }
        if (!links.isEmpty()) meanUtil /= links.size();
        for (Job job : jobs.values()) {
            double critical = 0.0;
            if (job.hosts.length >= 2) {
                for (int i = 0; i < job.hosts.length; i++) {
                    List<String> route = path(job.hosts[i], job.hosts[(i + 1) % job.hosts.length]);
                    double delay = 0.0;
                    for (String key : route) {
                        Link link = links.get(key);
                        delay += CYCLES * (link.latency + PACKET_BYTES * 8.0
                                / (LINK_MBPS * 1_000_000.0 * Math.max(0.01, 1.0 - link.utilization())));
                    }
                    critical = Math.max(critical, delay);
                }
            }
            job.delayedSeconds += critical;
            int newExtra = (int) Math.floor(job.delayedSeconds / INTERVAL);
            if (newExtra > job.delaySteps) {
                int difference = newExtra - job.delaySteps;
                job.duration += difference;
                addedSteps += difference;
                job.delaySteps = newExtra;
            }
            totalDelay += critical;
            maxJobDelay = Math.max(maxJobDelay, critical);
        }
        double totalBytes = crossBytes + localBytes;
        crossRackFraction = totalBytes > 0 ? crossBytes / totalBytes : 0.0;
        return new double[]{crossBytes, localBytes, totalBytes, crossRackFraction,
                meanUtil, maxUtil, totalDelay, maxJobDelay, links.size(), addedSteps, 0.0, 0.0};
    }

    public double[] stepGang() {
        network = computeNetwork();
        Arrays.fill(rackPower, 0.0);
        Arrays.fill(rackPeak, 0.0);
        Arrays.fill(rackViolations, 0);
        int fineSamples = 1500;
        for (int r = 0; r < RACKS; r++) {
            double totalPower = 0.0;
            for (int sample = 0; sample < fineSamples; sample++) {
                double power = 0.0;
                for (int host = r * RACK_SIZE; host < Math.min(HOSTS, (r + 1) * RACK_SIZE); host++) {
                    power += hostPower(host, step, sample);
                }
                totalPower += power;
                rackPeak[r] = Math.max(rackPeak[r], power);
                if (power > BUDGET_W) rackViolations[r]++;
            }
            rackPower[r] = totalPower / fineSamples;
            energyKwh += rackPower[r] * INTERVAL / 3_600_000.0;
        }
        step++;
        double[] result = new double[GLOBAL_DIM + 14];
        result[GLOBAL_DIM + 1] = step >= MAX_STEPS ? 1.0 : 0.0;
        result[GLOBAL_DIM + 3] = step;
        return result;
    }

    public static void main(String[] args) {
        int port = args.length > 0 ? Integer.parseInt(args[0])
                : Integer.parseInt(System.getenv().getOrDefault("BRIDGE_PORT", "25333"));
        GatewayServer server = new GatewayServer(new GangBridge(), port);
        server.start();
        System.out.println("[GangBridge] Listening on " + port + " with " + HOSTS + " hosts");
    }
}
