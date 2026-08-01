package com.dacn.advanced;

import org.cloudsimplus.utilizationmodels.UtilizationModelAbstract;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Đọc file CSV từ Gen-Parallel-Workloads (như Theta, Philly, BW).
 * Quy đổi thành chuỗi phần trăm CPU/GPU theo từng interval.
 * V3: Random Segment Sampling — each episode samples a different 288-step
 *     window from the full trace (31K+ intervals for Philly 108-day data).
 *     This preserves real autocorrelation, heavy-tail, and periodicity.
 */
public class UtilizationModelGenParallel extends UtilizationModelAbstract {
    // Cache stores the FULL normalized trace (31K+ intervals), not compressed
    private static Map<String, double[]> fullTraceCache = new HashMap<>();
    
    private double[] utilizationTrace;
    private int traceLength;
    private double interval;
    private int vmIndex; // Which VM this model belongs to

    private static final int TARGET_INTERVALS = 288; // 86400s / 300s
    private static final Random WINDOW_RNG = new Random();

    public UtilizationModelGenParallel(String csvPath, double intervalSecs) {
        this(csvPath, intervalSecs, 0);
    }
    
    public UtilizationModelGenParallel(String csvPath, double intervalSecs, int vmIndex) {
        this.interval = intervalSecs;
        this.vmIndex = vmIndex;
        
        if (fullTraceCache.containsKey(csvPath)) {
            // Cache hit: sample a NEW random window each time (each episode)
            double[] fullTrace = fullTraceCache.get(csvPath);
            double[] window = sampleRandomWindow(fullTrace);
            this.utilizationTrace = generateVmTrace(window, vmIndex);
            this.traceLength = utilizationTrace.length;
        } else {
            loadAndParseCsv(csvPath, intervalSecs);
        }
    }

    private void loadAndParseCsv(String path, double intervalSecs) {
        System.out.println("[GenParallel V3] Parsing CSV Trace: " + path);
        List<Job> jobs = new ArrayList<>();
        double currentTime = 0;
        
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] p = line.split(",");
                if (p.length < 7) continue;
                double interArrival = Double.parseDouble(p[5]);
                double runTime = Double.parseDouble(p[6]);
                currentTime += interArrival;
                
                double cpu = Double.parseDouble(p[3]);
                double gpu = Double.parseDouble(p[2]);
                double node = Double.parseDouble(p[4]);
                
                double capacity = cpu > 0 ? cpu : (gpu > 0 ? gpu : node);
                if (capacity <= 0) capacity = 1;
                
                jobs.add(new Job(currentTime, currentTime + runTime, capacity));
            }
        } catch (Exception e) {
            e.printStackTrace();
            this.utilizationTrace = generateDefaultTrace(vmIndex);
            this.traceLength = utilizationTrace.length;
            return;
        }

        if (jobs.isEmpty()) {
            this.utilizationTrace = generateDefaultTrace(vmIndex);
            this.traceLength = utilizationTrace.length;
            return;
        }

        // Create FULL master trace: bin jobs into intervals 
        double maxTime = jobs.stream().mapToDouble(j -> j.endTime).max().orElse(0);
        int rawIntervals = (int) Math.ceil(maxTime / intervalSecs) + 1;
        double[] rawBins = new double[rawIntervals];
        
        for (Job job : jobs) {
            int startIdx = Math.max(0, (int) (job.startTime / intervalSecs));
            int endIdx = Math.min(rawIntervals - 1, (int) (job.endTime / intervalSecs));
            for (int i = startIdx; i <= endIdx; i++) {
                rawBins[i] += job.capacity;
            }
        }

        // Normalize to [0,1]
        double maxCap = Arrays.stream(rawBins).max().orElse(1);
        if (maxCap == 0) maxCap = 1;
        double[] normalizedFull = new double[rawIntervals];
        for (int i = 0; i < rawIntervals; i++) {
            normalizedFull[i] = Math.min(1.0, rawBins[i] / maxCap);
        }

        // Cache the FULL trace (not compressed) for random segment sampling
        fullTraceCache.put(path, normalizedFull);
        
        int possibleWindows = Math.max(0, rawIntervals - TARGET_INTERVALS);
        System.out.println("[GenParallel V3] Parsed " + jobs.size() + " jobs, " 
                         + rawIntervals + " total intervals (" 
                         + String.format("%.1f", rawIntervals * intervalSecs / 86400.0) + " days).");
        System.out.println("[GenParallel V3] Available random windows: " + possibleWindows);
        System.out.println("[GenParallel V3] Full trace stats: avg=" 
                         + String.format("%.4f", Arrays.stream(normalizedFull).average().orElse(0))
                         + " max=" + String.format("%.4f", Arrays.stream(normalizedFull).max().orElse(0)));
        
        // Sample first window for this VM
        double[] window = sampleRandomWindow(normalizedFull);
        this.utilizationTrace = generateVmTrace(window, vmIndex);
        this.traceLength = utilizationTrace.length;
    }
    
    /**
     * Sample a random 288-step window from the full trace.
     * Each call returns a DIFFERENT segment — this is the core of
     * Random Segment Sampling for episode variation.
     */
    private double[] sampleRandomWindow(double[] raw) {
        if (raw.length <= TARGET_INTERVALS) {
            // Pad if too short (cycle)
            double[] result = new double[TARGET_INTERVALS];
            for (int i = 0; i < TARGET_INTERVALS; i++) {
                result[i] = raw[i % raw.length];
            }
            return result;
        }
        
        // Random start offset — different every episode
        int maxStart = raw.length - TARGET_INTERVALS;
        int start = WINDOW_RNG.nextInt(maxStart);
        double[] result = new double[TARGET_INTERVALS];
        System.arraycopy(raw, start, result, 0, TARGET_INTERVALS);
        return result;
    }
    
    /**
     * Generate a unique per-VM trace from the sampled window.
     * V4: Higher utilization + periodic spikes to create real overload.
     * Beloglazov 2012 uses PlanetLab traces with avg ~30-60% and spikes to 100%.
     * We replicate this pattern from Philly job-level data.
     */
    private double[] generateVmTrace(double[] master, int vmIdx) {
        Random rng = new Random(); // Non-deterministic: varies each episode
        double[] trace = new double[master.length];
        int phase = rng.nextInt(master.length); // Phase shift
        double scale = 0.8 + rng.nextDouble() * 0.7; // 0.8 to 1.5 scaling (was 0.6-1.4)
        double baseLoad = 0.3 + rng.nextDouble() * 0.2; // 0.3 to 0.5 base load (was 0.1-0.3)
        
        // Spike parameters: random burst periods simulating job arrival surges
        int spikeStart = rng.nextInt(master.length);
        int spikeDuration = 10 + rng.nextInt(30); // 10-40 intervals of high load
        double spikeIntensity = 0.85 + rng.nextDouble() * 0.15; // 0.85-1.0
        
        for (int i = 0; i < master.length; i++) {
            int srcIdx = (i + phase) % master.length;
            double val = master[srcIdx] * scale + baseLoad;
            // Add noise
            val += (rng.nextDouble() - 0.5) * 0.12;
            // Periodic spike: simulates workload burst
            int distFromSpike = Math.min(
                Math.abs(i - spikeStart),
                master.length - Math.abs(i - spikeStart));
            if (distFromSpike < spikeDuration) {
                val = Math.max(val, spikeIntensity + (rng.nextDouble() - 0.5) * 0.1);
            }
            trace[i] = Math.max(0.05, Math.min(1.0, val));
        }
        return trace;
    }
    
    private double[] generateDefaultTrace(int vmIdx) {
        Random rng = new Random(vmIdx * 42L + 7);
        double[] trace = new double[TARGET_INTERVALS];
        double base = 0.35 + rng.nextDouble() * 0.25; // 0.35-0.60 base (was 0.2-0.5)
        for (int i = 0; i < TARGET_INTERVALS; i++) {
            // Sinusoidal pattern with noise + higher amplitude
            trace[i] = base + 0.35 * Math.sin(2 * Math.PI * i / TARGET_INTERVALS) 
                      + (rng.nextDouble() - 0.5) * 0.15;
            trace[i] = Math.max(0.05, Math.min(1.0, trace[i]));
        }
        return trace;
    }

    @Override
    protected double getUtilizationInternal(double time) {
        if (traceLength == 0) return 0.3;
        int idx = (int) (time / interval);
        idx = Math.abs(idx % traceLength);
        return Math.max(0.01, Math.min(1.0, utilizationTrace[idx]));
    }

    private static class Job {
        double startTime;
        double endTime;
        double capacity;
        Job(double s, double e, double c) { startTime=s; endTime=e; capacity=c; }
    }
}
