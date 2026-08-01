package com.dacn.advanced;

import org.cloudsimplus.utilizationmodels.UtilizationModelAbstract;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

/**
 * Reads pre-processed Azure VM CPU trace files (one value per line, 288 lines).
 * Each line = avg CPU utilization as fraction [0, 1] for a 5-minute interval.
 * 
 * V2: Added ±5% noise per reading for realistic variation between episodes.
 * Created by azure_to_cloudsim.py from Azure Public Dataset V2.
 */
public class UtilizationModelAzure extends UtilizationModelAbstract {
    private double[] utilizationTrace;
    private int traceLength;
    private double interval;
    private final Random noise = new Random();

    private static final int TARGET_INTERVALS = 288;

    public UtilizationModelAzure(String csvPath, double intervalSecs) {
        this.interval = intervalSecs;
        loadTrace(csvPath);
    }

    private void loadTrace(String path) {
        List<Double> values = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // Handle CSV format: may have quotes or just a number
                String val = line.replace("\"", "").split(",")[0];
                values.add(Double.parseDouble(val));
            }
        } catch (Exception e) {
            System.err.println("[AzureModel] Error reading " + path + ": " + e.getMessage());
            values.clear();
        }

        if (values.isEmpty()) {
            // Fallback: constant 30% utilization
            utilizationTrace = new double[TARGET_INTERVALS];
            Arrays.fill(utilizationTrace, 0.3);
        } else {
            utilizationTrace = new double[TARGET_INTERVALS];
            for (int i = 0; i < TARGET_INTERVALS; i++) {
                utilizationTrace[i] = values.get(i % values.size());
                utilizationTrace[i] = Math.max(0.01, Math.min(1.0, utilizationTrace[i]));
            }
        }
        traceLength = utilizationTrace.length;
    }

    @Override
    protected double getUtilizationInternal(double time) {
        if (traceLength == 0) return 0.3;
        int idx = (int) (time / interval);
        idx = Math.abs(idx % traceLength);
        // ±5% noise per reading — models measurement noise and natural variation
        double val = utilizationTrace[idx] + (noise.nextDouble() - 0.5) * 0.10;
        return Math.max(0.01, Math.min(1.0, val));
    }
}
