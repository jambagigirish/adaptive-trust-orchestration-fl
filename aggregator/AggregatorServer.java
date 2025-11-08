package com.example.aggregator;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class AggregatorServer {
    private Map<String, Double> trustScores = new ConcurrentHashMap<>();
    private Map<String, ClientUpdate> recentUpdates = new ConcurrentHashMap<>();
    private double gamma = 0.8;

    public AggregatorServer() {
        // initialize trust scores to 1.0 for known clients
    }

    public synchronized void receiveUpdate(String clientId,
        double[] deltaWeights, Metadata md, double localScore) {
        double prevTrust = trustScores.getOrDefault(clientId, 1.0);
        // compute weight
        double weight = clip(prevTrust, 0.1, 1.0);
        // accumulate for aggregation
        recentUpdates.put(clientId, new ClientUpdate(clientId, deltaWeights, weight, md, localScore));
    }

    public synchronized void performAggregation() {
        // compute global model update
        double[] sumWeighted = null;
        double sumWeights = 0.0;
        for (ClientUpdate cu : recentUpdates.values()) {
            if (sumWeighted == null) sumWeighted = new double[cu.delta.length];
            for (int i=0; i<cu.delta.length; i++) {
                sumWeighted[i] += cu.weight * cu.delta[i];
            }
            sumWeights += cu.weight;
        }
        double[] newGlobal = new double[sumWeighted.length];
        for (int i=0; i<sumWeighted.length; i++) {
            newGlobal[i] = sumWeighted[i] / sumWeights;
        }
        // broadcast newGlobal to all clients via REST
        // compute drift detection
        double drift = detectDrift(recentUpdates, newGlobal);
        // update trust scores
        for (ClientUpdate cu : recentUpdates.values()) {
            double newTrust = gamma * trustScores.getOrDefault(cu.clientId, 1.0)
                + (1-gamma) * scoreFunction(cu.localScore, cu.metadata, drift);
            trustScores.put(cu.clientId, newTrust);
        }
        // clear recentUpdates
        recentUpdates.clear();
    }

    private double scoreFunction(double localScore, Metadata md, double drift) {
        // simple: combine localScore and normalized divergence & drift
        double divergence = md.getUpdateNorm() / md.getExpectedNorm();
        double s = localScore * (1.0 / (1.0 + divergence)) * (1.0 / (1.0 + drift));
        return s;
    }
}

