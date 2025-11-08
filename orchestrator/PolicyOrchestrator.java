package com.example.orchestrator;

import com.example.aggregator.AggregatorServer;

public class PolicyOrchestrator {
    private AggregatorServer agg;
    private double tau_low=0.3, tau_medium=0.6, tau_high=0.8;

    public PolicyOrchestrator(AggregatorServer agg) {
        this.agg = agg;
    }

    public void evaluateNodePolicies() {
        for (String clientId : agg.getAllClientIds()) {
            double trust = agg.getTrustScore(clientId);
            if (trust < tau_low) {
                applyPolicy(clientId, "ISOLATE");
            } else if (trust < tau_medium) {
                applyPolicy(clientId, "RESTRICT");
            } else if (trust < tau_high) {
                applyPolicy(clientId, "FLAG");
            } else {
                applyPolicy(clientId, "NORMAL");
            }
        }
    }

    private void applyPolicy(String clientId, String action) {
        // invoke cloud/edge orchestration API: e.g., restrict firewall, detach from aggregator, raise alert
        System.out.printf("Applying policy %s to client %s (%f)\n", action, clientId, agg.getTrustScore(clientId));
    }
}
