// common/TrustScore.java
// Immutable-ish POJO + utilities for trust scoring and clipping.
// Java 11+ recommended.

package common;

import java.util.Objects;

public final class TrustScore {

    private final String clientId;
    private final double value;       // current trust value in [0,1]
    private final long updatedAtMs;   // epoch millis when last updated

    public TrustScore(String clientId, double value, long updatedAtMs) {
        this.clientId = Objects.requireNonNull(clientId, "clientId");
        this.value = clamp01(value);
        this.updatedAtMs = updatedAtMs;
    }

    public String getClientId() { return clientId; }
    public double getValue() { return value; }
    public long getUpdatedAtMs() { return updatedAtMs; }

    public TrustScore withValue(double newValue) {
        return new TrustScore(this.clientId, clamp01(newValue), System.currentTimeMillis());
    }

    public TrustScore decay(double gamma) {
        // gamma in [0,1]; closer to 1.0 means slower decay (more inertia)
        double g = Math.max(0.0, Math.min(1.0, gamma));
        double decayed = g * this.value;
        return new TrustScore(this.clientId, decayed, System.currentTimeMillis());
    }

    // -----------------------------
    // Static utilities
    // -----------------------------

    public static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    public static double clip(double v, double min, double max) {
        return Math.max(min, Math.min(max, v));
    }

    /**
     * Combines a client's local signal with system-level drift and metadata.
     * This mirrors the paper's f(S_local, metadata, drift) mapping into [0,1].
     *
     * @param localScore  S_i^{local} in [0,1] (e.g., from anomaly & reliability)
     * @param updateNorm  observed update norm (higher could be suspicious)
     * @param expectedNorm baseline expected norm for this client/model
     * @param drift       system-wide drift indicator in [0, +inf) (0 = no drift)
     * @return normalized score in [0,1]
     */
    public static double scoreFunction(double localScore, double updateNorm, double expectedNorm, double drift) {
        double ls = clamp01(localScore);
        double div = updateNorm / (expectedNorm + 1e-6);
        double divPenalty = 1.0 / (1.0 + div);      // bigger divergence -> smaller factor
        double driftPenalty = 1.0 / (1.0 + drift);  // more drift -> smaller factor
        double s = ls * divPenalty * driftPenalty;
        return clamp01(s);
    }

    /**
     * Exponential moving average trust update:
     *   T_i(t) = gamma * T_i(t-1) + (1-gamma) * score
     */
    public static double emaUpdate(double prevTrust, double score, double gamma) {
        double g = clip(gamma, 0.0, 1.0);
        double t = g * clamp01(prevTrust) + (1.0 - g) * clamp01(score);
        return clamp01(t);
    }

    /**
     * Convert trust value into an aggregation weight in [wMin, wMax].
     * Often used as FedAvg weight for the client's update.
     */
    public static double toAggWeight(double trust, double wMin, double wMax) {
        double t = clamp01(trust);
        if (wMax < wMin) {
            double tmp = wMin;
            wMin = wMax;
            wMax = tmp;
        }
        return wMin + (wMax - wMin) * t;
    }

    @Override
    public String toString() {
        return "TrustScore{" +
                "clientId='" + clientId + '\'' +
                ", value=" + value +
                ", updatedAtMs=" + updatedAtMs +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TrustScore)) return false;
        TrustScore that = (TrustScore) o;
        return Double.compare(that.value, value) == 0 &&
                updatedAtMs == that.updatedAtMs &&
                clientId.equals(that.clientId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(clientId, value, updatedAtMs);
    }
}

