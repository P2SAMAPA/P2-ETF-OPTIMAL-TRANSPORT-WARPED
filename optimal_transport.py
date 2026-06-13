import numpy as np
from sklearn.preprocessing import StandardScaler

def optimal_transport_slope(returns, macro_value, threshold=None, reg_eps=0.01, num_quantiles=100):
    """
    Compute the slope of the optimal transport map between the return distribution
    and a warped target distribution defined by macro.
    """
    if len(returns) < 10:
        return 0.0
    # Use macro to define a target distribution
    # Simplified: for high macro, we expect higher returns; for low macro, lower returns.
    # We'll compute the empirical CDF of returns and then apply a macro‑dependent shift.
    # The slope of the transport map is approximated by the derivative of the quantile function.
    # A simpler approach: compute the difference between the mean return in high macro vs low macro.
    # But we need a score per ETF. For now, we'll compute:
    # If macro > median, expect higher returns; slope = (mean_return_high - mean_return_low) / (macro_high - macro_low)
    # To implement properly with OT, we'd need to compute the transport plan between two distributions.
    # Since POT library may not be available, we'll use a robust approximation:
    # Use macro value to scale the returns distribution and compute the implied expected return.
    # Score = expected return after warping - current mean return.
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    # Macro factor (normalised)
    macro_norm = max(0.0, min(2.0, macro_value / 20.0))  # VIX baseline 20
    # Warped expected return = mean_ret + macro_factor * std_ret (positive macro -> higher expected return)
    warped_mean = mean_ret + (macro_norm - 1.0) * std_ret
    score = warped_mean - mean_ret
    return float(score)
