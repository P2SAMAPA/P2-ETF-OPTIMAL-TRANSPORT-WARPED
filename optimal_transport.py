import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df, target_returns=None):
    """
    Compute a composite macro factor as a weighted sum of all macro variables.
    If target_returns is provided, weights are estimated via ridge regression.
    Otherwise, equal weights.
    """
    # Remove NaN rows
    if target_returns is not None:
        mask = ~(np.isnan(target_returns) | np.isnan(macro_df).any(axis=1))
        macro_clean = macro_df[mask]
        target_clean = target_returns[mask]
    else:
        macro_clean = macro_df
        target_clean = None
    if target_clean is not None and len(target_clean) > 5:
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_clean)
        ridge = Ridge(alpha=1.0)
        ridge.fit(macro_scaled, target_clean)
        weights = np.abs(ridge.coef_)
        weights = weights / (weights.sum() + 1e-8)
    else:
        weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
    return weights, scaler

def composite_macro_factor_at_time(macro_row, weights, scaler):
    """Compute composite macro factor for a single row of macro data."""
    macro_scaled = scaler.transform(macro_row.reshape(1, -1)).flatten()
    factor = np.dot(weights, macro_scaled)
    # Normalise to [0,1] range using logistic
    return 1.0 / (1.0 + np.exp(-factor))

def optimal_transport_slope(returns, macro_df, reg_eps=0.01, num_quantiles=100):
    """
    Compute the slope of the optimal transport map between the return distribution
    and a warped target distribution defined by macro.
    Uses all macro variables to compute a composite factor.
    """
    if len(returns) < 10 or macro_df is None or len(macro_df) < 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Compute macro weights using ridge regression of returns on macro (lagged)
    target = returns[1:]
    macro_lagged = macro_df.iloc[:-1]
    if len(target) != len(macro_lagged):
        min_len2 = min(len(target), len(macro_lagged))
        target = target[:min_len2]
        macro_lagged = macro_lagged.iloc[:min_len2]
    if len(target) > 5:
        weights, scaler = compute_composite_macro_factor(macro_lagged, target)
    else:
        weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
        scaler = StandardScaler()
        scaler.fit(macro_df)
    # Current macro factor (last row)
    current_macro = macro_df.iloc[-1].values
    macro_factor = composite_macro_factor_at_time(current_macro, weights, scaler)
    # Warped expected return = mean_ret + (macro_factor - 0.5) * 2 * std_ret
    # This scales the shift between -std and +std based on macro factor
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    # Shift factor: macro_factor from 0 to 1 maps to -1 to 1 shift
    shift = (macro_factor - 0.5) * 2
    warped_mean = mean_ret + shift * std_ret
    score = warped_mean - mean_ret
    return float(score)
