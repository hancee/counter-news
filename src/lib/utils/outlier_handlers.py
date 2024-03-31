import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def acceptable_range(
    series,
    how="auto",
    threshold="auto",
    metric_lower_limit=None,
    metric_upper_limit=None,
):
    """
    This function takes a pandas Series object and returns the
    acceptable non-outlier range based on parameters `how` and
    `threshold`. `metric_lower_limit` and `metric_upper_limit`
    overrides initial non-outlier range.

    When `how` is set to 'auto', method is selected based on
    skew of `series`. When skew is between [-1,1], roughly
    normally distributed, 'z-score' is used. Acceptable values
    are 3 standard deviations from both side of the mean. When
    skew is not between [-1,1], more skewed, 'iqr' is used.
    Acceptable values are 1.5 * interquartile range.

    : param `series` : pandas.Series
        Input series to evaluate
    : param `how` : str in ['auto', 'iqr', 'z-score'], default
        value = 'auto'
        Defines how acceptable non-outlier range is evaluated.
        When set to 'auto', series' skew is evaluated before
        choosing between 'iqr' and 'z-score'.
    : param `threshold` : str or int or float, default value =
        'auto'
        Defines how acceptable non-outlier range is evaluated.
        When set to `auto`, default values for 'iqr' and
        'z-score' method is selected.
    : param `metric_lower_limit` : int or float, default value
        = None
        The expected minimum value for `series`. Ie: a human's
        age is expected to be at least 0.
    : param `metric_upper_limit`: int or float, default value
        = None
        The expected maximum value for `series`. Ie: a
        completion rate is expected to be at most 1.
    """

    # Identify method
    if how == "auto":
        if abs(series.skew()) <= 1:
            how = "z-score"
        else:
            how = "iqr"

    # Identify threshold
    if threshold == "auto":
        t_map = {"iqr": 1.5, "z-score": 3}
        threshold = t_map[how]

    # Get bounds
    if how == "iqr":
        p_25, p_75 = np.percentile(series, [25, 75])
        iqr = p_75 - p_25
        iqr_t = iqr * threshold
        lower, upper = p_25 - iqr_t, p_75 + iqr_t
    elif how == "z-score":
        lower = np.mean(series) - 3 * (np.std(series))
        upper = np.mean(series) + 3 * (np.std(series))

    # Adjust range if needed
    # Ie: for percent metrics like CSAT, we only expect 0% - 100%
    if metric_lower_limit is not None:
        if lower < metric_lower_limit:
            lower = metric_lower_limit  # reassign based on limit
    if metric_upper_limit is not None:
        if upper > metric_upper_limit:
            upper = metric_upper_limit  # reassign based on limit

    return lower, upper


def unskew_series(series):
    """
    This functions takes a pandas Series object and returns
    another pandas Series of each skewness reduction methods
    and their improved skewness.
    : param `series` : pandas.Series
        Input series to transform
    """

    # Without transformation
    raw_skew = series.skew()

    # Log transform
    log_series = np.log1p(series)
    log_skew = log_series.skew()

    # Squareroot transform
    sqrt_series = np.sqrt(series)
    sqrt_skew = sqrt_series.skew()

    # Boxcox-- make series positive and add a constant of 1
    constant = abs(series.min()) + 1
    boxcox_series = pd.Series(stats.boxcox(series + 1)[0])
    boxcox_skew = boxcox_series.skew()

    # Series in a dict
    transformed_series = {
        "raw skew": series,
        "log transformed skew": log_series,
        "square root transformed skew": sqrt_series,
        "boxcox transformed skew": boxcox_series,
    }

    # Skew in a dict
    summary = {
        "raw skew": raw_skew,
        "log transformed skew": log_skew,
        "square root transformed skew": sqrt_skew,
        "boxcox transformed skew": boxcox_skew,
    }

    # Find series with least skew
    best_method = pd.Series(summary).abs().sort_values().index[0]
    best_transformed_series = transformed_series[best_method]

    return best_transformed_series, pd.Series(summary)


def evaluate_series(
    series,
    how="auto",
    threshold="auto",
    metric_lower_limit=None,
    metric_upper_limit=None,
    plot=True,
):
    """
    This function returns a summary on the input series that
    can help recommend next steps for proprocessing the data.
    : param `series` : pandas.Series
        Input series to evaluate
    : param `how` : str in ['auto', 'iqr', 'z-score'], default
        value = 'auto'
        Defines how acceptable non-outlier range is evaluated.
        When set to 'auto', series' skew is evaluated before
        choosing between 'iqr' and 'z-score'.
    : param `threshold` : str or int or float, default value =
        'auto'
        Defines how acceptable non-outlier range is evaluated.
        When set to `auto`, default values for 'iqr' and
        'z-score' method is selected.
    : param `metric_lower_limit` : int or float, default value
        = None
        The expected minimum value for `series`. Ie: a human's
        age is expected to be at least 0.
    : param `metric_upper_limit`: int or float, default value =
        None
        The expected maximum value for `series`. Ie: a
        completion rate is expected to be at most 1.
    : param `plot`: boolean, default value = True
        Plots the distributions of the (a) raw input, (b)
        non-outlier range, and (c) unskewed range.
    """

    # Identify method
    skew = series.skew()
    if how == "auto":
        if abs(skew) <= 1:
            how = "z-score"
        else:
            how = "iqr"

    # Get bounds
    l, u = acceptable_range(
        series,
        how=how,
        threshold=threshold,
        metric_lower_limit=metric_lower_limit,
        metric_upper_limit=metric_upper_limit,
    )
    mask = (series >= l) & (series <= u)
    non_outliers = series[mask]

    # Basic desccriptives
    mean = np.mean(non_outliers)
    median = np.percentile(series, 50)
    mode = np.mean(stats.mode(series))
    std = np.std(non_outliers)

    # Build summary
    dist_summary = {
        "method": how,
        "mean": mean,
        "median": median,
        "mode": mode,
        "std": std,
        "lower limit": l,
        "upper limit": u,
        "pct below lower bound": (series < l).sum() / series.size,
        "pct above upper bound": (series > u).sum() / series.size,
    }
    unskewed_series, unskew_summary = unskew_series(series)

    # Plot if set to True
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        _ = fig.suptitle(series.name)
        _ = sns.distplot(series, ax=axes[0])
        _ = axes[0].set_xlabel(f"Raw (skew={np.round(skew,4)})")
        _ = sns.distplot(non_outliers, ax=axes[1])
        _ = axes[1].set_xlabel(f"Non-Outliers (skew={np.round(non_outliers.skew(),4)})")
        _ = sns.distplot(unskewed_series, ax=axes[2])
        _ = axes[2].set_xlabel(f"Unskewed (skew={np.round(unskewed_series.skew(),4)})")

    return pd.Series(dist_summary).append(unskew_summary)
