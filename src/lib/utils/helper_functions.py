#Standard libraries
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Collections
from collections import ChainMap
from itertools import chain, islice

#Text processing
import re
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#Serialization
import pickle
import csv
import json

seed = 19

###############################
### SERIALIZATION FUNCTIONS ###
###############################

def pickle_working_file(item):
    """
    This function pickles a working file to Working Files
    directory.
    : param `item` : obj
        Item to be pickles
    """
    #Get name of item
    namespace = globals()
    item_name = [name for name in namespace if namespace[name] is item][0]

    #Pickle
    fname = f'../Working Files/{item_name}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(item, f)
        f.close()

    #Print location
    print(fname)

def unpickle_working_file(item_name):
    """
    This function unpickles a working file to Working Files
    directory.
    : param `item` : obj
        Item to be pickles
    """
    #Unpickle
    fname = f'../Working Files/{item_name}.pkl'
    with open(fname, 'rb') as f:
        item = pickle.load(f)
        f.close()

    #Return item
    return item

###################################
### EDA AND WRANGLING FUNCTIONS ###
###################################

def acceptable_range(series, how='auto', threshold='auto',
                     metric_lower_limit=None, metric_upper_limit=None):
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

    #Identify method
    if how=='auto':
        if abs(series.skew())<=1:
            how='z-score'
        else:
            how='iqr'

    #Identify threshold
    if threshold=='auto':
        t_map = {'iqr' : 1.5, 'z-score' : 3}
        threshold = t_map[how]

    #Get bounds
    if how=='iqr':
        p_25, p_75 = np.percentile(series, [25,75])
        iqr = p_75 - p_25
        iqr_t = iqr * threshold
        lower, upper = p_25 - iqr_t, p_75 + iqr_t
    elif how=='z-score':
        lower = np.mean(series) - 3*(np.std(series))
        upper = np.mean(series) + 3*(np.std(series))

    #Adjust range if needed
    #Ie: for percent metrics like CSAT, we only expect 0% - 100%
    if metric_lower_limit is not None:
        if lower<metric_lower_limit:
            lower = metric_lower_limit #reassign based on limit
    if metric_upper_limit is not None:
        if upper>metric_upper_limit:
            upper = metric_upper_limit #reassign based on limit

    return lower, upper

def unskew_series(series):
    """
    This functions takes a pandas Series object and returns
    another pandas Series of each skewness reduction methods
    and their improved skewness.
    : param `series` : pandas.Series
        Input series to transform
    """

    #Without transformation
    raw_skew = series.skew()

    #Log transform
    log_series = np.log1p(series)
    log_skew = log_series.skew()

    #Squareroot transform
    sqrt_series = np.sqrt(series)
    sqrt_skew = sqrt_series.skew()

    #Boxcox-- make series positive and add a constant of 1
    constant = abs(series.min())+1
    boxcox_series = pd.Series(stats.boxcox(series+1)[0])
    boxcox_skew = boxcox_series.skew()

    #Series in a dict
    transformed_series = {
        'raw skew' : series, 'log transformed skew' : log_series
        , 'square root transformed skew' : sqrt_series, 'boxcox transformed skew' : boxcox_series}

    #Skew in a dict
    summary = {
        'raw skew' : raw_skew, 'log transformed skew' : log_skew
        , 'square root transformed skew' : sqrt_skew, 'boxcox transformed skew' : boxcox_skew
    }

    #Find series with least skew
    best_method = pd.Series(summary).abs().sort_values().index[0]
    best_transformed_series = transformed_series[best_method]

    return best_transformed_series, pd.Series(summary)

def evaluate_series(series, how='auto', threshold='auto',
                      metric_lower_limit=None, metric_upper_limit=None,
                      plot=True):
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

    #Identify method
    skew = series.skew()
    if how=='auto':
        if abs(skew)<=1:
            how='z-score'
        else:
            how='iqr'

    #Get bounds
    l, u = acceptable_range(series, how=how, threshold=threshold,
                            metric_lower_limit=metric_lower_limit,
                            metric_upper_limit=metric_upper_limit)
    mask = (series>=l) & (series<=u)
    non_outliers = series[mask]

    #Basic desccriptives
    mean = np.mean(non_outliers)
    median = np.percentile(series, 50)
    mode = np.mean(stats.mode(series))
    std = np.std(non_outliers)

    #Build summary
    dist_summary = {
        'method' : how, 'mean' : mean, 'median' : median, 'mode' : mode
        , 'std' : std, 'lower limit' : l, 'upper limit' : u
        , 'pct below lower bound' : (series<l).sum() / series.size
        , 'pct above upper bound' : (series>u).sum() / series.size
    }
    unskewed_series, unskew_summary = unskew_series(series)

    #Plot if set to True
    if plot:
        fig, axes = plt.subplots(1,3, figsize=(12,3))
        _ = fig.suptitle(series.name)
        _ = sns.distplot(series, ax=axes[0])
        _ = axes[0].set_xlabel(f'Raw (skew={np.round(skew,4)})')
        _ = sns.distplot(non_outliers, ax=axes[1])
        _ = axes[1].set_xlabel(f'Non-Outliers (skew={np.round(non_outliers.skew(),4)})')
        _ = sns.distplot(unskewed_series, ax=axes[2])
        _ = axes[2].set_xlabel(f'Unskewed (skew={np.round(unskewed_series.skew(),4)})')

    return pd.Series(dist_summary).append(unskew_summary)

def pareto_encode(series, tol=0.01, alias='Others'):
    """
    This function transforms the data (masking minority
    classes) if 80% of the samples fall within top 20% classes,
    following Pareto's 80-20 rule.
    : param `series` : pd.Series
        Input series with dtype string
    : param `tol` : float, default_value = 0.01
        Tolerance of error from 0.20 for total cumulative
        share of majority classes
    : param `alias` : string
        Imputer for minority classes
    """

    #Check if `series` has dtype string
    if series.dtype!='O':
        return series
    else:
        #Get cumulative value counts
        series_cdf = series.value_counts(normalize=True, dropna=False).cumsum()
        #Find how many classes have <= 0.8 share of total samples
        n_majority_classes = series_cdf[series_cdf<=0.8].size
        if 0.8 not in series_cdf:
            n_majority_classes += 1
        #Check if 0.8 share of total samples is distributed within 0.2 of classes
        cum_pct_majority_classes = n_majority_classes / series.nunique(dropna=False)
        #Map values based on evaluation
        if np.abs(0.20 - cum_pct_majority_classes) <= tol:
            majority_classes = series_cdf[:n_majority_classes].index.tolist()
            return series.apply(lambda x: x if x in majority_classes else alias)
        else:
            return series

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Source: https://towardsdatascience.com/heres-how-to-calculate-distance-between-2-geolocations-in-python-93ecab5bbba4
    """
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)



#################################
### TEXT PROCESSING FUNCTIONS ###
#################################

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """
    SOURCE: https://www.datacamp.com/tutorial/fuzzy-string-python
    levenshtein_ratio_and_distance:
    Calculates levenshtein distance between two strings.
    If ratio_calc = True, the function computes the
    levenshtein distance ratio of similarity between two strings
    For all i and j, distance[i,j] will contain the Levenshtein
    distance between the first i characters of s and the
    first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        #return "The strings are {} edits away".format(distance[row][col])
        return distance[row][col]

def evaluate_string_similarity(string1, string2, exclude=[]):
    """
    This function evaluates string similarity between two
    strings: `string1` and `string2`:
    : param `string1` : str
        First input string
    : param `string2` : str
        Second input string
    : param `exclude` : list, default value = []
        Metrics to exclue generally to reduce computing
        time or when not relevant
    """
    eval_dict = {}
    if 'levenshtein_ratio' not in exclude:
        eval_dict['levenshtein_ratio'] = levenshtein_ratio_and_distance(string1, string2, ratio_calc=True)
    if 'fuzz_ratio' not in exclude:
        eval_dict['fuzz_ratio'] = fuzz.ratio(string1, string2)
    if 'fuzz_partial_ratio' not in exclude:
        eval_dict['fuzz_partial_ratio'] = fuzz.partial_ratio(string1, string2)
    if 'fuzz_token_sort_ratio' not in exclude:
        eval_dict['fuzz_token_sort_ratio'] = fuzz.token_sort_ratio(string1, string2)

    return eval_dict

def filter_string_list(main_string, string_list, string_len_window=(0.75,1.5), min_char_ratio=0.6):
    """
    This functions isolates items in `string_list` that
    are likely to be similar to `main_string` based on
    length (`string_len_window`) and distinct common
    characters (`min_char_ratio`)
    : param `main_string` :
    """

    #Format
    main_string = str(main_string)
    filtered_string_list = string_list

    #Filter by string length
    min_ratio, max_ratio = string_len_window
    min_len, max_len = int(np.floor(len(main_string)*min_ratio)), int(np.ceil(len(main_string)*max_ratio))
    filtered_string_list = [m for m in filtered_string_list if (len(str(m))>=min_len) & (len(str(m))<=max_len)]

    #Filter by count of similar characters (ignores reps)
    main_string_chars = set(main_string)
    min_char_intersection = len(main_string_chars)*min_char_ratio
    filtered_string_list = [m for m in filtered_string_list if len(main_string_chars.intersection(set(str(m))))>=min_char_intersection]

    return filtered_string_list

filter_string_list_params = {
    'string_len_window':(0.75,1.5), 'min_char_ratio':0.6
}
def fuzzy_matcher(
    main_string, string_list, exclude=[]
    , threshold={'levenshtein_ratio':80, 'fuzz_ratio':80, 'fuzz_partial_ratio':80, 'fuzz_token_sort_ratio':80}
    , filter_string_list=True
    , **filter_string_list_params
):
    """

    """
    eval_dict = {}
    fxn = lambda string: evaluate_string_similarity(main_string, string, exclude=exclude)
    if filter_string_list:
        filtered_string_list = filter_string_list(main_string, string_list, **filter_string_list_params)
    else:
        filtered_string_list = string_list
    eval_dict = {m:fxn(m) for m in filtered_string_list if m!=main_string}
    eval_df = pd.DataFrame.from_dict(eval_dict, orient='index')
    filtered_df = eval_df
    for metric in eval_df.columns:
        mask = filtered_df[metric]>=threshold[metric]
        filtered_df = filtered_df[mask]
    fuzzy_matches = filtered_df.index.tolist()
    return fuzzy_matches

###########################
### STATISTICAL TESTING ###
###########################

def vif_report(data):
    #Isolate numeric and drop nulls
    data = data.select_dtypes(exclude='O').dropna()

    #VIF for each feature
    vif = {data.columns[i]:variance_inflation_factor(data.values, i) for i in range(len(data.columns))}
    vif = pd.DataFrame.from_dict(vif, orient='index', columns=['VIF'])

    return vif.sort_values('VIF', ascending=False)

def get_maj_classes(series):
    #Cumulative share per class
    cumsum = series.value_counts(normalize=True).cumsum()
    #Classes within majority
    n_class = cumsum[cumsum<0.8].size
    if 0.8 in cumsum:
        n_class += 1
    maj_classes = cumsum[:n_class].index.tolist()
    return maj_classes

def plot_cats(data, target, nunique_lim=8, n_cols=2):
    #Required params to construct plot
    items = data.select_dtypes(include='O').columns.tolist()

    #Plot subplots
    n_rows = max([i//n_cols for i in range(len(items))]) + 1
    n_subplots = n_cols * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,3*n_rows))
    _ = fig.subplots_adjust(hspace=0.4)
    for item in items:
        i = items.index(item)
        ax = axes[i//n_cols, i%n_cols]
        series = data[item]
        _ = ax.set_title(f'{item}')
        #If no of classes is within limit
        if series.nunique() <= nunique_lim:
            _ = sns.boxplot(data=data, x=item, y=target, ax=ax)
        #Plot only majority classes if beyond limit
        else:
            maj_classes = get_maj_classes(series)
            if len(maj_classes)>0:
                mask = series.isin(maj_classes)
                _ = sns.boxplot(data=data[mask], x=item, y=target, ax=ax)
            else:
                _ = ax.set_axis_off()
                dom = series.value_counts(normalize=True)[:1]
                dom_class = dom.index[0]
                dom_pct = np.round(dom.values[0]*100, 2)
                _ = ax.text(0, 0, f'Needs more checking.\nClass `{dom_class}` has {dom_pct}% share.')
                #_ = ax.text(0.5, 0.5, f'Needs further checking')


    #Despine blank subplots
    if n_subplots > len(items):
        _ = [axes[-1,-1*(i+1)].axis('off') for i in range(n_subplots-len(items))]

def highlight_abs_min(s, props=''):
    return np.where(abs(s) == np.nanmin(np.abs(s.values)), props, '')

def plot_corrs(data, target):

    #Set up plot
    #fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,5), gridspec_kw={'width_ratios': [1, 2]})
    fig, axes = plt.subplot_mosaic([['top', 'right'],['bottom', 'right']], figsize=(12,5), gridspec_kw={'width_ratios': [1, 2]})

    #Get correlation
    corr = data.corr()
    corr_cols = [c for c in corr.index.tolist() if c!=target]+[target]
    corr = corr.loc[corr_cols, corr_cols]

    #Default plot params
    _ = fig.suptitle(f'Correlations to `{target}`')
    plot_params = {'annot':True, 'fmt':'.2f', 'linewidth':0.5}

    #Vars that are negatively associated with target
    neg_ax = axes['top']
    neg_vars = [c for c in corr[corr[target]<0].index.tolist() if c!=target]+[target]
    neg_corr = corr.loc[neg_vars,neg_vars]
    neg_cmap = sns.color_palette("light:r_r", as_cmap=True)
    neg_params = {
        'data':neg_corr, 'cmap':neg_cmap, 'vmin':-1, 'vmax':0
        , 'mask':np.triu(np.ones_like(neg_corr)), 'ax':neg_ax}
    _ = sns.heatmap(**neg_params, **plot_params)
    _ = _.set(xticklabels=[])
    _ = neg_ax.set_title('Negative')

    #Vars that are positively associated with target
    pos_ax = axes['bottom']
    pos_vars = [c for c in corr[corr[target]>0].index.tolist() if c!=target]+[target]
    pos_corr = corr.loc[pos_vars,pos_vars]
    pos_cmap = sns.color_palette("light:b", as_cmap=True)
    pos_params = {
        'data':pos_corr, 'cmap':pos_cmap, 'vmin':0, 'vmax':1
        , 'mask':np.triu(np.ones_like(pos_corr)), 'ax':pos_ax}
    _ = sns.heatmap(**pos_params, **plot_params)
    _ = _.set(xticklabels=[])
    _ = pos_ax.set_title('Positive')

    #All vars
    all_ax = axes['right']
    _ = sns.heatmap(corr, ax=all_ax, mask=np.triu(np.ones_like(corr)), **plot_params)
    _ = _.set(yticklabels=[])
    _ = all_ax.set_title('All Features')

def is_not_normal(x, alpha=0.05):
    """
    This function uses Shapiro-Wilk Test to determin if there
    is sufficient evidence that `x` does not follow a Gaussian
    distribution.
    : param x : np.array or pd.Series
    """
    p = stats.shapiro(x).pvalue
    return p < alpha

def categorical_x_continuous_y_test(x, y):
    """
    This function tests intra-group differences given categorical
    `x` and continuous `y`. This returns a dictionary with keys
    ['Test', 'Statistic', 'P-Value'].

    : param `x` : np.array, pd.Series
        The categorical independent variable.

    : param `y` : np.array, pd.Series
        The continuous dependent variable.

    ===
    TO DO:
    - Add support for independent testing
    - Input validation
    - Error handling

    """


    #Identify groups in x
    x_classes = x.value_counts().index.tolist()
    args = [y[x==x_class] for x_class in x_classes]

    #No testing
    if len(x_classes)<2:
        test = ''
        s, p = np.nan, np.nan

    else:
        #Tests
        tests = {
            'MANN-WHITNEY U':stats.mannwhitneyu
            , 'KRUSKAL-WALLIS':stats.kruskal
            , 'STANDARD TWO-SAMPLE TTEST':stats.ttest_ind
            , "WELCH'S TTEST":stats.ttest_ind
            , 'ANOVA':stats.f_oneway
        }

        #Compare population variances if there's only two classes
        if len(x_classes)==2:
            equal_var = args[0].var()==args[0].var()
            kwargs = {'random_state':seed, 'equal_var':equal_var}
        else:
            kwargs = {}

        #If y does not follow a normal distribution, use mean-based methods
        if is_not_normal(y):
            #MANN-WHITNEY U if n_classes==2
            if len(x_classes)==2:
                test = 'MANN-WHITNEY U'
            #KRUSKAL-WALLIS if n_classes>2
            else:
                test = 'KRUSKAL-WALLIS'

        #If y follows a normal distribution, use median-based methods
        else:
            #TWO-SAMPLE TTEST if n_classes==2
            if len(x_classes)==2:
                #STANDARD TWO-SAMPLE TTEST if 2 x classes have equal population variance for y
                if equal_var:
                    test = 'STANDARD TWO-SAMPLE TTEST'
                #WELCH'S TTEST if 2 x classes does not have equal population variance for y
                else:
                    test = "WELCH'S TTEST"
            #ONE-WAY ANOVA if n_classes>2
            else:
                test = 'ANOVA'
        s, p = tests[test](*args, **kwargs)

    return {'Test':test, 'Statistic':s, 'P-value':p}

def test_for_independence(series_a, series_b):
    #Check if dtypes are the same
    if series_a.dtype!=series_b.dtype:
        test = ''
        s, p = np.nan, np.nan
    else:
        #CHI2 TEST if categorical
        if series_a.dtype=='O':
            test = 'CHI SQUARED'
            vc_a = series_a.value_counts().to_dict()
            vc_b = series_b.value_counts().to_dict()
            #Contingency does not expect 0 freq
            classes = sorted(set(vc_a.keys()).intersection(set(vc_b.keys())))
            #Get frequency for common classes
            freqs_a = [vc_a[c] for c in classes]
            freqs_b = [vc_b[c] for c in classes]
            #Test
            s, p = stats.chi2_contingency([freqs_a,freqs_b])[:2]
        #TTEST if numeric
        else:
            test = 'STANDARD TWO-SAMPLE TTEST'
            s, p = stats.ttest_ind(series_a.dropna(), series_b.dropna())
    return {'Test':test, 'Statistic':s, 'P-value':p}
