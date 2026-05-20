import scipy as sp
from statsmodels.stats.anova import AnovaRM
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def AnovaRM_with_post_hoc(data, dep_var, subject, within, only_significant = False):
    # One within
    anova = AnovaRM(data, dep_var, subject, within)
    print(anova.fit())
    # Post-hoc with ttest
    pairwise_ttest_rel(data,
                       dep_var,
                       within = within,
                       only_significant = only_significant
                      )        


def pairwise_ttest_rel(data, dep_var, within, only_significant = False, only_first_within_comprisons = True):
    # ttest related measures - One indep_var
    if len(within) == 1:
        conditions = data[within[0]].unique()
        list_of_ttests = list(itertools.combinations(conditions, 2))
    elif len(within) == 2:
        list1 = data[within[0]].unique()
        list2 = data[within[1]].unique()
        list_product = list(itertools.product(list1,list2))
        list_of_ttests = list(itertools.combinations(list_product, 2))
        
    print("             Post Hoc inter {}\n==========================================================================".format(' and '.join(within)))
    print("{:<48}{:>12} {:>12}".format('Test', 'p-value', 't-value'))
    indep_var = within[0]
    for combination_of_conditions in list_of_ttests:
        if len(within) == 1:
            query1 = indep_var + "==" + "'" + combination_of_conditions[0] + "'"
            query2 = indep_var + "==" + "'" + combination_of_conditions[1] + "'"
            at_least_one_same_cond = True
        elif len(within) == 2:
            at_least_one_same_cond = (combination_of_conditions[0][0] == combination_of_conditions[1][0]) or (combination_of_conditions[0][1] == combination_of_conditions[1][1])
            other_indep_var = within[1]
            query1 = indep_var + "==" "'" + combination_of_conditions[0][0] + "' & " + other_indep_var + "==" "'" + combination_of_conditions[0][1] + "'"
            query2 = indep_var + "==" "'" + combination_of_conditions[1][0] + "' & " + other_indep_var + "==" "'" + combination_of_conditions[1][1] + "'"
        
        if at_least_one_same_cond and only_first_within_comprisons:
            ttest = sp.stats.ttest_rel(data.query(query1)[dep_var],
                                        data.query(query2)[dep_var])
            
            if len(within) == 1:
                sep = ''
            elif len(within) == 2:
                sep = ' '

            if ttest.pvalue <= 0.05:
                print("\033[91m{:>22} VS {:<22}{:>12.3f}{:>12.3f}\033[0m".format(sep.join(combination_of_conditions[0]), sep.join(combination_of_conditions[1]),
                                                                   ttest.pvalue,ttest.statistic))
            elif not only_significant:
                print("{:>22} VS {:<22}{:>12.3f}{:>12.3f}".format(sep.join(combination_of_conditions[0]), sep.join(combination_of_conditions[1]),
                                                                   ttest.pvalue,ttest.statistic))


    print("==========================================================================\n\n")

    

def remove_outliers(df, columns = ['all'], zscore = 3, dropna = False):
    new_df = pd.DataFrame()
    for colName, colData in df.iteritems():
        if columns == ['all']:
            outliers = np.abs(sp.stats.zscore(colData)) < zscore
            data_without_outliers = colData[outliers]
            new_df = new_df.assign(**{colName : data_without_outliers})
        else:
            if colName in columns:
                outliers = np.abs(sp.stats.zscore(colData,nan_policy='omit')) < zscore
                data_without_outliers = colData[outliers]
                new_df = new_df.assign(**{colName : data_without_outliers})
            else:
                new_df = new_df.assign(**{colName : colData})
    if dropna:
        new_df = new_df.dropna(subset = columns)
    return new_df

def remove_outliers_per_condition(df, columns = ['all'], condition = 'condition', zscore = 3, detailed = True):
    clean_df = []
    description = []
    for cond in df[condition].unique():
        data = df.query("{} == @cond".format(condition))
        data_without_outliers = remove_outliers(
            df = data,
            columns = columns,
            zscore = zscore,
            dropna = True,
            )
        if detailed:
            n_outliers = len(data) - len(data_without_outliers)
            n_non_outliers = len(data_without_outliers)
            percent_outliers = (len(data) - len(data_without_outliers))/len(data_without_outliers)*100
            description.append({
                condition : cond,
                "Observations" : len(data),
                "Outliers" : n_outliers,
                "Non-outliers" : n_non_outliers,
                "% of outliers" : percent_outliers
            })
            # print("-"*30)
            # print("\033[1m" + "Condition: {}".format(cond) + "\033[0m")
            # print('\tIdentified outliers: {}'.format(len(data) - len(data_without_outliers)))
            # print('\tNon-outlier observations: {}'.format(len(data_without_outliers)))
            # print('\t{:.1f} % of outliers'.format(
            #     (len(data) - len(data_without_outliers))/len(data_without_outliers)*100))
        clean_df.append(data_without_outliers)
        res_description = pd.DataFrame(description).round({"% of outliers" : 2}).set_index(condition)
    return pd.concat(clean_df), res_description