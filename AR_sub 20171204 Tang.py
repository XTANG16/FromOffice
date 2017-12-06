from sklearn import ensemble
import os
import itertools
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from math import sqrt
from math import e
from math import exp
from math import log
import operator
from sklearn.metrics import r2_score
import time
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm
from copy import deepcopy
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn import feature_selection as fs
from sklearn import decomposition
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
import random
import ast

class GeneralSolver:
    def __init__(self):
        # <editor-fold desc="Variable Declare">
        # 2017-11-18 Harry: cleared duplicated declares

        # these meta-param vectors control the search over all solver runs
        self.fractions = [0.6]
        self.counters = [9]
        prob_shifts = [1]
        n_estimators = [150]
        max_depth = [2]
        max_features = [2]
        min_samples_leaf = [2]
        min_samples_split = [50]  # from program 1 result: 500, 1000, 1500 does not matter much
        learning_rate = [0.05]
        subsample = [0.5]

        # create all combos of these meta-params
        self.iterables = [prob_shifts, n_estimators, max_depth, max_features,
                     min_samples_leaf, min_samples_split, learning_rate, subsample]
        self.input_features = ['DCF_REMAILH', 'D_afil_bank_indH', 'LD_ltv_pct', 'cf_curr_cltv', 'ld_ltv_max_6m']
        # self.combos = itertools.product(*iterables)

        self.auc_OOTV_1_list = []
        self.auc_test_list = []
        self.auc_total_train_list = []
        self.counter_list = []
        self.fraction_list = []
        self.k_columns_full_list = []
        self.k_list = []
        self.ks_mape_OOTV_1_list = []
        self.ks_mape_test_list = []
        self.ks_mape_total_train_list = []
        self.ks_OOTV_1_list = []
        self.ks_ratio_OOTV_1_list = []
        self.ks_ratio_test_list = []
        self.ks_test_list = []
        self.ks_total_train_list = []
        self.learning_rate_list = []
        self.max_depth_list = []
        self.max_feature_list = []
        self.min_samples_leaf_list = []
        self.min_samples_split_list = []
        self.n_est_list = []
        self.prob_shift_list = []
        self.psi_ITV_OOTV_1_list = []
        self.psi_OOTV_1_list = []
        self.psi_test_list = []
        self.rank_order_test_list = []
        self.rank_order_OOTV_1_list = []
        self.rank_order_total_train_list = []
        self.subsample_list = []
        self.ver1_ks_mape_OOTV_1_list = []
        self.ver1_ks_mape_test_list = []
        self.ver1_ks_OOTV_1_list = []
        self.ver1_ks_test_list = []
        # </editor-fold>
    # end of function


    def ks_max_mape(self,data, scr, resp, wgt):
        data['m'] = 1
        dec_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sample = pd.DataFrame(data.pred_prob.quantile(dec_range))

        sample1 = sample.transpose()
        sample1['m'] = 1

        sample2 = pd.merge(data, sample1, on='m', how='inner')

        sample2.loc[(sample2['pred_prob'] > sample2[0.9]), 'decile'] = 1
        sample2.loc[(sample2['pred_prob'] > sample2[0.8]) & (sample2['pred_prob'] <= sample2[0.9]), 'decile'] = 2
        sample2.loc[(sample2['pred_prob'] > sample2[0.7]) & (sample2['pred_prob'] <= sample2[0.8]), 'decile'] = 3
        sample2.loc[(sample2['pred_prob'] > sample2[0.6]) & (sample2['pred_prob'] <= sample2[0.7]), 'decile'] = 4
        sample2.loc[(sample2['pred_prob'] > sample2[0.5]) & (sample2['pred_prob'] <= sample2[0.6]), 'decile'] = 5
        sample2.loc[(sample2['pred_prob'] > sample2[0.4]) & (sample2['pred_prob'] <= sample2[0.5]), 'decile'] = 6
        sample2.loc[(sample2['pred_prob'] > sample2[0.3]) & (sample2['pred_prob'] <= sample2[0.4]), 'decile'] = 7
        sample2.loc[(sample2['pred_prob'] > sample2[0.2]) & (sample2['pred_prob'] <= sample2[0.3]), 'decile'] = 8
        sample2.loc[(sample2['pred_prob'] > sample2[0.1]) & (sample2['pred_prob'] <= sample2[0.2]), 'decile'] = 9
        sample2.loc[(sample2['pred_prob'] <= sample2[0.1]), 'decile'] = 10

        sample2['wt_cnt_app'] = sample2.wt * sample2.cnt_app
        sample2['wt_1-cnt_app'] = sample2.wt * (1 - sample2.cnt_app)

        decile_data = pd.DataFrame({'min_score': sample2.groupby('decile')['pred_prob'].min(),
                                    'mean_score': sample2.groupby('decile')['pred_prob'].mean(),
                                    'max_score': sample2.groupby('decile')['pred_prob'].max(),
                                    'goods_wtd': sample2.groupby('decile')['wt_cnt_app'].sum(),
                                    'bads_wtd': sample2.groupby('decile')['wt_1-cnt_app'].sum(),
                                    'total_wtd': sample2.groupby('decile')['cnt_app'].count(),
                                    'm': 1
                                    })

        sums = pd.DataFrame()
        sums['tot_goods'] = [decile_data.goods_wtd.sum()]
        sums['tot_bads'] = [decile_data.bads_wtd.sum()]
        sums['tot_all'] = sums['tot_goods'] + sums['tot_bads']
        sums['m'] = 1

        ks_wtd = pd.merge(decile_data, sums, on='m', how='inner')

        ks_wtd1 = ks_wtd
        ks_wtd1['pct_goods'] = ks_wtd1['goods_wtd'] / ks_wtd1['tot_goods']
        ks_wtd1['pct_bads'] = ks_wtd1['bads_wtd'] / ks_wtd1['tot_bads']
        ks_wtd1['pct_accts'] = ks_wtd1['total_wtd'] / ks_wtd1['tot_all']
        ks_wtd1['cum_pct_goods'] = ks_wtd1.pct_goods.cumsum()
        ks_wtd1['cum_pct_bads'] = ks_wtd1.pct_bads.cumsum()
        ks_wtd1['cum_pct_accts'] = ks_wtd1.pct_accts.cumsum()
        ks_wtd1['ks'] = (ks_wtd1['cum_pct_goods'] - ks_wtd1['cum_pct_bads']) * 100
        ks_wtd1['odds_rnk'] = ks_wtd1.tot_goods / ks_wtd1.tot_bads

        ks_max = ks_wtd1['ks'].max()

        ks_wtd1['actual_response_rate'] = ks_wtd1.goods_wtd / ks_wtd1.total_wtd
        ks_wtd1['mape'] = abs(ks_wtd1.mean_score - ks_wtd1.actual_response_rate) / ks_wtd1.actual_response_rate
        ks_mape = ks_wtd1['mape'].mean()
        act_res_rate = list(ks_wtd1['actual_response_rate'])

        return (ks_max, ks_mape, act_res_rate)
    # end of function


    def ks_max_mape_ver1(self,data, data_dv, scr, resp, wgt):
        data['m'] = 1
        dec_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        sample = pd.DataFrame(data.pred_prob.quantile(dec_range))
        sample1 = sample.transpose()
        sample1['m'] = 1
        sample2 = pd.merge(data, sample1, on='m', how='inner')

        sampledv = pd.DataFrame(data_dv.pred_prob.quantile(dec_range))
        sampledv1 = sampledv.transpose()
        sampledv1['m'] = 1
        sampledv2 = pd.merge(data, sampledv1, on='m', how='inner')

        sample2.loc[(sample2['pred_prob'] > sampledv2[0.9]), 'decile'] = 1
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.8]) & (sample2['pred_prob'] <= sampledv2[0.9]), 'decile'] = 2
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.7]) & (sample2['pred_prob'] <= sampledv2[0.8]), 'decile'] = 3
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.6]) & (sample2['pred_prob'] <= sampledv2[0.7]), 'decile'] = 4
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.5]) & (sample2['pred_prob'] <= sampledv2[0.6]), 'decile'] = 5
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.4]) & (sample2['pred_prob'] <= sampledv2[0.5]), 'decile'] = 6
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.3]) & (sample2['pred_prob'] <= sampledv2[0.4]), 'decile'] = 7
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.2]) & (sample2['pred_prob'] <= sampledv2[0.3]), 'decile'] = 8
        sample2.loc[(sample2['pred_prob'] > sampledv2[0.1]) & (sample2['pred_prob'] <= sampledv2[0.2]), 'decile'] = 9
        sample2.loc[(sample2['pred_prob'] <= sampledv2[0.1]), 'decile'] = 10

        sample2['wt_cnt_app'] = sample2.wt * sample2.cnt_app
        sample2['wt_1-cnt_app'] = sample2.wt * (1 - sample2.cnt_app)

        decile_data = pd.DataFrame({'min_score': sample2.groupby('decile')['pred_prob'].min(),
                                    'mean_score': sample2.groupby('decile')['pred_prob'].mean(),
                                    'max_score': sample2.groupby('decile')['pred_prob'].max(),
                                    'goods_wtd': sample2.groupby('decile')['wt_cnt_app'].sum(),
                                    'bads_wtd': sample2.groupby('decile')['wt_1-cnt_app'].sum(),
                                    'total_wtd': sample2.groupby('decile')['cnt_app'].count(),
                                    'm': 1
                                    })

        sums = pd.DataFrame()
        sums['tot_goods'] = [decile_data.goods_wtd.sum()]
        sums['tot_bads'] = [decile_data.bads_wtd.sum()]
        sums['tot_all'] = sums['tot_goods'] + sums['tot_bads']
        sums['m'] = 1

        ks_wtd = pd.merge(decile_data, sums, on='m', how='inner')

        ks_wtd1 = ks_wtd
        ks_wtd1['pct_goods'] = ks_wtd1['goods_wtd'] / ks_wtd1['tot_goods']
        ks_wtd1['pct_bads'] = ks_wtd1['bads_wtd'] / ks_wtd1['tot_bads']
        ks_wtd1['pct_accts'] = ks_wtd1['total_wtd'] / ks_wtd1['tot_all']
        ks_wtd1['cum_pct_goods'] = ks_wtd1.pct_goods.cumsum()
        ks_wtd1['cum_pct_bads'] = ks_wtd1.pct_bads.cumsum()
        ks_wtd1['cum_pct_accts'] = ks_wtd1.pct_accts.cumsum()
        ks_wtd1['ks'] = (ks_wtd1['cum_pct_goods'] - ks_wtd1['cum_pct_bads']) * 100
        ks_wtd1['odds_rnk'] = ks_wtd1.tot_goods / ks_wtd1.tot_bads

        ks_max = ks_wtd1['ks'].max()

        ks_wtd1['actual_response_rate'] = ks_wtd1.goods_wtd / ks_wtd1.total_wtd
        ks_wtd1['mape'] = abs(ks_wtd1.mean_score - ks_wtd1.actual_response_rate) / ks_wtd1.actual_response_rate
        ks_mape = ks_wtd1['mape'].mean()
        act_res_rate = list(ks_wtd1['actual_response_rate'])

        return (ks_max, ks_mape, act_res_rate)
    # end of function


    def psia(self,data, data_dv, scr, resp, wgt):
        data['m'] = 1
        dec_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        sample = pd.DataFrame(data.pred_prob.quantile(dec_range))
        sample1 = sample.transpose()
        sample1['m'] = 1
        sample2 = pd.merge(data, sample1, on='m', how='inner')

        sampledv = pd.DataFrame(data_dv.pred_prob.quantile(dec_range))
        sampledv1 = sampledv.transpose()
        sampledv1['m'] = 1
        sampledv2a = pd.merge(data, sampledv1, on='m', how='inner')

        sampledv2 = pd.merge(data_dv, sampledv1, on='m', how='inner')

        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.9]), 'decile'] = 1
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.8]) & (sample2['pred_prob'] <= sampledv2a[0.9]), 'decile'] = 2
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.7]) & (sample2['pred_prob'] <= sampledv2a[0.8]), 'decile'] = 3
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.6]) & (sample2['pred_prob'] <= sampledv2a[0.7]), 'decile'] = 4
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.5]) & (sample2['pred_prob'] <= sampledv2a[0.6]), 'decile'] = 5
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.4]) & (sample2['pred_prob'] <= sampledv2a[0.5]), 'decile'] = 6
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.3]) & (sample2['pred_prob'] <= sampledv2a[0.4]), 'decile'] = 7
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.2]) & (sample2['pred_prob'] <= sampledv2a[0.3]), 'decile'] = 8
        sample2.loc[(sample2['pred_prob'] > sampledv2a[0.1]) & (sample2['pred_prob'] <= sampledv2a[0.2]), 'decile'] = 9
        sample2.loc[(sample2['pred_prob'] <= sampledv2a[0.1]), 'decile'] = 10

        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.9]), 'decile'] = 1
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.8]) & (sampledv2['pred_prob'] <= sampledv2[0.9]), 'decile'] = 2
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.7]) & (sampledv2['pred_prob'] <= sampledv2[0.8]), 'decile'] = 3
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.6]) & (sampledv2['pred_prob'] <= sampledv2[0.7]), 'decile'] = 4
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.5]) & (sampledv2['pred_prob'] <= sampledv2[0.6]), 'decile'] = 5
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.4]) & (sampledv2['pred_prob'] <= sampledv2[0.5]), 'decile'] = 6
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.3]) & (sampledv2['pred_prob'] <= sampledv2[0.4]), 'decile'] = 7
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.2]) & (sampledv2['pred_prob'] <= sampledv2[0.3]), 'decile'] = 8
        sampledv2.loc[(sampledv2['pred_prob'] > sampledv2[0.1]) & (sampledv2['pred_prob'] <= sampledv2[0.2]), 'decile'] = 9
        sampledv2.loc[(sampledv2['pred_prob'] <= sampledv2[0.1]), 'decile'] = 10

        sample2['wt_cnt_app'] = sample2.wt * sample2.cnt_app
        sample2['wt_1-cnt_app'] = sample2.wt * (1 - sample2.cnt_app)

        sampledv2['wt_cnt_app'] = sampledv2.wt * sampledv2.cnt_app
        sampledv2['wt_1-cnt_app'] = sampledv2.wt * (1 - sampledv2.cnt_app)

        decile_data = pd.DataFrame({'min_score': sample2.groupby('decile')['pred_prob'].min(),
                                    'mean_score': sample2.groupby('decile')['pred_prob'].mean(),
                                    'max_score': sample2.groupby('decile')['pred_prob'].max(),
                                    'goods_wtd': sample2.groupby('decile')['wt_cnt_app'].sum(),
                                    'bads_wtd': sample2.groupby('decile')['wt_1-cnt_app'].sum(),
                                    'total_wtd': sample2.groupby('decile')['cnt_app'].count(),
                                    'm': 1
                                    })

        decile_datadv = pd.DataFrame({'min_score': sampledv2.groupby('decile')['pred_prob'].min(),
                                      'mean_score': sampledv2.groupby('decile')['pred_prob'].mean(),
                                      'max_score': sampledv2.groupby('decile')['pred_prob'].max(),
                                      'goods_wtd': sampledv2.groupby('decile')['wt_cnt_app'].sum(),
                                      'bads_wtd': sampledv2.groupby('decile')['wt_1-cnt_app'].sum(),
                                      'total_wtd': sampledv2.groupby('decile')['cnt_app'].count(),
                                      'm': 1
                                      })

        sums = pd.DataFrame()
        sums['tot_goods'] = [decile_data.goods_wtd.sum()]
        sums['tot_bads'] = [decile_data.bads_wtd.sum()]
        sums['tot_all'] = sums['tot_goods'] + sums['tot_bads']
        sums['m'] = 1

        sumsdv = pd.DataFrame()
        sumsdv['tot_goods'] = [decile_datadv.goods_wtd.sum()]
        sumsdv['tot_bads'] = [decile_datadv.bads_wtd.sum()]
        sumsdv['tot_all'] = sums['tot_goods'] + sums['tot_bads']
        sumsdv['m'] = 1

        ks_wtd = pd.merge(decile_data, sums, on='m', how='inner')
        ks_wtddv = pd.merge(decile_datadv, sumsdv, on='m', how='inner')

        ks_wtd1 = ks_wtd
        ks_wtd1['pct_goods'] = ks_wtd1['goods_wtd'] / ks_wtd1['tot_goods']
        ks_wtd1['pct_bads'] = ks_wtd1['bads_wtd'] / ks_wtd1['tot_bads']
        ks_wtd1['pct_accts'] = ks_wtd1['total_wtd'] / ks_wtd1['tot_all']
        ks_wtd1['cum_pct_goods'] = ks_wtd1.pct_goods.cumsum()
        ks_wtd1['cum_pct_bads'] = ks_wtd1.pct_bads.cumsum()
        ks_wtd1['cum_pct_accts'] = ks_wtd1.pct_accts.cumsum()
        ks_wtd1['ks'] = (ks_wtd1['cum_pct_goods'] - ks_wtd1['cum_pct_bads']) * 100
        ks_wtd1['odds_rnk'] = ks_wtd1.tot_goods / ks_wtd1.tot_bads

        ks_wtddv1 = ks_wtddv
        ks_wtddv1['pct_goods'] = ks_wtddv1['goods_wtd'] / ks_wtddv1['tot_goods']
        ks_wtddv1['pct_bads'] = ks_wtddv1['bads_wtd'] / ks_wtddv1['tot_bads']
        ks_wtddv1['pct_accts'] = ks_wtddv1['total_wtd'] / ks_wtddv1['tot_all']
        ks_wtddv1['cum_pct_goods'] = ks_wtddv1.pct_goods.cumsum()
        ks_wtddv1['cum_pct_bads'] = ks_wtddv1.pct_bads.cumsum()
        ks_wtddv1['cum_pct_accts'] = ks_wtddv1.pct_accts.cumsum()
        ks_wtddv1['ks'] = (ks_wtddv1['cum_pct_goods'] - ks_wtddv1['cum_pct_bads']) * 100
        ks_wtddv1['odds_rnk'] = ks_wtddv1.tot_goods / ks_wtddv1.tot_bads

        psi = pd.DataFrame(
            {
                'total_counts_dev': ks_wtddv1.total_wtd,
                'total_counts_ootv': ks_wtd1.total_wtd
            }
        )

        psi['counts % _ dev'] = psi.total_counts_dev / psi.total_counts_dev.sum()
        psi['counts % _ ootv'] = psi.total_counts_ootv / psi.total_counts_ootv.sum()

        psi['%ootv-%dev'] = psi['counts % _ ootv'] - psi['counts % _ dev']
        psi['ln(%ootv/%dev)'] = np.log(psi['counts % _ ootv'] / psi['counts % _ dev'])

        psi['psi'] = psi['%ootv-%dev'] * psi['ln(%ootv/%dev)']
        psi = psi.psi.sum()
        return (psi)
    # end of function


    def split_x_y(self,data, x_title, y_titles):
        return data[list(x_title)], data[(y_titles)]
    # end of function


    def my_model_fit(self, model_fit, x_train, y_train):
        # 2017-11-28 Harry: last mod
        # use model defined and fit to train based on x,y data set
        y_predict_proba = model_fit.predict_proba(x_train)
        y_predict_proba = y_predict_proba[:, 1]
        y_predict = model_fit.predict(x_train)
        # computes area under prob curve
        auc_total_train = roc_auc_score(y_train, y_predict_proba)
        return y_predict_proba, y_predict, auc_total_train
    # end of function


    def AttrToDrop(self):
        # 2017-11-28 Evan: last mod
        # returns vector of attributes to drop
        # not all of these might be in data files
        drop1 = ['ind_remail',
                 'cf_ind_streamline',
                 'LD_FR320',
                 'LD_FR350',
                 'LD_HAR10',
                 'DO_channel_corr_brok',
                 'DO_owner_flagL',
                 'afil_bank_ind',
                 'cf_num_units',
                 'cf_note_rate',
                 'cf_ORIG_AMT',
                 'cf_TERM_OF_LOAN',
                 'cf_PRINCIPAL_BAL',
                 'cf_rate_diff',
                 'ind_prop_coop',
                 'cf_mthly_prin_interest_pmt'
                 ]
        drop2 = ['key', 'ACCOUNT_NBR', ',']
        return drop1 + drop2


    def ReadCleanData(self,FileName):
        # 2017-11-28 Evan: last mod
        # returns SAS file converted into DataFrame with unneeded cols dropped
        dataset = pd.read_sas(FileName)
        return dataset.drop(self.AttrToDrop, 1, errors='ignore')

    def rebalance_data(self, data, y_column, positive_value, copy_times):
        positive_dataset = data[data[y_column] == positive_value]
        for i in range(copy_times): data = data.append(positive_dataset)
        return data

    def Solver(self, InFile1, InFile2, InFile3):
        # 2017-11-28 Evan: last mod
        # main solver - continue to modularize this
        # InFile1 = training dataset file
        # InFile2 = testing dataset file
        # InFile3 = out of time testing file

        start = time.time()

        # SAS dataset already excluded the drop_list and some attributes to be dropped
        # more attributes to drop: ['key','ACCOUNT_NBR',','] as well as the indexes column
        # Evan asked on 11/21/2017 that we should join the test and training set so
        # that we can get whatever split ratio we want
        train_dataset = self.ReadCleanData(InFile1)
        test_dataset = self.ReadCleanData(InFile2)
        oot_dataset = self.ReadCleanData(InFile3)

        # we glue Train and Test set together to get original set
        train_data_full = pd.concat([train_dataset,test_dataset])

        # attribute name of model output variable
        # this is value we are trying to predict
        output_var = 'cnt_app'

        # 12-4-17 Harry: rebalance training set
        # optional data re-balancing to copy the REFI instances multiple times. We need to make REFI cases more significant compared with non-REFI cases
        copy_times = 0
        #train_data_full = self.rebalance_data(train_data_full,output_var,1,copy_times)

        # attribute names of model input variables
        # these are ALL inputs, even those without importance
        explanatory_vars = set(train_data_full.columns.values)
        explanatory_vars.remove(output_var)

        # 11-22-17 Harry: Page #36 explains why these cols are chosen. They have most feature importance
        # IN FUTURE: CHANGE THIS INPUT TO COME FROM SPREADSHEET DRIVING THIS ALGORITHM
        # input_features = ['DCF_REMAILH', 'D_afil_bank_indH', 'LD_ltv_pct', 'cf_curr_cltv', 'ld_ltv_max_6m']

        # what is counter?  seems to be used to generate random sampling of training data
        # but why is it multiplied by 3?
        for fraction in self.fractions:
            for counter in self.counters:
                # randomly sample training set and testing set
                # random_state: seed for random number generator
                # we can just read the SAS file for the already-cut data set
                # as long as the random_state stays the same, the sampling output stays the same
                train_dataset = train_data_full.sample(frac=fraction, random_state=3*counter)
                test_dataset = train_data_full.loc[~train_data_full.index.isin(train_dataset.index)]

                ootv_1_dataset = oot_dataset

                X_total_train, y_total_train = self.split_x_y(train_dataset, self.input_features, output_var)
                X_test, y_test = self.split_x_y(test_dataset, self.input_features, output_var)
                X_OOTV_1, y_OOTV_1 = self.split_x_y(ootv_1_dataset, self.input_features, output_var)

                k = 1
                # grid - search for optimal meta - params
                for combo in itertools.product(*self.iterables):
                    # all these output lists should be members of class - make this entire file into CLASS
                    self.InternalSolver(fraction, counter, combo, X_total_train, y_total_train,
                       output_var, X_test, y_test, X_OOTV_1, y_OOTV_1, k)

        end = time.time()
        duration = end - start
        print("duration", duration)

        # HARRY - RIP THIS STUFF OUT INTO FUNCTION CALL
        RF_results_summary_090116 = pd.DataFrame({
            'learning_rate': self.learning_rate_list,
            'subsample': self.subsample_list,
            'n_estimators': self.n_est_list,

            'max_depth': self.max_depth_list,
            'max_feature': self.max_feature_list,
            'min_samples_leaf': self.min_samples_leaf_list,
            'min_samples_split': self.min_samples_split_list,
            #   'class_weight': class_weight_list,
            'k': self.k_list,
            'kbest_columns': self.k_columns_full_list,
            'fraction': self.fraction_list,
            'counter': self.counter_list,
            'auc_total_train': self.auc_total_train_list,
            'ks_total_train': self.ks_total_train_list,
            'auc_test': self.auc_test_list,
            'ks_test': self.ks_test_list,
            'ks_mape_test': self.ks_mape_test_list,
            'ver1_ks_test': self.ver1_ks_test_list,
            'ver1_ks_mape_test': self.ver1_ks_mape_test_list,
            'psi_test': self.psi_test_list,

            'auc_OOTV_1': self.auc_OOTV_1_list,
            'ks_OOTV_1': self.ks_OOTV_1_list,
            'ks_mape_OOTV_1': self.ks_mape_OOTV_1_list,
            'ver1_ks_OOTV_1': self.ver1_ks_OOTV_1_list,
            'ver1_ks_mape_OOTV_1': self.ver1_ks_mape_OOTV_1_list,
            'psi_OOTV_1': self.psi_OOTV_1_list,

            'rank_order_test': self.rank_order_test_list,

            'rank_order_OOTV_1': self.rank_order_OOTV_1_list,
            'ks_ratio_test': self.ks_ratio_test_list,

            'ks_ratio_OOTV_1': self.ks_ratio_OOTV_1_list,

            'psi_ITV_OOTV_1': self.psi_ITV_OOTV_1_list,
            'rank_order_total_train': self.rank_order_total_train_list,
            'ks_mape_total_train': self.ks_mape_total_train_list,
            'prob_shift': self.prob_shift_list

        })

        # 12-5-17 Harry: rearrange the output columns
        column_list = [
            # meta parameters
            'fraction',
            'counter',
            'k',
            'kbest_columns',
            'prob_shift',
            'n_estimators',
            'max_depth',
            'max_feature',
            'min_samples_leaf',
            'min_samples_split',
            'learning_rate',
            'subsample',

            # performance measures by category
            'ks_total_train',
            'ks_ratio_test',
            'ks_ratio_OOTV_1',

            'auc_total_train',
            'auc_test',
            'auc_OOTV_1',

            'rank_order_total_train',
            'rank_order_test',
            'rank_order_OOTV_1',
            
            'ks_mape_total_train',
            'ks_mape_test',
            'ks_mape_OOTV_1',

            'ver1_ks_test',
            'ver1_ks_mape_test',
            'ver1_ks_OOTV_1',
            'ver1_ks_mape_OOTV_1',
            
            'ks_test',
            'ks_OOTV_1',

            'psi_test',
            'psi_OOTV_1',
            'psi_ITV_OOTV_1',
        ]
        RF_results_summary_090116 = RF_results_summary_090116[column_list]
        RF_results_summary_090116.to_csv('AR_output.csv')
    # end of function


    def FindRatios(self, df, ActualHead, PredictHead):
        # 2017-12-4 Evan: last mod
        # returns prob of correct predictions for given dataset
        # INPUTS:
        #   df (dataframe) has:
        #     ActualHead col that is actual output
        #     PredictHead col that is predicted output
        # OUTPUT:
        #    a[0] = Pr(  correct predict | no REFI) - true-neg
        #    a[1] = Pr(incorrect predict | no REFI) - false-pos
        #    a[2] = Pr(incorrect predict |    REFI) - false-neg
        #    a[3] = Pr(  correct predict |    REFI) - true-pos
        mydf = df
        a = []
        for x in [0,1]:
            ntotal = mydf[mydf[ActualHead] == x].count()[1]
            for y in [0,1]:
                v = mydf[(mydf[ActualHead] == x) & (mydf[PredictHead] == y)]
                a.append(v.count()[1] / ntotal)
        return a


    def InternalSolver(self, fraction, counter, combo, X_total_train, y_total_train,
                       output_var, X_test, y_test, X_OOTV_1, y_OOTV_1, k):
        # 2017-12-1 Evan: last mod
        # calls model WITHIN grid-search for optimal meta-params

        # split combo into meta-params
        prob_shift, i, q, r, s, t, u, v = combo

        # predict full training data set
        model = ensemble.GradientBoostingClassifier(
            n_estimators=i,
            max_depth=q,
            max_features=r,
            min_samples_leaf=s,
            min_samples_split=t,
            learning_rate=u,
            subsample=v,
            random_state=3)

        # train the model using x and y training set
        model_fit = model.fit(X_total_train, y_total_train)

        # predict using model fitted using Training set
        y_predict_proba, y_predict, auc_total_train = self.my_model_fit(model_fit, X_total_train, y_total_train)

        # add the y column back to the x columns and add prediction probs
        # as well as the weight to each of the records (guessed)
        X_total_train_PR = X_total_train.copy()
        X_total_train_PR[output_var] = y_total_train.copy()
        X_total_train_PR['pred_prob'] = y_predict_proba * prob_shift
        X_total_train_PR['wt'] = 1
        # 2017-12-4 EVAN: appended new col for y_predict, so all in one dataframe
        X_total_train_PR['predict'] = y_predict

        # 2017-12-4 EVAN: calc false-pos, false-neg errors, etc
        w = self.FindRatios(X_total_train_PR, output_var, 'predict')
        print(w)
        # some kind of KS statistic for estimating accuracy
        ks_total_train, ks_mape_total_train, rank_order_total_train = self.ks_max_mape(
            X_total_train_PR, 'pred_prob', output_var, 1)

        # append the results to output arrays
        self.auc_total_train_list.append(auc_total_train)
        self.ks_total_train_list.append(ks_total_train)
        self.ks_mape_total_train_list.append(ks_mape_total_train)
        self.rank_order_total_train_list.append(rank_order_total_train)
        self.fraction_list.append(fraction)
        self.counter_list.append(counter)
        self.k_list.append(k)
        self.k_columns_full_list.append(self.input_features)
        self.n_est_list.append(i)
        self.max_depth_list.append(q)
        self.max_feature_list.append(r)
        self.min_samples_leaf_list.append(s)
        self.min_samples_split_list.append(t)
        self.learning_rate_list.append(u)
        self.subsample_list.append(v)
        self.prob_shift_list.append(prob_shift)

        #print("auc_total_train", auc_total_train)
        #print("ks_total_train", ks_total_train)

        # 11-21-17 Harry thinks this is doing the test comparison if there is some testing set
        if fraction != 1:
            y_predict_proba, y_predict, auc_test = self.my_model_fit(model_fit, X_test, y_test)

            X_test_PR = X_test.copy()
            X_test_PR['cnt_app'] = y_test.copy()
            X_test_PR['pred_prob'] = y_predict_proba * prob_shift
            X_test_PR['wt'] = 1
            # 2017-12-4 Harry: appended new col for y_predict, so all in one dataframe
            X_test_PR['predict'] = y_predict

            # model accuracy calculation for testing set
            w = self.FindRatios(X_test_PR, output_var, 'predict')

            ks_test, ks_mape_test, rank_order_test = self.ks_max_mape(X_test_PR, 'pred_prob', 'cnt_app', 1)
            ver1_ks_test, ver1_ks_mape_test, a = self.ks_max_mape_ver1(X_test_PR,
                                                                  X_total_train_PR,
                                                                  'pred_prob',
                                                                  'cnt_app', 1)
            psi_test = self.psia(X_test_PR, X_total_train_PR, 'pred_prob', 'cnt_app', 1)

            # following statistic doesn't get used, but we think it may be important
            # ks_test_single = self.ks_max(X_test_PR, 'pred_prob', 'cnt_app', 1)

            self.auc_test_list.append(auc_test)
            self.ks_test_list.append(ks_test)
            self.ks_mape_test_list.append(ks_mape_test)
            self.ver1_ks_test_list.append(ver1_ks_test)
            self.ver1_ks_mape_test_list.append(ver1_ks_mape_test)
            self.psi_test_list.append(psi_test)
            self.ks_ratio_test_list.append(ks_total_train / ks_test - 1)
            self.rank_order_test_list.append(rank_order_test)

            print("auc_test", auc_test)
            print("ks_test", ks_test)

            print("ks_ver1_test", ver1_ks_test)
            print("ks_mape_test", ks_mape_test)
            print("psi_test", psi_test)

        else:
            auc_test = 0
            ks_test = 0
            ks_mape_test = 0
            ver1_ks_test = 0
            ver1_ks_mape_test = 0
            psi_test = 0


        """ "predict the OOTV_1 data set " """
        y_predict_proba, y_predict, auc_OOTV_1 = self.my_model_fit(model_fit, X_OOTV_1, y_OOTV_1)

        X_OOTV_1_PR = X_OOTV_1.copy()
        X_OOTV_1_PR['cnt_app'] = y_OOTV_1.copy()
        X_OOTV_1_PR['pred_prob'] = y_predict_proba * prob_shift
        X_OOTV_1_PR['wt'] = 1

        ks_OOTV_1, ks_mape_OOTV_1, rank_order_OOTV_1 = self.ks_max_mape(X_OOTV_1_PR,
                                                                   'pred_prob',
                                                                   'cnt_app', 1)

        ver1_ks_OOTV_1, ver1_ks_mape_OOTV_1, a = self.ks_max_mape_ver1(X_OOTV_1_PR,
                                                                  X_total_train_PR,
                                                                  'pred_prob',
                                                                  'cnt_app', 1)

        psi_OOTV_1 = self.psia(X_OOTV_1_PR, X_total_train_PR, 'pred_prob', 'cnt_app', 1)

        psi_ITV_OOTV_1 = self.psia(X_OOTV_1_PR, X_test_PR, 'pred_prob', 'cnt_app', 1)

        self.auc_OOTV_1_list.append(auc_OOTV_1)
        self.ks_OOTV_1_list.append(ks_OOTV_1)
        self.ks_mape_OOTV_1_list.append(ks_mape_OOTV_1)
        self.ver1_ks_OOTV_1_list.append(ver1_ks_OOTV_1)
        self.ver1_ks_mape_OOTV_1_list.append(ver1_ks_mape_OOTV_1)
        self.psi_OOTV_1_list.append(psi_OOTV_1)
        self.ks_ratio_OOTV_1_list.append(ks_total_train / ks_OOTV_1 - 1)
        self.rank_order_OOTV_1_list.append(rank_order_OOTV_1)
        self.psi_ITV_OOTV_1_list.append(psi_ITV_OOTV_1)

        #print("auc_OOTV_1", auc_OOTV_1)
        #print("ks_OOTV_1", ks_OOTV_1)
        #print("ks_mape_OOTV_1", ks_mape_OOTV_1)
        #print("psi_OOTV_1", psi_OOTV_1)
    # end of function


# run gradient boost solver
x = GeneralSolver()
x.Solver("ar_train_may16_1228.sas7bdat",
         "ar_test_may16_1228.sas7bdat",
         "ar_otv_jul15_v2.sas7bdat")