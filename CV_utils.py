"""
Utils file for running NASH-DDT CV analysis
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import itertools
#from r_models import get_confintervals
from sklearn.utils import resample
#from av_cv_utils import AVSimulator, RepAgreement
import random
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import cohens_kappa
import multiprocessing as mp
from functools import partial
from sklearn.metrics import confusion_matrix
from functools import reduce
random.seed(1234)

class CVAnalysis():
    def __init__(self, cd_dict_m, cd_dict_a,participant_col_name, participant_col_name_aim,
                 gs_df, manifest_df,labels_dict, files_path, special_params,tp_dict, special_params_v2, util_dict, cd_dict_p,patho_suffix_aim,agree_suffix_aim):
        self.cd_dict_m = cd_dict_m
        self.cd_dict_a = cd_dict_a
        self.manifest_df = manifest_df 
        self.gs_df = gs_df
        self.participant_col_name_aim = participant_col_name_aim
        self.participant_col_name = participant_col_name
        self.files_path = files_path
        self.labels_dict = labels_dict
        self.special_params = special_params
        self.tp_dict = tp_dict
        self.special_params_v2 = special_params_v2
        self.util_dict = util_dict
        self.cd_dict_p = cd_dict_p   
        self.agree_suffix_aim =agree_suffix_aim
        self.patho_suffix_aim = patho_suffix_aim
    def get_utility_distributions(self, aim_df, read_type):
        util_dict = self.util_dict
        cd_dict_m = self.cd_dict_m
        a_df = aim_df.loc[aim_df.read_type != "Consensus"]
        init_df = a_df.loc[a_df.read_type == "Initial"]
        if read_type == "Initial":
            r_df = init_df
        else:
            s_df = a_df.loc[a_df.read_type == read_type]
            r_df = pd.concat([init_df, s_df])
        out_list = []
        for param_key in list(cd_dict_m.keys()):
            count_df = pd.DataFrame(r_df[util_dict[param_key]].value_counts().sort_index())
            count_df.insert(0, "utility",list(count_df.index))
            count_df.insert(0, "Parameter",[cd_dict_m[param_key]] * len(count_df))
            count_df = count_df.rename(columns = {util_dict[param_key]:"Counts"})
            count_df["Total"] = [count_df["Counts"].sum()]*len(count_df)
            count_df["Percent"] = np.round((count_df["Counts"]/count_df["Total"]) * 100, 2)
            count_df = count_df.reset_index(drop = True)
            out_list.append(count_df)
        return pd.concat(out_list)
    def get_aim_patho_agreement(self, aim_df):
        cd_dict_a = self.cd_dict_p
        patho_suffix_aim = self.patho_suffix_aim
        agree_suffix_aim = self.agree_suffix_aim
        out_list_count_prop = []
        out_list_count = []
        for param_key in list(cd_dict_a.keys()):
            if param_key in ["s", "b", "i"]:
                adeq_col = "he_sample_adequacy"
            else:
                adeq_col = "tri_sample_adequacy"
            param_df = aim_df[["read_type","identifier", "participant_id", adeq_col] + [cd_dict_a[param_key] + suff for suff in [patho_suffix_aim,agree_suffix_aim]]]

            initial_df = param_df.loc[param_df["read_type"] == "Initial"]
            count_prop_series = initial_df[cd_dict_a[param_key] +agree_suffix_aim].value_counts()/len(initial_df)
            count_series = initial_df[cd_dict_a[param_key] +agree_suffix_aim].value_counts()
            out_list_count.append(count_series)
            out_list_count_prop.append(count_prop_series)

        out_df = np.round(pd.DataFrame(out_list_count_prop) * 100, 2)
        out_count_df = pd.DataFrame(out_list_count)
        out_count_df["Total"] = out_count_df.apply(lambda x: np.nansum(x), axis = 1)
        for df in [out_count_df, out_df]:
            df["Parameter"] = list(self.cd_dict_m.values())
            df = df.reset_index(drop = True)
        mlt_df_count = pd.melt(out_count_df, value_vars = list(out_count_df)[:-2], id_vars = ["Parameter", "Total"]).dropna()
        mlt_df = pd.melt(out_df, value_vars = list(out_df)[:-1], id_vars = ["Parameter"]).dropna()
        mlt_df_count = mlt_df_count.rename(columns = {"variable": "Agree", "value":"Count"})
        mlt_df = mlt_df.rename(columns = {"variable":"Agree", "value":"Percent"})
        merge_df = mlt_df.merge(mlt_df_count, on = ["Parameter", "Agree"], how = "inner")
        return merge_df
    def _recode_cols(self, df, col, labels_list, alt_val = 9):
        for label in labels_list:
            df[col] = df[col].replace({label:alt_val})
        return df
    def _round_col(self,df, col_name, n_digits = 3):
        """
        Function for rounding a given column of a dataframe to a set number of digits after the decimal point.
        """
        df[col_name]= np.round(df[col_name],n_digits)
        return df
    def _distribution_table(self,df, gt= True):
        """
        Get distribution table 
        """
        out_list = []
        if gt:
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
        for col_name in list(cd_dict.values()):
            # df = df.dropna(subset = [col_name])
            f_df = df[[participant_col, col_name]].groupby(col_name).count()
            f_df = f_df.rename(columns = {participant_col: "Count"})
            f_df["Total"] = f_df["Count"].sum()
            f_df["Frequency"] = (f_df["Count"]/f_df["Total"]) * 100        
            f_df.insert(0, "Score", list(f_df.index))
            f_df.insert(0,"Feature", [col_name] * len(f_df))
            out_list.append(f_df)
        return out_list
        
    def get_distribution_table_score(self, df = None, gt = True):
        if df is None and gt == True:
            df = self.gs_df
        # else:
        #     df = df.merge(manifest_df[["Partner", "timepoint", "identifier"]].drop_duplicates(),
        #                              on = "identifier", how ="inner")
        # df = self.gs_df
        out_df = pd.concat(self._distribution_table(df, gt = gt))
        out_df = self._round_col(out_df, "Frequency")
        return(out_df)    
    def _get_nas_sum_check(self,row, cd_dict,n_nas, n_sum):
        nas_params= [cd_dict[param] for param in ["s", "b", "i"]]
        row_arr = row[nas_params].values.astype("float")
        if len(row_arr[np.isnan(row_arr)]) > 0 :
            out_val = np.nan
        else:
            if n_nas in row_arr and np.sum(row_arr) == n_sum:
                out_val = 1
            elif n_nas not in row_arr and np.sum(row_arr) == n_sum:
                out_val = 2
            else:
                out_val= 0
        return out_val 
    def get_distribution_table_nas(self, df, gt = True):
        if gt:
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
        nas_params = ["b", "s", "i"]
        df["NAS_sum"] = df[[cd_dict[param] for param in nas_params]].sum(axis = 1)
        df["NAS_sum_check_0_4"] = df.apply(self._get_nas_sum_check,args = (cd_dict,0, 4 ), axis = 1)
        # df["NAS_sum_check_1_4"] = df.apply(self._get_nas_sum_check,args = (cd_dict,1, 4 ), axis = 1)
        df["NAS_sum_check_0_5"] = df.apply(self._get_nas_sum_check,args = (cd_dict,0, 5), axis = 1)
        df = df.dropna(subset = [cd_dict[param] for param in ["b", "s","i"]])
        col_name = "NAS_sum"
        f_df = df[[participant_col,col_name]].groupby(col_name).count()
        f_df = f_df.rename(columns = {participant_col: "Count"})
        f_df = f_df.drop([4,5])
        f_df.loc["4; score of at least 0","Count"] = len(df.loc[df["NAS_sum_check_0_4"] == 1])
        f_df.loc["NAS=4; score of at least 1","Count"] = len(df.loc[df["NAS_sum_check_0_4"] == 2])
        f_df.loc["5; score of at least 0","Count"] = len(df.loc[df["NAS_sum_check_0_5"] == 1])
        f_df.loc["5","Count"] = len(df.loc[df["NAS_sum_check_0_5"] == 2])
        f_df["Total"] = f_df["Count"].sum()
        f_df["Frequency"] = np.round((f_df["Count"]/f_df["Total"]) * 100, 2)        
        f_df.insert(0,"Feature", [col_name] * len(f_df))
        f_df.insert(0, "Score", list(f_df.index))
        # out_list.append(f_df)
        return f_df.reset_index(drop = True)
    
    def get_distribution_table_nas_by_sponsor(self,sponsor_col,df,  gt = True):
        if gt:
            in_df = self.gs_df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name],
                                      how = "inner")
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            in_df = df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name_aim],
                                      how = "inner")
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
            
        out_list = []
        for sponsor in sorted(set(in_df[sponsor_col])):
            s_df = in_df.loc[in_df[sponsor_col] == sponsor]
            s_df_count = self.get_distribution_table_nas(df = s_df, gt = gt)
            s_df_count.insert(0, sponsor_col, [sponsor]*len(s_df_count))
            out_list.append(s_df_count)
        out_df = pd.concat(out_list)
        return out_df
    def get_slide_distribution_by_sponsor_score(self,sponsor_col,df = None,  gt = True):
        if df is None and gt == True:
            in_df = self.gs_df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name],
                                      how = "inner")
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            in_df = df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name_aim],
                                      how = "inner")
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
        out_list = []
        for param in list(cd_dict.keys()):
            param_df = in_df[[participant_col, cd_dict[param], sponsor_col]].drop_duplicates().dropna()
            grp_df = param_df.groupby([cd_dict[param], sponsor_col]).count()
            grp_df.insert(0,sponsor_col,[m_idx[1] for m_idx in list(grp_df.index)])
            grp_df.insert(0,"Score",[m_idx[0] for m_idx in list(grp_df.index)])
            grp_df = grp_df.rename(columns = {participant_col:"Counts"})
            grp_df = grp_df.reset_index(drop = True)
            sum_df = pd.DataFrame()
            n_cats = sorted(set(grp_df[sponsor_col]))
            n_score = sorted(set(grp_df["Score"]))
            sum_df = pd.DataFrame()
            for idx,n_cat in list(enumerate(n_cats)):
                # for idx, score in list(enumerate(n_score)):
                sum_df.loc[idx, sponsor_col] = n_cat
                sum_df.loc[idx, "total"] = grp_df.loc[grp_df[sponsor_col] == n_cat]["Counts"].sum()
            grp_df = grp_df.merge(sum_df, on = sponsor_col, how = "inner")
            grp_df["Percent"] = np.round(grp_df["Counts"]/grp_df["total"]*100,2)
            grp_df.insert(0, "Parameter", [self.cd_dict_m[param]]*len(grp_df))
            out_list.append(grp_df)
        out_df = pd.concat(out_list)
        return out_df
    
    def get_slide_distribution_by_sponsor(self, sponsor_col,df = None, gt = True):
        if df is None and gt == True:
            in_df = self.gs_df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name],
                                      how = "inner")
        else:
            in_df = df.merge(self.manifest_df[["identifier", sponsor_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name_aim],
                                      how = "inner")
        freq_df = pd.DataFrame(in_df[sponsor_col].value_counts())
        freq_df = freq_df.rename(columns = {sponsor_col: "Counts"})
        freq_df["Total"] = [freq_df["Counts"].sum()] * len(freq_df)
        freq_df.insert(0, sponsor_col, list(freq_df.index))
        freq_df["Percent"] = np.round(freq_df["Counts"]/freq_df["Total"]*100,2)
        freq_df = freq_df.reset_index(drop = True)
        return freq_df
    def get_slide_distribution_by_timepoint(self, time_col, df = None, gt = True):
        if df is None and gt == True:
            in_df = self.gs_df.merge(self.manifest_df[["identifier", time_col]].drop_duplicates(),
                                      right_on = ["identifier"], left_on = [self.participant_col_name],
                                      how = "inner")
        else:
            in_df = df.merge(self.manifest_df[["identifier", time_col]].drop_duplicates(),
                                      left_on = ["identifier"], right_on = [self.participant_col_name_aim],
                                      how = "inner")
        tp_df1 = in_df.copy()
        tp_df2 = in_df.copy()
        tp_dict = self.tp_dict
        for tp in list(tp_dict.keys()):
            for tp_val in list(tp_dict[tp]):
                tp_df1[time_col] = tp_df1[time_col].replace({tp_val:tp})
        for tp_val in tp_dict["Baseline"]:
            tp_df2[time_col] = tp_df2[time_col].replace({tp_val:"Baseline"})
        out_list = []
        for df in [tp_df1, tp_df2]:
            freq_df = pd.DataFrame(df[time_col].value_counts())
            freq_df = freq_df.rename(columns = {time_col: "Counts"})
            freq_df["Total"] = [freq_df["Counts"].sum()] * len(freq_df)
            freq_df.insert(0, time_col, list(freq_df.index))
            freq_df["Percent"] = np.round(freq_df["Counts"]/freq_df["Total"]*100,2)
            freq_df = freq_df.reset_index(drop = True)
            out_list.append(freq_df)
        return out_list   
    def _get_agg_rate(self, arr1, arr2):
        return np.mean(arr1 == arr2)
    def _get_kappa(self, arr1,arr2, labels, weights = "linear"):
        # agg_rate = np.mean(arr1 == arr2)
        # if agg_rate == 1:
        #     kappa = 1
        # else:
        kappa = cohen_kappa_score(arr1, arr2,labels = labels, weights= weights)
        return kappa
    def compute_npa_ppa_conf_matrix(arr1, arr2, a_col):
        conf_matrix  = confusion_matrix(arr2, arr1)
        opa = np.trace(conf_matrix) / np.sum(conf_matrix)
        npa = np.sum(conf_matrix[0,0]) / np.sum(conf_matrix[0,:])
        ppa = np.sum(conf_matrix[1,1]) / np.sum(conf_matrix[1,:])
        return opa,npa, ppa
    def get_nas_kappa(self, df, method = "AIM", drop_duplicates = True, metric = "Kappa"):
        l1 = list(range(4))
        l2 = list(range(4,9))
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a["s"], self.cd_dict_a["b"],
                          self.cd_dict_a["i"]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m["s"],self.cd_dict_m["b"],
                                  self.cd_dict_m["i"]]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a["s"], self.cd_dict_a["b"],
                          self.cd_dict_a["i"]]] 
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m["s"],self.cd_dict_m["b"],
                                  self.cd_dict_m["i"]]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            merge_df["NAS_AIM"] = merge_df[self.cd_dict_a["s"]] +  merge_df[self.cd_dict_a["b"]] + merge_df[self.cd_dict_a["i"]]
            merge_df["NAS_GT"] = merge_df[self.cd_dict_m["s"]] +  merge_df[self.cd_dict_m["b"]] + merge_df[self.cd_dict_m["i"]]
            m_df = merge_df.copy()
            num_score = len(m_df)
            m_df = self._recode_cols(m_df, "NAS_AIM", l1,10)
            m_df = self._recode_cols(m_df, "NAS_GT", l1, 10)
            m_df = self._recode_cols(m_df, "NAS_AIM", l2, 11)
            m_df = self._recode_cols(m_df, "NAS_GT", l2, 11)
            kappa_p = self._get_kappa(m_df["NAS_AIM"], m_df["NAS_GT"],None, weights = None)
        else:
            gs_slide_col = "H & E Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m["s"],self.cd_dict_m["b"], self.cd_dict_m["i"]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m["s"],self.cd_dict_m["b"], self.cd_dict_m["i"], gs_slide_col]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m["s"],self.cd_dict_m["b"], self.cd_dict_m["i"]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m["s"],self.cd_dict_m["b"], self.cd_dict_m["i"], gs_slide_col]]
                
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            merge_df["NAS_M"] = merge_df[self.cd_dict_m["s"] + "_x"] +  merge_df[self.cd_dict_m["b"] + "_x"] + merge_df[self.cd_dict_m["i"] + "_x"]
            merge_df["NAS_GT"] = merge_df[self.cd_dict_m["s"] + "_y"] +  merge_df[self.cd_dict_m["b"] + "_y"] + merge_df[self.cd_dict_m["i"] + "_y"]
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            merge_df = self._recode_cols(merge_df, "NAS_M", l1,10)
            merge_df = self._recode_cols(merge_df, "NAS_GT",l1, 10)
            merge_df = self._recode_cols(merge_df, "NAS_M", l2, 11)
            merge_df = self._recode_cols(merge_df, "NAS_GT", l2, 11)
            num_score = len(set(merge_df[self.participant_col_name]))
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NAS_M"], p_df["NAS_GT"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
        return ["NAS_4", np.round(kappa_p,6), num_score]
    def get_merge_analysis_df(self, df, param_key, method = "AIM", drop_duplicates = True):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
        else:
            if param_key == "f":
                gs_slide_col = "Trichrome Slide"
            else:
                gs_slide_col = "H & E Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]]
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
        return merge_df
    def get_accuracy_kappa(self, df, param_key, method = "AIM", drop_duplicates = True, per_score = False):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            if per_score:
                labels_list = self.labels_dict[param_key]
                o_list = []
                for score in labels_list:
                    alt_labels = sorted(set(labels_list) - set([score]))
                    m_df = merge_df.copy()
                    m_df = self._recode_cols(m_df, self.cd_dict_a[param_key], alt_labels)
                    m_df = self._recode_cols(m_df, self.cd_dict_m[param_key], alt_labels)
                    kappa_p = self._get_kappa(m_df[self.cd_dict_a[param_key]], m_df[self.cd_dict_m[param_key]],None, weights = None)
                    num_score = len(m_df.loc[np.logical_and(m_df[self.cd_dict_a[param_key]] == score,
                                                                m_df[self.cd_dict_m[param_key]] == score)])
                    o_list.append([score, np.round(kappa_p,6),num_score])
            else:
                kappa_p = self._get_kappa(merge_df[self.cd_dict_a[param_key]], merge_df[self.cd_dict_m[param_key]], 
                                      self.labels_dict[param_key],weights = "linear")
                n_samples = len(merge_df)
                o_list = [np.round(kappa_p,6), n_samples]

        else:
            if param_key == "f":
                gs_slide_col = "Trichrome Slide"
            else:
                gs_slide_col = "H & E Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]]
                
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            if per_score:
                o_list = []
                labels_list = self.labels_dict[param_key]
                for score in labels_list:
                    kappa_list = []
                    alt_labels = sorted(set(labels_list) - set([score]))
                    m_df1 = merge_df.copy()
                    n_score = len(set(m_df1.loc[np.logical_and(m_df1[self.cd_dict_m[param_key]+"_x"] == score,
                                                               m_df1[self.cd_dict_m[param_key]+"_y"] == score)][self.participant_col_name]))
                    m_df1 = self._recode_cols(m_df1, self.cd_dict_m[param_key]+ "_x", alt_labels)
                    m_df1 = self._recode_cols(m_df1, self.cd_dict_m[param_key]+ "_y", alt_labels)
                    for patho in patho_list:
                        p_df = m_df1.loc[m_df1.user_name == patho]
                        if drop_duplicates:
                            p_df = p_df.drop_duplicates()
                        kappa_p = self._get_kappa(p_df[self.cd_dict_m[param_key]+"_x"], p_df[self.cd_dict_m[param_key]+"_y"],None,
                                                      weights = None)
                        kappa_list.append(kappa_p)
                    kappa_list = [k for k in kappa_list if not np.isnan(k)]
                    kappa_p = np.mean(kappa_list)
                    o_list.append([score, np.round(kappa_p,6),n_score])
            else:
                kappa_list = []
                for patho in patho_list:
                    p_df = merge_df.loc[merge_df.user_name == patho]
                    if drop_duplicates:
                        p_df = p_df.drop_duplicates()
                    kappa_list.append(self._get_kappa(p_df[self.cd_dict_m[param_key]+"_x"], p_df[self.cd_dict_m[param_key]+"_y"],
                                                      self.labels_dict[param_key],weights = "linear"))
                kappa_list = [k for k in kappa_list if not np.isnan(k)]
                kappa_p = np.mean(kappa_list)
                n_samples = len(set(merge_df[self.participant_col_name]))
                o_list = [np.round(kappa_p,6), n_samples]
        return o_list
    def get_f2_f3_accuracy_kappa(self, df,method = "AIM", drop_duplicates = True):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, "F2_F3_AIM"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, "F2_F3_GT"]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, "F2_F3_AIM"]]
                g_df = self.gs_df[[self.participant_col_name, "F2_F3_GT"]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            m_df = merge_df.copy()
            kappa_p = self._get_kappa(m_df["F2_F3_GT"], m_df["F2_F3_AIM"],None, weights = None)
            num_score = len(m_df)
        else:
            gs_slide_col = "Trichrome Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","F2_F3_M"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, gs_slide_col, "F2_F3_GT"]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","F2_F3_M"]]
                g_df = self.gs_df[[self.participant_col_name, gs_slide_col, "F2_F3_GT"]]        
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["F2_F3_GT"], p_df["F2_F3_M"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_p = np.nanmean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["f2_f3", np.round(kappa_p,6),num_score] 
    def get_f0_f1_accuracy_kappa(self, df,method = "AIM", drop_duplicates = True):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, "F0_F1_AIM"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, "F0_F1_GT"]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, "F0_F1_AIM"]]
                g_df = self.gs_df[[self.participant_col_name, "F0_F1_GT"]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            m_df = merge_df.copy()
            kappa_p = self._get_kappa(m_df["F0_F1_GT"], m_df["F0_F1_AIM"],None, weights = None)
            num_score = len(m_df)
        else:
            gs_slide_col = "Trichrome Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","F0_F1_M"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, gs_slide_col, "F0_F1_GT"]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","F0_F1_M"]]
                g_df = self.gs_df[[self.participant_col_name, gs_slide_col, "F0_F1_GT"]]        
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["F0_F1_GT"], p_df["F0_F1_M"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_p = np.nanmean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["f0_f1", np.round(kappa_p,6),num_score] 
    def _get_nas_1(self,row, cd_dict, col_name):
        nas_params= [cd_dict[param] for param in ["s", "b", "i"]]
        row_arr = row[nas_params].values.astype("float")
        if len(row_arr[np.isnan(row_arr)]) > 0 :
            out_val = np.nan
        elif 0 in row_arr or row[col_name] < 4:
            out_val = 0
        else:
            out_val = 1
        return out_val 
    def _get_fibrosis_subscores(self,row, cd_dict, fib_subscores = [0,1]):
        fib_score = row[cd_dict["f"]]
        if np.isnan(fib_score):
            return np.nan
        elif fib_score in fib_subscores:
            return 1
        else:
            return 0
    def _get_nas_4(self,row,nas_col, thr = 4):
        nas_score = row[nas_col]
        if np.isnan(nas_score):
            return np.nan
        elif nas_score >= thr:
            return 1
        else:
            return 0
    def _get_nash_res(self,row, cd_dict, col_name):
        nas_params= [cd_dict[param] for param in ["s", "b", "i"]]
        row_arr = row[nas_params].values.astype("float")
        if len(row_arr[np.isnan(row_arr)]) > 0 :
            out_val = np.nan
        elif row[cd_dict["b"]] == 0 and row[cd_dict["i"]] < 2:
            out_val = 1
        else:
            out_val = 0
        return out_val 
    def _get_nash_res_fib(self,row, cd_dict, col_name):
        nas_params= [cd_dict[param] for param in ["s", "b", "i"]]
        row_arr = row[list(cd_dict.values())].values.astype("float")
        if len(row_arr[np.isnan(row_arr)]) > 0 :
            out_val = np.nan
        elif row[cd_dict["b"]] == 0 and row[cd_dict["i"]] < 2:
            out_val = 1
        else:
            out_val = 0
        return out_val 
    
    def get_nas_kappa_v2(self, df, method = "AIM", drop_duplicates = True):
        l1 = list(range(4))
        l2 = list(range(4,9))
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, "NAS_4_AIM"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, "NAS_4_GT"]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, "NAS_4_AIM"]]
                g_df = self.gs_df[[self.participant_col_name, "NAS_4_GT"]]
            m_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            m_df = m_df.dropna()
            kappa_p = self._get_kappa(m_df["NAS_4_AIM"], m_df["NAS_4_GT"],None, weights = None)
            num_score = len(m_df)
        else:
            gs_slide_col = "H & E Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","NAS_4_M"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name,gs_slide_col, "NAS_4_GT"]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","NAS_4_M"]]
                g_df = self.gs_df[[self.participant_col_name,gs_slide_col, "NAS_4_GT"]]
                
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NAS_4_M"], p_df["NAS_4_GT"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_p = np.nanmean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["NAS_4", np.round(kappa_p,6), num_score]  
    def get_nash_res_kappa(self, df, method = "AIM", drop_duplicates = True):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, "NASH_res_AIM"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name,"NASH_res_GT"]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, "NASH_res_AIM"]]
                g_df = self.gs_df[[self.participant_col_name,"NASH_res_GT"]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            m_df = merge_df.copy()
            kappa_p = self._get_kappa(m_df["NASH_res_AIM"], m_df["NASH_res_GT"],None, weights = None)
            num_score = len(m_df)
        else:
            gs_slide_col = "H & E Slide"
            if drop_duplicates:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","NASH_res_M"]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name,gs_slide_col, "NASH_res_GT"]].drop_duplicates()
            else:
                m_df = df[[self.participant_col_name,"user_name", "slide ID","NASH_res_M"]]
                g_df = self.gs_df[[self.participant_col_name,gs_slide_col, "NASH_res_GT"]]
                
            merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NASH_res_M"], p_df["NASH_res_GT"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_p = np.nanmean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["NASH_res", np.round(kappa_p,6), num_score]      
    def get_accuracy_df_special_v2(self, accuracy_df, manual_df, drop_duplicates = True):
        special_params = self.special_params_v2
        a_list_f2_f3 = self.get_f2_f3_accuracy_kappa(accuracy_df,method = "AIM", drop_duplicates = drop_duplicates)
        m_list_f2_f3 = self.get_f2_f3_accuracy_kappa(manual_df,method = "Manual", drop_duplicates = drop_duplicates)
        a_list_nas_4 = self.get_nas_kappa_v2(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_nas_4 = self.get_nas_kappa_v2(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        a_list_nash_res = self.get_nash_res_kappa(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_nash_res = self.get_nash_res_kappa(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        p_list = list(itertools.chain.from_iterable([[special_params[0]]*2, [special_params[1]]*2, [special_params[2]]*2]))
        d_list = list(itertools.chain.from_iterable([[a_list_f2_f3[1] - m_list_f2_f3[1]]*2, [a_list_nas_4[1] - m_list_nas_4[1]]* 2,
                                                    [a_list_nash_res[1] - m_list_nash_res[1]]*2]))
        
        out_df = pd.DataFrame({"Parameter": p_list,"Method": ["AIM-NASH", "Manual pathologists"] * 3,
                               "Kappa":[a_list_f2_f3[1], m_list_f2_f3[1], a_list_nas_4[1], m_list_nas_4[1],a_list_nash_res[1], m_list_nash_res[1]],"Difference": d_list, "N": [a_list_f2_f3[2], m_list_f2_f3[2], a_list_nas_4[2], m_list_nas_4[2],
                                                            a_list_nash_res[2],m_list_nash_res[2]]})
        return out_df    
    
    def get_per_patho_kappa(self,df, param_key, drop_duplicates = False):
        if param_key == "f":
            gs_slide_col = "Trichrome Slide"
        else:
            gs_slide_col = "H & E Slide"
        if drop_duplicates:
            m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
            g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]].drop_duplicates()
        else:
            m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]]
            g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]]
        merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"], right_on = [self.participant_col_name,gs_slide_col],
                      how = "inner")
        merge_df = merge_df.dropna()
        patho_list = sorted(set(merge_df.user_name))
        kappa_list = []
        for patho in patho_list:
            p_df = merge_df.loc[merge_df.user_name == patho]
            if drop_duplicates:
                p_df = p_df.drop_duplicates()
            kappa_list.append([patho, len(p_df),self._get_kappa(p_df[self.cd_dict_m[param_key]+"_x"], p_df[self.cd_dict_m[param_key]+"_y"],
                                                                self.labels_dict[param_key],
                                                                                    weights = "linear")])
        kappa_df = pd.DataFrame(kappa_list)
        kappa_df.columns = ["Pathologist", "N", "Kappa"]
        return kappa_df
    def get_accuracy_df(self, accuracy_df, manual_df, drop_duplicates = True, per_score = False):
        param_list = sorted(self.cd_dict_m.keys())
        out_list = []
        for param in param_list:
            if per_score:
                a_list = self.get_accuracy_kappa(accuracy_df, param, method = "AIM", drop_duplicates = drop_duplicates, per_score = True)
                m_list = self.get_accuracy_kappa(manual_df, param, method = "Manual", drop_duplicates = drop_duplicates, per_score = True)
                labels_list = self.labels_dict[param]
                kappa_list = list(itertools.chain.from_iterable([[a_list[i][1],m_list[i][1]] for i in range(len(a_list))]))
                score_list = list(itertools.chain.from_iterable([[i] * 2 for i in labels_list]))
                diff_list = list(itertools.chain.from_iterable([[a_list[i][1] - m_list[i][1], np.nan] for i in range(len(a_list))]))
                num_list = list(itertools.chain.from_iterable([[a_list[i][2],m_list[i][2]] for i in range(len(a_list))]))
                param_df = pd.DataFrame({"Parameter": [self.cd_dict_m[param]]* int(len(a_list) *2),
                                        "Method": ["AIM-NASH", "Manual pathologists"] * len(a_list),
                                        "Score": score_list,
                                        "Kappa": kappa_list,
                                        "Difference": diff_list,
                                        "N": num_list})
            else:
                a_list = self.get_accuracy_kappa(accuracy_df, param, method = "AIM", drop_duplicates = drop_duplicates)
                m_list = self.get_accuracy_kappa(manual_df, param, method = "Manual", drop_duplicates = drop_duplicates)
                param_df = pd.DataFrame({"Parameter":[self.cd_dict_m[param]]*2,
                                     "Method": ["AIM-NASH", "Manual pathologists"],
                                     "Kappa": [a_list[0], m_list[0]],
                                     "Difference": [a_list[0] - m_list[0],np.nan] ,
                                     "N": [a_list[1], m_list[1]]})
            out_list.append(param_df)
        out_df = pd.concat(out_list)
        return out_df
    def get_accuracy_df_iteration(self,  accuracy_df_bootstrap, manual_df_bootstrap,analysis_type,itr):
        iteration_accuracy_df = accuracy_df_bootstrap.loc[accuracy_df_bootstrap.Iteration == itr]
        iteration_manual_df = manual_df_bootstrap.loc[manual_df_bootstrap.Iteration == itr]
        if analysis_type == "all":
            out_df = self.get_accuracy_df(iteration_accuracy_df, iteration_manual_df, drop_duplicates = False)
        elif analysis_type == "per_sponsor":
            out_df = self.get_per_sponsor_accuracy(iteration_accuracy_df, iteration_manual_df, drop_duplicates = False)
        elif analysis_type == "per_timepoint":
            out_df = self.get_per_timepoint_accuracy(iteration_accuracy_df, iteration_manual_df, drop_duplicates = False)
        elif analysis_type == "per_score":
            out_df = self.get_accuracy_df(iteration_accuracy_df, iteration_manual_df,drop_duplicates = False,per_score = True)
        elif analysis_type == "special_v2":
            out_df = self.get_accuracy_df_special_v2(iteration_accuracy_df, iteration_manual_df,drop_duplicates = False)
        else:
            out_df = self.get_accuracy_df_special(iteration_accuracy_df, iteration_manual_df,drop_duplicates = False)
        out_df.insert(0, "Iteration", [itr] * len(out_df))
        return out_df
   
    def get_accuracy_df_bootstrap(self, accuracy_df_bootstrap, manual_df_bootstrap,analysis_type,pool = True, n_iterations = 10):
        partial_func = partial(self.get_accuracy_df_iteration, accuracy_df_bootstrap, manual_df_bootstrap, analysis_type)
        if pool:
            pool = mp.Pool()
            out_list = list(pool.map(partial_func, list(range(n_iterations))))
            del(pool)
        else:
            out_list = list(map(partial_func, list(range(n_iterations))))
        return out_list
    def get_accuracy_df_special(self, accuracy_df, manual_df, drop_duplicates = True):
        special_params = self.special_params
        a_list_f0_f1 = self.get_f0_f1_accuracy_kappa(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_f0_f1 = self.get_f0_f1_accuracy_kappa(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        a_list_nas_4 = self.get_nas_kappa(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_nas_4 = self.get_nas_kappa(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        p_list = list(itertools.chain.from_iterable([[special_params[0]]*2, [special_params[1]]*2]))
        d_list = list(itertools.chain.from_iterable([[a_list_f0_f1[1] - m_list_f0_f1[1]]*2, [a_list_nas_4[1] - m_list_nas_4[1]]* 2]))
        out_df = pd.DataFrame({"Parameter": p_list,"Method": ["AIM-NASH", "Manual pathologists"] * 2, "Kappa":[a_list_f0_f1[1], m_list_f0_f1[1], a_list_nas_4[1], m_list_nas_4[1]],
                                 "Difference": d_list, "N": [a_list_f0_f1[2], m_list_f0_f1[2], a_list_nas_4[2], m_list_nas_4[2]]})
        return out_df
    def get_per_timepoint_accuracy(self, accuracy_df, manual_df, drop_duplicates = True):
        manifest_df= self.manifest_df
        tp_dict = self.tp_dict
        out_list = []
        for tp in list(tp_dict.keys()):
            tp_participants = sorted(set(manifest_df.loc[manifest_df.timepoint.isin(tp_dict[tp])]["identifier"]))
            a_df = accuracy_df.loc[accuracy_df[self.participant_col_name_aim].isin(tp_participants)]
            m_df = manual_df.loc[manual_df[self.participant_col_name].isin(tp_participants)]
            tp_acc_df = self.get_accuracy_df(a_df, m_df, drop_duplicates = drop_duplicates)
            tp_acc_df.insert(0, "Time Point", [tp]*len(tp_acc_df))
            out_list.append(tp_acc_df)
        out_df = pd.concat(out_list)
        return out_df  
    
    def get_per_timepoint_tables(self, accuracy_df, manual_df, drop_duplicates = True):
        manifest_df= self.manifest_df
        tp_dict = self.tp_dict
        out_dict_tp_manual = {}
        out_dict_tp_aim = {}
        for tp in list(tp_dict.keys()):
            tp_participants = sorted(set(manifest_df.loc[manifest_df.timepoint.isin(tp_dict[tp])]["identifier"]))
            a_df = accuracy_df.loc[accuracy_df[self.participant_col_name_aim].isin(tp_participants)]
            m_df = manual_df.loc[manual_df[self.participant_col_name].isin(tp_participants)]
            out_dict_tp_manual[tp] = m_df
            out_dict_tp_aim[tp] = a_df
            # tp_acc_df = self.get_accuracy_df(a_df, m_df, drop_duplicates = drop_duplicates)
            # tp_acc_df.insert(0, "Time Point", [tp]*len(tp_acc_df))
            # out_list.append(tp_acc_df)
        # out_df = pd.concat(out_list)
        return out_dict_tp_manual, out_dict_tp_aim  
    
    def get_nash_res_kappa_fib(self, df, method = "AIM", drop_duplicates = True):
        g_df = self.get_bl_pbl_df(self.gs_df, method = "GT")
        if method == "AIM":
            a_df = self.get_bl_pbl_df(df, method = method)
            if drop_duplicates:
                a_df = a_df[["NASH_res_fib","subject_id"]].drop_duplicates()
                g_df = g_df[["NASH_res_fib","subject_id"]].drop_duplicates()
            else:
                a_df = a_df[["NASH_res_fib","subject_id"]]
                g_df = g_df[["NASH_res_fib","subject_id"]]
            merge_df = a_df.merge(g_df, on = "subject_id", how = "inner")
            merge_df = merge_df.dropna()
            m_df = merge_df.copy()
            kappa_p = self._get_kappa(m_df["NASH_res_fib_x"], m_df["NASH_res_fib_y"],None, weights = None)
            num_score = len(m_df)
        else:
            m_df = self.get_bl_pbl_df(df, method = method)
            if drop_duplicates:
                m_df = m_df[["NASH_res_fib", "subject_id", "user_name"]].drop_duplicates()
                g_df = g_df[["NASH_res_fib", "subject_id"]].drop_duplicates()
            else:
                m_df = m_df[["NASH_res_fib", "subject_id", "user_name"]]
                g_df = g_df[["NASH_res_fib", "subject_id"]]
            merge_df = m_df.merge(g_df, on ="subject_id",
                          how = "inner")
            merge_df = merge_df.dropna()
            patho_list = sorted(set(merge_df.user_name))
            kappa_list = []
            dict1 = {}
            dict2 = {}
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NASH_res_fib_x"], p_df["NASH_res_fib_y"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
            num_score = len(set(merge_df["subject_id"]))
        # return merge_df
        return ["NASH_res_fib", np.round(kappa_p,6), num_score]      
    def _get_nash_res_fib(self,row, cd_dict_bl, cd_dict_pbl):
        row_arr = row[list(cd_dict_bl.values()) + list(cd_dict_pbl.values())].values.astype("float")
        if len(row_arr[np.isnan(row_arr)]) > 0 :
            out_val = np.nan
        elif row[cd_dict_pbl["b"]] == 0 and row[cd_dict_pbl["i"]] < 2 and row[cd_dict_pbl["f"]] <= row[cd_dict_bl["f"]]:
            out_val = 1
        else:
            out_val = 0
        return out_val  
    
    def get_per_sponsor_accuracy(self, accuracy_df, manual_df, drop_duplicates = True):
        manifest_df = self.manifest_df
        datasets = sorted(set(manifest_df.Partner))
        out_list = []
        for dataset in datasets:
            d_participants = sorted(set(manifest_df.loc[manifest_df.Partner == dataset]["identifier"]))
            a_df = accuracy_df.loc[accuracy_df[self.participant_col_name_aim].isin(d_participants)]
            m_df = manual_df.loc[manual_df[self.participant_col_name].isin(d_participants)]
            d_acc_df = self.get_accuracy_df(a_df, m_df, drop_duplicates = drop_duplicates)
            d_acc_df.insert(0, "dataset", [dataset]*len(d_acc_df))
            out_list.append(d_acc_df)
        out_df = pd.concat(out_list)
        return out_df
    def get_bl_pbl_df(self, in_df, method = "AIM"):
        if method == "Manual":
            in_df =  self.collapse_patho_rr_df(in_df)          
        manifest_df = self.manifest_df
        tp_dict = self.tp_dict
        grp_df = manifest_df[["subject_id","visit"]].drop_duplicates().groupby("subject_id").count()
        bl_pbl_idx = list(grp_df.loc[grp_df.visit == 2].index)
        bl_pbl_df = manifest_df.loc[manifest_df.subject_id.isin(bl_pbl_idx)][["identifier", "subject_id", "visit"]].drop_duplicates()
        if method == "AIM":
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
        else:
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        merge_df = in_df.merge(bl_pbl_df, left_on = participant_col, right_on = "identifier", how = "inner")
        merge_df_bl = merge_df.loc[merge_df.visit == "Baseline"]
        merge_df_pbl = merge_df.loc[merge_df.visit == "Post Baseline"]
        bl_dict= {}
        pbl_dict = {}
        for param in list(cd_dict.keys()):
            bl_dict[cd_dict[param]] = cd_dict[param] + "_bl"
            pbl_dict[cd_dict[param]] = cd_dict[param] + "_pbl"
        merge_df_bl =merge_df_bl.rename(columns = bl_dict)
        merge_df_pbl = merge_df_pbl.rename(columns = pbl_dict)
        if method == "Manual":
            out_df =  merge_df_bl.merge(merge_df_pbl, on = ["subject_id","user_name"], how = "inner")
        else:
            out_df = merge_df_bl.merge(merge_df_pbl, on = "subject_id", how = "inner")
        cd_dict_bl = {}
        cd_dict_pbl = {}
        for param, value in cd_dict.items():
            cd_dict_bl[param] = value + "_bl"
            cd_dict_pbl[param] = value + "_pbl"
        out_df["NASH_res_fib"] = out_df.apply(self._get_nash_res_fib, args = (cd_dict_bl, cd_dict_pbl), axis =1)
        return out_df
    
    def get_bootstrap_samples_itr(self, input_df,metric,sample_col, drop_cols,itr):
        if metric == "accuracy_aim":
            b_df = input_df.sample(len(input_df), replace = True)
        else:
            sample_list = sorted(set(input_df[sample_col]))
            b_samples = random.choices(sample_list, k= len(sample_list))
            b_df =  pd.concat([input_df.loc[input_df[sample_col] == b] for b in b_samples])
        b_df.insert(0, "Iteration", itr)
        b_df = b_df.drop(drop_cols, axis = 1)
        return b_df
    def get_bootstrap_samples_df(self,input_df, metric, sample_col,drop_cols,  n_iterations = 2000, write = True,
                                set_seed = True):
        if set_seed:
            random.seed(1234)
        partial_func_bootstrap = partial(self.get_bootstrap_samples_itr, input_df,  metric, sample_col,drop_cols)
        pool = mp.Pool()
        itr_list = list(range(n_iterations))
        b_list = list(pool.map(partial_func_bootstrap, itr_list))
        del(pool)
        out_df = pd.concat(b_list)
        if write:
            out_df.to_csv(self.files_path + "NASH_DDT_CV_production_data_bootstrap_" + metric + "_032123.csv", index = False)
        else:
            return out_df
        
    def get_ci_df(self, bootstrap_list, s_col =None, d_col = "Difference", metric_col = "Kappa", n_iterations = 10, alpha = 0.05, interpolation = "midpoint", precomputed_bootstraps= False):
        if precomputed_bootstraps:
            in_df = bootstrap_list
        else:
            in_df = pd.concat(bootstrap_list)
        if s_col is None:
            met_df = pd.DataFrame()
            m_list = []
            for param in sorted(set(in_df.Parameter)):
                p_df = in_df.loc[in_df.Parameter == param]
                met_df.loc[param, "CI_LOW_Metric"]= np.round(p_df.dropna()[d_col].quantile(alpha/2, interpolation= interpolation), 6)
                met_df.loc[param, "CI_UP_Metric"]= np.round(p_df.dropna()[d_col].quantile(1 - (alpha/2), interpolation= interpolation), 6)
                met_df.loc[param, "P_value"] = self._get_p_proportion(list(p_df.dropna()[d_col]),
                                                                      endpoint = "Accuracy") 
                c_list = list(p_df.dropna()[d_col])
                met_df.loc[param, "P_values"] = len(np.where(np.array(c_list) < 0.0)[0])/len(c_list)
                method_df = pd.DataFrame()
                for method in sorted(set(in_df.Method)):
                    method_df.loc[method,"CI_LOW_Method"] = np.round(p_df.loc[p_df.Method == method][metric_col].quantile(alpha/2,interpolation= interpolation), 6)
                    method_df.loc[method,"CI_UP_Method"] = np.round(p_df.loc[p_df.Method == method][metric_col].quantile(1 - (alpha/2),interpolation= interpolation), 6)
                method_df.insert(0, "Method", list(method_df.index))
                method_df.insert(0, "Parameter", [param]*len(method_df))
                method_df = method_df.reset_index(drop = True)
                m_list.append(method_df)
            met_df.insert(0, "Parameter", sorted(set(in_df.Parameter)))
            method_concat_df = pd.concat(m_list)
            out_df = met_df.merge(method_concat_df, on = ["Parameter"], how = "outer")
        else:
            out_list = []
            met_list = []
            method_list = []
            for param in sorted(set(in_df.Parameter)):
                met_df = pd.DataFrame()
                p_df = in_df.loc[in_df.Parameter == param]
                m_list = []
                for s_cat in sorted(set(p_df[s_col])):
                    s_df = p_df.loc[p_df[s_col] == s_cat]
                    met_df.loc[s_cat,"CI_LOW_Metric"] = np.round(s_df.dropna()[d_col].quantile(alpha/2,interpolation= interpolation), 6)
                    met_df.loc[s_cat,"CI_UP_Metric"] = np.round(s_df.dropna()[d_col].quantile(1 - (alpha/2),interpolation= interpolation), 6)
                    met_df.loc[s_cat, "P_value"] = self._get_p_proportion(list(s_df.dropna()[d_col]),
                                                                      endpoint = "Accuracy")
                    method_df = pd.DataFrame()
                    for method in sorted(set(in_df.Method)):
                        method_df.loc[method, "CI_LOW_Method"] =  np.round(s_df.loc[s_df.Method == method][metric_col].quantile(alpha/2,interpolation= interpolation), 6)
                        method_df.loc[method, "CI_UP_Method"] = np.round(s_df.loc[s_df.Method == method][metric_col].quantile(1-(alpha/2),interpolation= interpolation), 6)
                    method_df.insert(0, "Method", list(method_df.index))
                    method_df.insert(0, s_col, [s_cat]*len(method_df))
                    m_list.append(method_df)
                m_df = pd.concat(m_list)
                met_df.insert(0, s_col, list(met_df.index))
                met_df.insert(0, "Parameter", [param] * len(met_df))
                m_df.insert(0, "Parameter", [param] * len(m_df))
                m_df = m_df.reset_index(drop = True)
                met_df = met_df.reset_index(drop = True)
                met_list.append(met_df)
                method_list.append(m_df)
            method_concat_df = pd.concat(method_list)
            met_concat_df = pd.concat(met_list)
            out_df = method_concat_df.merge(met_concat_df, on = ["Parameter", s_col], how = "outer")
        return out_df  
    def get_result_df_with_ci(self,results_df, bootstrap_list,metric_col = "Kappa",
                              d_col = "Difference",
                              endpoint = "Accuracy",n_iterations = 10, s_col = None, special= False,interpolation = "midpoint", precomputed_bootstraps = False):
        ci_df = self.get_ci_df(bootstrap_list, s_col =s_col, d_col = d_col,
                               metric_col = metric_col, n_iterations = n_iterations, 
                  alpha = 0.05, interpolation = interpolation, precomputed_bootstraps = precomputed_bootstraps)
        if s_col is None:
            out_df = results_df.merge(ci_df, on = ["Parameter", "Method"],how = "inner")
        else:
            out_df = results_df.merge(ci_df, on = ["Parameter", "Method", s_col], how ="inner")
        return out_df.round(6)
    def _get_p_proportion(self, in_list, endpoint = "Accuracy"):
        if endpoint == "Accuracy":
            out_p = len(np.where(np.array(in_list) < -0.1)[0])/len(in_list)
        else:
            out_p= len(np.where(np.array(in_list) <= 0.85)[0])/len(in_list)
        return out_p

class CVExploratory(CVAnalysis):
    def __init__(self,names_dict, cd_dict_m, cd_dict_a, cd_dict_a1,participant_col_name,participant_col_name_aim,gs_df,
             manifest_df, labels_dict, files_path, special_params,
                 tp_dict, special_params_v2, util_dict, cd_dict_p, patho_suffix_aim, agree_suffix_aim, ml_suffix_aim):
        super().__init__(cd_dict_m, cd_dict_a, participant_col_name,participant_col_name_aim,
                         gs_df,
             manifest_df, labels_dict, files_path, special_params,
                 tp_dict, special_params_v2, util_dict, cd_dict_p, patho_suffix_aim, agree_suffix_aim)
        self.names_dict = names_dict
        self.ml_suffix_aim = ml_suffix_aim
        self.cd_dict_a1 = cd_dict_a1
    def get_accuracy_df(self, df,
                        param_key, 
                        method = "AIM", 
                        drop_duplicates = True):
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
        else:
            if method == "Manual":
                if param_key == "f":
                    gs_slide_col = "Trichrome Slide"
                else:
                    gs_slide_col = "H & E Slide"
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]]
                m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]].drop_duplicates()
                merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
            else:
                if param_key == "f":
                    gs_slide_col = "Trichrome"
                else:
                    gs_slide_col = "H&E"
                if drop_duplicates:
                    g_df = df[[self.participant_col_name, self.cd_dict_m[param_key],
                               "user_name", "stain"]].drop_duplicates()
                    g_df = g_df.loc[g_df.stain == gs_slide_col]
                else:
                    m_df = df[[self.participant_col_name,"user_name", "slide ID",self.names_dict[param_key]]]
                    g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]]
                    # out_list = []
                merge_df= pd.DataFrame()
                for identifier in sorted(set(g_df.identifier)):
                    id_df = g_df.loc[g_df.identifier == identifier]
                    try:
                        assert len(id_df) == 2, print("Identifier " + identifier + " does not have 2 rows")
                        user_list= list(id_df.user_name)
                        merge_df.loc[identifier, "PathA"] = float(id_df.loc[id_df.user_name == user_list[0]][self.cd_dict_m[param_key]].values[0])
                        merge_df.loc[identifier, "PathB"] = float(id_df.loc[id_df.user_name == user_list[1]][self.cd_dict_m[param_key]].values[0])
                    except:
                        merge_df.loc[identifier, "PathA"] = np.nan
                        merge_df.loc[identifier, "PathB"] = np.nan
                merge_df = merge_df.dropna()
        return merge_df
    def _get_agg_rate(self,arr1,arr2, labels_list = []):
        if len(labels_list) == 0:
            ctab = confusion_matrix(arr1, arr2)
        else:
            ctab = confusion_matrix(arr1, arr2, labels = labels_list)
        agg_rate = np.sum(np.diag(ctab))/np.sum(ctab)
        return agg_rate
    def compute_avg_agreement(self, a_df, param_key):
        a_df = self.get_accuracy_df(a_df, param_key) 
        labels_list = self.labels_dict[param_key]
        linear_kappa = self._get_kappa(a_df[self.cd_dict_a[param_key]], a_df[self.cd_dict_m[param_key]],
                                       labels = labels_list)
        #return agg_rate_ceil, agg_rate_floor
        ctab_df = pd.crosstab(a_df[self.cd_dict_a[param_key]], a_df[self.cd_dict_m[param_key]])
        out_path = self.files_path
        out_df = ctab_df.melt()
        out_df.columns = ["GT score", "Count"]
        out_df["GT score"] =  out_df["GT score"].astype("int")
        out_df.insert(1, "AIM score",labels_list * len(labels_list))
        out_df.insert(0, "Parameter", [self.names_dict[param_key]] * len(out_df))
        return out_df
    def compute_avg_agreement_manual(self, a_df, param_key):
        a_df = self.get_accuracy_df(a_df, param_key, method = "Manual")
        labels_list = self.labels_dict[param_key]
        patho_list = sorted(set(a_df.user_name))
        out_path = self.files_path
        agg_df_list = []
        agg_df_dict= {}
        fig, axs = plt.subplots(2,int(len(patho_list)/2), figsize=(20, 10))
        for idx, patho in list(enumerate(patho_list)):
            p_df = a_df.loc[a_df.user_name == patho].dropna()
            linear_kappa = self._get_kappa(p_df[self.cd_dict_m[param_key] + "_x"], p_df[self.cd_dict_m[param_key] + "_y"],
                                           labels = labels_list)
            ctab_df = pd.crosstab(p_df[self.cd_dict_m[param_key] + "_x"], p_df[self.cd_dict_m[param_key] + "_y"])
            ctab = confusion_matrix(p_df[self.cd_dict_m[param_key] + "_x"], p_df[self.cd_dict_m[param_key] + "_y"], 
                                         normalize = "pred",labels = labels_list)
            str_1 = "Rater " + str(idx + 1)
            str_2 = "GT"
            agg_df = ctab_df.melt()
            agg_df.columns = ["GT score", "Count"]
            agg_df["GT score"] =  agg_df["GT score"].astype("int")
            l_list = sorted(set(agg_df["GT score"]))
            l_list_manual = sorted(set(ctab_df.index))
            agg_df.insert(1, "Manual score",l_list_manual * len(l_list))
            agg_df.insert(0, "Pathologist", [patho] * len(agg_df))
            agg_df.insert(0, "Parameter", [self.names_dict[param_key]] * len(agg_df))
            for score in l_list:
                score_len = len(p_df.loc[p_df[self.cd_dict_m[param_key] + "_y"] == score])
                p_len = len(agg_df.loc[agg_df["GT score"] == score])
                agg_df.loc[agg_df["GT score"] == score, "Total"]= [score_len] * p_len
            agg_df["Percent_Agreement"] = np.round(agg_df["Count"]/agg_df["Total"] * 100, 6)
            # agg_df_dict[patho] = ctab_df
            agg_df = agg_df.loc[agg_df["Count"] != 0]
            agg_df_list.append(agg_df)
        out_df = pd.concat(agg_df_list)
        return out_df
    def compute_avg_agreement_gt_only(self,
                                      gt_initial_df, param_key,return_kappa = False):
        a_df = self.get_accuracy_df(gt_initial_df, param_key, method = "GT")
        # g_df = self.gs_df
        # merge_df = 
        labels_list = self.labels_dict[param_key]
        linear_kappa = self._get_kappa(a_df["PathA"], a_df["PathB"],
                                       labels = labels_list)
        #return agg_rate_ceil, agg_rate_floor
        ctab_df = pd.crosstab(a_df["PathA"], a_df["PathB"])
        out_path = self.files_path
        out_df = ctab_df.melt()
        out_df.columns = ["GT2 score", "Count"]
        out_df["GT2 score"] =  out_df["GT2 score"].astype("int")
        out_df.insert(0, "GT1 score",labels_list * len(labels_list))
        out_df.insert(0, "Parameter", [self.names_dict[param_key]] * len(out_df))
        # return out_df
        if return_kappa:
            return linear_kappa, int(len(a_df))
        else:
            return out_df
        # return ctab_df
    def get_avg_agreement_rate_df(self,df, method = "GT" ):
        out_list = []
        param_list = sorted(self.cd_dict_m.keys())
        for param in param_list:
            if method =="Manual":
                avg_agg_df = self.compute_avg_agreement_manual(df,param)
            elif method == "GT":
                avg_agg_df = self.compute_avg_agreement_gt_only(df,param)
            else:
                avg_agg_df = self.compute_avg_agreement(df,param)
            out_list.append(avg_agg_df)
        out_df = pd.concat(out_list)
        return out_df  
    def get_kappa_df_gt_only(self,df):
        out_dict = {}
        param_list = sorted(self.cd_dict_m.keys())
        for param in param_list:
            out_dict[self.names_dict[param]] = self.compute_avg_agreement_gt_only(df, param,
                                                                                 return_kappa = True)
            
        out_df = pd.DataFrame(out_dict).T
        out_df.columns = ["Linear_Kappa", "N"]
        out_df.insert(0, "Parameter", list(out_df.index))
        return out_df
    def _parse_spike_cases(self,identifier, df):
        id_df = df.loc[df.identifier == identifier]
        if len(id_df.loc[np.logical_and(id_df.participant_id.str.contains("v"), id_df.case_label == "Base case")]) > 0:
            id_df.loc[id_df["case_label"] != "Base case", "spike_read"] = id_df.loc[id_df["case_label"] != "Base case", "spike_read"] - 1
        spike_read_list = sorted(set(id_df.spike_read))
        drop_list = []
        for s_read in spike_read_list:
            if s_read > 1:
                s_df = id_df.loc[id_df.spike_read == s_read]
                o_reads = sorted(set(spike_read_list) - set([s_read]))
                o_reads = [o_read for o_read in o_reads if o_read < s_read]
                s_pathos = sorted(set(s_df.pathologist))
                # for o_read in o_reads:
                o_pathos = sorted(set(id_df.loc[id_df.spike_read.isin(o_reads)]["pathologist"]))
                if len(set.intersection(set(s_pathos), set(o_pathos))) > 0:
                    drop_list.append(list(s_df.participant_id))
        if len(drop_list) > 0:
            drop_list = list(itertools.chain.from_iterable(drop_list))
            id_df = id_df.loc[~id_df.participant_id.isin(drop_list)]
        else:
            id_df = id_df
        id_df = id_df.loc[id_df.read_type == "Initial"]
        return id_df
    def _get_final_score_patho(self,row, param):
        patho_suffix_aim = self.patho_suffix_aim
        agree_suffix_aim = self.agree_suffix_aim
        ml_suffix_aim = self.ml_suffix_aim
        if param in ["b", "s", "i"]:
            adeq_col = "he_sample_adequacy"
            two_stage_str = "No - 2 Stage Disagreement"
        else:
            adeq_col = "tri_sample_adequacy"
            two_stage_str = "No - 2 Stage Agreement"
        if row[cd_dict_p[param] + agree_suffix_aim] == "Not Applicable" or row[cd_dict_p[param] + patho_suffix_aim] == "Not Evaluable" or row[adeq_col] == "No":
            out_val = np.nan
        elif row[cd_dict_p[param] + agree_suffix_aim] == two_stage_str:
            out_val = row[cd_dict_p[param] + patho_suffix_aim]
        else:
            out_val = row[cd_dict_p[param] + ml_suffix_aim]
        return out_val
    def _get_spike_read_label(self,row):
        p_id = row["participant_id"]
        if p_id.find("v") != -1 and row["case_label"] != "Base case":
            out_val = int(p_id.split("v")[1])
        else:
            out_val = 1
        return out_val
    def process_spike_df(self, aim_df,aim_df_processed, param, drop_stage_1 = False, return_spike_case_ids = False):
        cd_dict_a = self.cd_dict_a1
        a_df = aim_df_processed.copy()
        a_df = a_df.loc[a_df.read_type != "Consensus"]
        agree_suffix_aim = self.agree_suffix_aim 
        # a_df["steatosis_final_score"] = a_df.apply(get_final_score_patho, args = ("s"), axis = 1)
        a_df["spike_read"] = a_df.apply(self._get_spike_read_label, axis = 1)
        a_df = a_df[["read_type", "identifier", "participant_id", "pathologist", "case_label", "spike_read",
                     self.cd_dict_p[param]+agree_suffix_aim]].drop_duplicates()
        if param in ["b", "s", "i"]:
            agree_str = 'No - 1 Stage Disagreement'
        else:
            agree_str = 'No - 1 Stage Agreement'
        a_df["pathologist"] = a_df["pathologist"].replace({"Amitabh Srivastava" : "AMITABH SRIVASTAVA",
                                                           "Sandy Liu":"Sandy  Liu"})
        a_df1 = a_df.loc[~a_df.identifier.str.contains("v")]
        a_df2 = a_df.loc[a_df.identifier.str.contains("v")]
        a_df2["identifier"] = [idt.split("v")[0][:-1] for idt in list(a_df2["identifier"])]
        a_df_processed = pd.concat([a_df1, a_df2])
        spike_case_ids = sorted(set(a_df_processed.loc[a_df_processed.case_label == "Spike case"]["identifier"]))
        if return_spike_case_ids:
            return spike_case_ids
        else:
            spike_cases_df = a_df_processed.loc[a_df_processed.identifier.isin(spike_case_ids)]
            spike_case_df_processed = pd.concat([self._parse_spike_cases(identifier, spike_cases_df) for identifier in spike_case_ids])
            spike_case_ids_v2 = sorted(set([identifier for identifier in sorted(set(spike_case_df_processed.identifier)) if "Spike case" in list(spike_case_df_processed.loc[spike_case_df_processed.identifier == identifier]["case_label"])]))
            spike_case_df_processed_v2 = spike_case_df_processed.loc[spike_case_df_processed.identifier.isin(spike_case_ids_v2)]
            # spike_case_df_processed_v2 = spike_case_df_processed_v2.loc[spike_case_df_processed_v2.read_type == "Initial"]
            spike_df1 = spike_case_df_processed_v2.loc[np.logical_and(spike_case_df_processed_v2.case_label == "Base case", 
                                                          spike_case_df_processed_v2.participant_id.str.contains("v"))]

            spike_df2 = spike_case_df_processed_v2.loc[~spike_case_df_processed_v2.participant_id.isin(spike_df1.participant_id)]
            spike_df1 = spike_df1.merge(aim_df[["identifier",cd_dict_a[param]]].drop_duplicates(),
                                                                          on = ["identifier"], how = "inner")
            spike_df2 = spike_df2.merge(aim_df[["identifier",cd_dict_a[param]]].drop_duplicates(),
                                                                          left_on = ["participant_id"],
                                        right_on = ["identifier"], how = "inner")
            spike_df2 = spike_df2.drop("identifier_y", axis =1)
            spike_df2 = spike_df2.rename(columns = {"identifier_x":"identifier"})
            spike_case_df_processed_v3 = pd.concat([spike_df1, spike_df2])
            spike_df_list = []
            for idt in sorted(set(spike_case_df_processed_v3.identifier)):
                id_df = spike_case_df_processed_v3.loc[spike_case_df_processed_v3.identifier == idt]
                id_df = id_df.sort_values(by = "participant_id")
                id_df["spike_read"] = list(range(1, len(id_df) + 1))
                spike_df_list.append(id_df)
            spike_df_processed_final= pd.concat(spike_df_list)
            if drop_stage_1:
                spike_df_processed_final = spike_df_processed_final.loc[spike_df_processed_final[self.cd_dict_p[param]+agree_suffix_aim] != agree_str] 
            pvt_df = pd.pivot(spike_df_processed_final,
                     index = ["identifier"],columns = ["spike_read"], values = [cd_dict_a[param]])
            pvt_df.columns = [col[1] for col in list(pvt_df)]
            return pvt_df
    def get_interrater_agg_rate(self, pvt_df, param):
        comb_list = list(itertools.combinations(list(pvt_df),2))
        comb_df = pd.DataFrame()
        c_list = []
        for comb in comb_list:
            r1 = comb[0]
            r2 = comb[1]
            str_1 = "Rater " + str(r1)
            str_2 = "Rater " + str(r2)
            str_index = str_1 + " vs. " + str_2
            c_list.append(r1)
            c_df= pvt_df[[r1,r2]].dropna()
            total = len(c_df)
            if total == 0:
                agg_rate = np.nan
                agg_counts = np.nan
            else:
                agg_counts = len(c_df.loc[c_df[r1] == c_df[r2]])
                agg_rate = (agg_counts/total) * 100
            comb_df.loc[str_index, "Counts_Agree"] = agg_counts
            comb_df.loc[str_index, "Total"] = total
            comb_df.loc[str_index, "Agg_rate"] = agg_rate
            comb_df.loc[str_index, "Counts_Disagree"] = total - agg_counts
            comb_df.loc[str_index, "Disagg_rate"] = 100 - agg_rate
        comb_df.insert(0, "Parameter", [self.cd_dict_m[param]] * len(comb_df))
        comb_df.insert(0, "Comparison", list(comb_df.index))
        comb_df = comb_df.reset_index(drop = True)
        # comb_df.reset_index(inplace = True, names = "Comparison")
        return comb_df
    def get_interrater_kappa(self, pvt_df, param):
        comb_list = list(itertools.combinations(list(pvt_df),2))
        comb_df = pd.DataFrame()
        label_list = self.labels_dict[param]
        c_list = []
        for comb in comb_list:
            r1 = comb[0]
            r2 = comb[1]
            str_1 = "Rater " + str(r1)
            str_2 = "Rater " + str(r2)
            str_index = str_1 + " vs. " + str_2
            c_list.append(r1)
            c_df= pvt_df[[r1,r2]].dropna()
            # agg_counts = len(c_df.loc[c_df[r1] == c_df[r2]])
            total = len(c_df)
            if total == 0:
                linear_kappa = np.nan
            else:
                linear_kappa = self._get_kappa(c_df[r1], c_df[r2], labels = label_list)
            comb_df.loc[str_index, "Total"] = total
            comb_df.loc[str_index, "Linear_Kappa"] = linear_kappa
        comb_df.insert(0, "Parameter", [self.cd_dict_m[param]] * len(comb_df))
        comb_df.insert(0, "Comparison", list(comb_df.index))
        comb_df = comb_df.reset_index(drop = True)
        # comb_df.reset_index(inplace = True, names = "Comparison")
        return comb_df
    def compile_interrater_agreement(self, aim_df,aim_df_processed, metric_type = "Agreement",drop_stage_1 = False):
        cd_dict_p = self.cd_dict_p
        agg_df_list = []
        for param in list(cd_dict_p.keys()):
            pvt_df = self.process_spike_df(aim_df,aim_df_processed,param, drop_stage_1 = drop_stage_1)
            if metric_type == "Agreement":
                agg_rate_df = self.get_interrater_agg_rate(pvt_df, param)
            else:
                agg_rate_df = self.get_interrater_kappa(pvt_df, param)
            agg_df_list.append(agg_rate_df)
        agg_df = pd.concat(agg_df_list)
        return agg_df        
    
    def process_intra_rater_df(self,aim_df, aim_df_processed, rr_df,gt_df, param, drop_stage_1 = False, gt = True):
        cd_dict_a = self.cd_dict_a1
        agree_suffix_aim = self.agree_suffix_aim
        a_df = aim_df_processed.copy()
        a_df = a_df.loc[a_df.read_type != "Consensus"]
        # a_df[cd_dict_a[param]] = a_df.apply(self._get_final_score_patho, args = (param), axis = 1)
        a_df = a_df[["identifier", "participant_id", "pathologist", "case_label","read_type",
                    self.cd_dict_p[param] + agree_suffix_aim]].drop_duplicates()
        a_df["pathologist"] = a_df["pathologist"].replace({"Amitabh Srivastava" : "AMITABH SRIVASTAVA",
                                                                   "Sandy Liu":"Sandy  Liu"})

        a_df1 = a_df.loc[~a_df.identifier.str.contains("v")]
        a_df2 = a_df.loc[a_df.identifier.str.contains("v")]
        a_df2["identifier"] = [idt.split("v")[0][:-1] for idt in list(a_df2["identifier"])]
        a_df_processed = pd.concat([a_df1, a_df2])
        m_df = rr_df[[self.participant_col_name, "user_name",  "slide ID",self.cd_dict_m[param]]].drop_duplicates().dropna()
        merge_df = a_df_processed.merge(m_df,
                                        left_on = [self.participant_col_name_aim, "pathologist"],
                                        right_on = [self.participant_col_name,"user_name"], 
                                        how = "inner").drop_duplicates()
        
        spike_case_ids = sorted(set(merge_df.loc[merge_df.case_label == "Spike case"]["identifier"]))
        spike_cases_df = merge_df.loc[merge_df.identifier.isin(spike_case_ids)]
        nonspike_cases_df = merge_df.loc[~merge_df.identifier.isin(spike_case_ids)]
        spike_cases_df["spike_read"] = spike_cases_df.apply(self._get_spike_read_label, axis = 1)
        spike_case_df_processed = pd.concat([self._parse_spike_cases(identifier, spike_cases_df) for identifier in spike_case_ids])                
        spike_df1 = spike_case_df_processed.loc[np.logical_and(spike_case_df_processed.case_label == "Base case", 
                                                      spike_case_df_processed.participant_id.str.contains("v"))]
        
        spike_df2 = spike_case_df_processed.loc[~spike_case_df_processed.participant_id.isin(spike_df1.participant_id)]
        nonspike_cases_df = nonspike_cases_df.merge(aim_df[["identifier",cd_dict_a[param]]].drop_duplicates(),
                                                                      on = ["identifier"], how = "inner")
        spike_df1 = spike_df1.merge(aim_df[["identifier",cd_dict_a[param]]].drop_duplicates(),
                                                                      on = ["identifier"], how = "inner")
        spike_df2 = spike_df2.merge(aim_df[["identifier",cd_dict_a[param]]].drop_duplicates(),
                                                                      left_on = ["participant_id"],
                                    right_on = ["identifier"], how = "inner")
        spike_df2 = spike_df2.drop("identifier_y", axis =1)
        spike_df2 = spike_df2.rename(columns = {"identifier_x":"identifier"})
        spike_df2 = spike_df2.drop(["spike_read"], axis = 1)
        spike_df1 = spike_df1.drop(["spike_read"], axis = 1)
        merge_df1 = pd.concat([nonspike_cases_df, spike_df1, spike_df2])
        # spike_case_df_processed = pd.concat([self._parse_spike_cases(identifier, spike_cases_df,
        #                                                             interrater = False) for identifier in spike_case_ids])        
        # spike_case_df_processed = spike_case_df_processed.drop(["spike_read"], axis = 1)
        # merge_df1 = pd.concat([nonspike_cases_df, spike_case_df_processed])
        merge_df1 = merge_df1.dropna().drop_duplicates()
        merge_df1["identifier_patho"] = merge_df1["identifier"] + "_" + merge_df1["pathologist"]
        merge_df_processed =  pd.concat([self._parse_intrarater_patho_df(id_patho, merge_df1, param) for id_patho in sorted(set(merge_df1.identifier_patho))])
        if param in ["b", "s", "i"]:
            agree_str = 'No - 1 Stage Disagreement'
        else:
            agree_str = 'No - 1 Stage Agreement'
        merge_df_processed = merge_df_processed.loc[merge_df_processed.read_type == "Initial"]
        if drop_stage_1:
            merge_df_processed = merge_df_processed.loc[merge_df_processed[self.cd_dict_p[param]+agree_suffix_aim] != agree_str]
        if param == "f":
            gs_slide_col = "Trichrome Slide"
        else:
            gs_slide_col = "H & E Slide"
        if gt:
            g_df = gt_df[[self.participant_col_name, self.cd_dict_m[param], gs_slide_col]].drop_duplicates()
            merge_df_processed = merge_df_processed.merge(g_df, left_on = [self.participant_col_name, "slide ID"],
                                  right_on = [self.participant_col_name,gs_slide_col],
                          how = "inner")
        return merge_df_processed
    def _parse_intrarater_patho_df(self,id_patho, df, param):
        id_df = df.loc[df.identifier_patho == id_patho]
        if len(id_df) == 1:
            out_df = id_df.copy()
        else:
            if "Base case" in list(id_df["case_label"]):
                out_df = id_df.loc[id_df.case_label == "Base case"]
                if len(out_df) > 1:
                    out_df= out_df.loc[out_df.read_type == "Initial"]
            else:
                id_df["spike_read"] = id_df.apply(self._get_spike_read_label, axis = 1)
                out_df = pd.DataFrame(id_df.sort_values(by = "spike_read").iloc[0, :]).T
        assert len(out_df) == 1, "Duplicate rows seen for " + id_patho
        return out_df
    def get_intrarater_kappa(self, aim_df,aim_df_processed, rr_df,gt_df, drop_stage_1 = False):
        out_list = []
        out_df_accuracy_list = []
        for param in list(self.cd_dict_p.keys()):
            merge_df = self.process_intra_rater_df(aim_df,aim_df_processed, rr_df,gt_df, param, drop_stage_1 = drop_stage_1)
            patho_list = sorted(set(merge_df.user_name))
            col1 = self.cd_dict_p[param] + "_final_score"
            col2 = self.cd_dict_m[param] + "_x"
            col_gt = self.cd_dict_m[param] + "_y"
            label_list = self.labels_dict[param]
            out_df_patho = pd.DataFrame()
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                p_df1 = p_df[[col1, col_gt]].dropna()
                p_df2 = p_df[[col2, col_gt]].dropna()
                kappa1 = self._get_kappa(p_df1[col1], p_df1[col_gt], labels = label_list)
                kappa2 =  self._get_kappa(p_df2[col2], p_df2[col_gt], labels = label_list)
                out_df_patho.loc[patho, "Kappa_AIM_CV"] = kappa1
                out_df_patho.loc[patho, "Kappa_Manual"] = kappa2
                out_df_patho.loc[patho, "N"] = len(p_df1)
            out_df_patho.loc["Mean Kappa", "Kappa_AIM_CV"] = np.nanmean(out_df_patho["Kappa_AIM_CV"])
            out_df_patho.loc["Mean Kappa", "Kappa_Manual"] = np.nanmean(out_df_patho["Kappa_Manual"])
            out_df_patho.loc["Mean Kappa", "N"] = len(merge_df.dropna())
            out_df_patho.insert(0, "Pathologist", list(out_df_patho.index))
            out_df_patho.insert(0, "Parameter", [self.cd_dict_m[param]] * len(out_df_patho))
            method_df = pd.DataFrame()
            method_list = ["AIM vs. GT", "Manual vs. GT"]
            kappa_list = ["Kappa_AIM_CV", "Kappa_Manual" ]
            for idx, method in list(enumerate(method_list)):
                method_df.loc[method, "Kappa"] = out_df_patho.loc["Mean Kappa",kappa_list[idx]]
                # method_df.loc["Manual vs. GT", "Kappa"] = out_df_patho.loc["Mean Kappa","Kappa_Manual"]
            method_df.loc["AIM vs. GT", "N"] = len(merge_df.dropna(subset = [col1, col_gt]))
            method_df.loc["Manual vs. GT", "N"] = len(merge_df.dropna(subset = [col2, col_gt]))
            method_df.loc["AIM vs. GT", "Difference"] =method_df.loc["AIM vs. GT", "Kappa"] - method_df.loc["Manual vs. GT", "Kappa"]
            method_df.insert(0, "Method", list(method_df.index))
            method_df.insert(0, "Parameter", [self.cd_dict_m[param]] * len(method_df))
            method_df = method_df.reset_index(drop = True)
            out_df_accuracy_list.append(method_df)
            out_list.append(out_df_patho)
        out_df_accuracy = pd.concat(out_df_accuracy_list)
        out_df = pd.concat(out_list)
        out_df = out_df.reset_index(drop = True)
        return out_df, out_df_accuracy
    def get_interrater_kappa_rr(self, aim_df,aim_df_processed,param, rr_df, return_pvt_df = True):
        spike_case_ids = self.process_spike_df(aim_df,aim_df_processed,"s", drop_stage_1 = False,
                                               return_spike_case_ids = True)
        rr_spike_df = rr_df.loc[rr_df.identifier.isin(spike_case_ids)]
        rr_spike_df = rr_spike_df[[self.participant_col_name, "user_name", self.cd_dict_m[param]]]
        rr_spike_df =rr_spike_df.dropna(subset = [self.cd_dict_m[param]])
        manual_df_list = []
        for idt in sorted(set(rr_spike_df[self.participant_col_name])):
            id_df = rr_spike_df.loc[rr_spike_df[self.participant_col_name] == idt]
            id_df = id_df.sort_values(by = "user_name")
            id_df["Read Number"] = [i + 1 for i in range(len(id_df))]
            manual_df_list.append(id_df)
        manual_df_processed = pd.concat(manual_df_list)
        pvt_df = pd.pivot(manual_df_processed, index = [self.participant_col_name],
                          columns = ["Read Number"], values = [self.cd_dict_m[param]])
        pvt_df.columns = [col[1] for col in list(pvt_df)]
        pvt_df.columns = ["R" + str(i+1) for i in range(len(list(pvt_df)))]
        if return_pvt_df:
            return pvt_df
        else: 
            comb_list = list(itertools.combinations(list(pvt_df),2))
            comb_df = pd.DataFrame()
            label_list = list(self.labels_dict[param])
            param_name = self.cd_dict_m[param]
            c_list = []
            for comb in comb_list:
                r1 = comb[0]
                r2 = comb[1]
                str_index = r1 + " vs. " + r2
                c_df= pvt_df[[r1,r2]].dropna()
                total = len(c_df)
                comb_df.loc[str_index, "Total"] = total
                linear_kappa = self._get_kappa(c_df[r1], c_df[r2], labels = label_list)
                comb_df.loc[str_index, "Linear_Kappa"] = np.round(linear_kappa, 6)
            comb_df.insert(0, "Parameter", [param_name] * len(comb_df))
            comb_df.insert(0, "Comparison", list(comb_df.index))
            comb_df = comb_df.reset_index(drop = True)
            return comb_df
    def compile_interrater_kappa_rr(self, aim_df,aim_df_processed, rr_df):
        param_list = self.cd_dict_m.keys()
        param_kappa_df = pd.concat([self.get_interrater_kappa_rr(aim_df,aim_df_processed,param, rr_df) for param in param_list])
        return param_kappa_df
    
    def get_missing_slides_freq_gt(self,processed_df_gt, gs_initial_df_final,gt_missing_df):
        cd_dict_m= self.cd_dict_m
        out_df_list = []
        for param_key in list(cd_dict_m.keys()):
            if param_key == "f":
                stain = "Trichrome"
                slide_col = "Trichrome Slide"
            else:
                stain = "H&E"
                slide_col= "H & E Slide"
            missing_ids = sorted(set(processed_df_gt.loc[processed_df_gt[cd_dict_m[param_key]].isnull()]["identifier"]))
            non_missing_ids = sorted(set(processed_df_gt.loc[~processed_df_gt[cd_dict_m[param_key]].isnull()]["identifier"]))
            gt_missing_id_df = gt_missing_df.loc[gt_missing_df.identifier.isin(missing_ids)]
            gt_missing_id_df = gt_missing_id_df.loc[gt_missing_id_df.stain == stain]
            gt_missing_id_df = gt_missing_id_df[["identifier", "slide_ID", "Category"]].dropna().drop_duplicates()
            diff_ids = set(missing_ids) - set(gt_missing_id_df.identifier)
            gt_sample_missing_df = processed_df_gt.loc[processed_df_gt.identifier.isin(missing_ids)]
            gt_sample_missing_df = gt_sample_missing_df.loc[gt_sample_missing_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "No"]
            a_df3 = pd.DataFrame({"identifier":list(gt_sample_missing_df.identifier),
                                  "slide_ID":list(gt_sample_missing_df[slide_col]),
                                     "Category":["Sample"]*len(gt_sample_missing_df)})
            init_df_missing = gs_initial_df_final.loc[gs_initial_df_final["slide ID"].isin(a_df3["slide_ID"])][["user_name", "slide ID", "NASH biopsy adequacy"]]     
            if len(diff_ids) > 0:
                diff_df = processed_df_gt.loc[processed_df_gt[cd_dict_m[param_key]].isnull()]
                diff_df = diff_df.loc[diff_df.identifier.isin(diff_ids)]
                diff_df1 = diff_df.loc[diff_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "No"]
                diff_df2 = diff_df.loc[diff_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "Yes"]
                a_df1 = pd.DataFrame({"identifier":list(diff_df1.identifier), "slide_ID":list(diff_df1[slide_col]),
                                     "Category":["Sample"]*len(diff_df1)})
                a_df2 = pd.DataFrame({"identifier":list(diff_df2.identifier), "slide_ID":list(diff_df2[slide_col]),
                                     "Category":["Other/Reason not provided"]*len(diff_df2)})
                gt_missing_id_df = pd.concat([gt_missing_id_df, a_df1, a_df2, a_df3])
            else:
                gt_missing_id_df = pd.concat([gt_missing_id_df, a_df3])
            missing_list = []
            for slide_id in sorted(set(gt_missing_id_df.slide_ID)):
                s_df = gt_missing_id_df.loc[gt_missing_id_df.slide_ID == slide_id]
                if len(s_df) == 1:
                    c_str = s_df.Category.values[0]
                else:
                    c_str_list = sorted(set(s_df.Category))
                    c_str = ",".join(c_str_list)
                missing_list.append([slide_id, c_str])
            m_df = pd.DataFrame(missing_list,columns = ["slide_ID", "Category"]).drop_duplicates()
            counts_df = pd.DataFrame(m_df.Category.value_counts())
            counts_df = counts_df.rename(columns = {"Category":"Counts"})
            missing_total = np.sum(counts_df["Counts"])
            assert missing_total + len(non_missing_ids) == len(set(processed_df_gt.identifier)), "Error adding up missing and non-missing ids"
            counts_df.loc["Non-missing", "Counts"] = len(non_missing_ids)
            counts_df["Total"] = int(len(set(processed_df_gt.identifier)))
            counts_df.insert(0, "Reason", list(counts_df.index))
            counts_df.insert(0, "Parameter", [cd_dict_m[param_key]]*len(counts_df))
            counts_df["Percent"] = np.round((counts_df["Counts"]/counts_df["Total"]) * 100, 6)
            out_df_list.append(counts_df)
        return pd.concat(out_df_list).reset_index(drop = True)
    def get_missing_slides_freq_rr(self,processed_df_rr, manifest_df, rr_missing_df):
        cd_dict_m = self.cd_dict_m
        out_df_list = []
        for param_key in list(cd_dict_m.keys()):
            if param_key == "f":
                stain = "Trichrome"
            else:
                stain = "H&E"
            rr_df_m = processed_df_rr.merge(manifest_df[["slide_id", "stain"]],left_on = ["slide ID"], right_on = ["slide_id"],
                                            how = "inner")
            rr_df_m= rr_df_m[["slide ID", "identifier","stain","user_name", cd_dict_m[param_key]]].drop_duplicates()
            rr_df_m = rr_df_m.loc[rr_df_m.stain == stain]
            pvt_df = pd.pivot(rr_df_m, index = ["slide ID"], columns = ["user_name"], values = [cd_dict_m[param_key]])
            pvt_df.columns = [col[1] for col in list(pvt_df)]
            pvt_df["nan_median"] = pvt_df.apply(lambda x: np.nanmedian(x), axis = 1)
            rr_df_missing_ids = sorted(set(pvt_df.loc[pvt_df.nan_median.isnull()].index))
            non_missing_ids = sorted(set(pvt_df.loc[~pvt_df.nan_median.isnull()].index))
            rr_df_missing_id_df = rr_missing_df.loc[rr_missing_df.slide_ID.isin(rr_df_missing_ids)]
            rr_df_missing_id_df = rr_df_missing_id_df[["slide_ID", "identifier", "Category"]].dropna().drop_duplicates()
            # rr_df_missing_id_df.Category.value_counts()
            counts_df = pd.DataFrame(rr_df_missing_id_df.Category.value_counts())
            counts_df = counts_df.rename(columns = {"Category":"Counts"})
            missing_total = np.sum(counts_df["Counts"])
            assert missing_total + len(non_missing_ids) == len(pvt_df), "Error adding up missing and non-missing ids"
            counts_df.loc["Non-missing", "Counts"] = len(non_missing_ids)
            counts_df["Total"] = int(len(pvt_df))
            counts_df.insert(0, "Reason", list(counts_df.index))
            counts_df.insert(0, "Parameter", [cd_dict_m[param_key]]*len(counts_df))
            counts_df["Percent"] = np.round((counts_df["Counts"]/counts_df["Total"]) * 100, 6)
            out_df_list.append(counts_df)
        return pd.concat(out_df_list).reset_index(drop = True)
    def get_aim_missingness_table(self,aim_final_df,aim_df,aim_missing_df):
        cd_dict_a = self.cd_dict_p
        ml_suffix_aim = self.ml_suffix_aim
        agree_suffix_aim = self.agree_suffix_aim
        patho_suffix_aim = self.patho_suffix_aim
        cd_dict_m = self.cd_dict_m
        out_df_list = []
        for param_key in list(cd_dict_a.keys()):
            if param_key in ["s", "b", "i"]:
                adeq_col = "he_sample_adequacy"
            else:
                adeq_col = "tri_sample_adequacy"
            param_df = aim_df[["read_type","identifier", "participant_id", adeq_col] + [cd_dict_a[param_key] + suff for suff in [patho_suffix_aim,
                                                                                                          ml_suffix_aim, agree_suffix_aim]]]
            # p_df1 =
            aim_nonmissing_ids_final = sorted(set(aim_final_df.loc[~aim_final_df[cd_dict_a[param_key] + "_final_score"].isnull()]["identifier"]))
            
            p_df1 = param_df.loc[param_df[adeq_col] == "No"]
            p_df1 = p_df1.loc[p_df1.read_type != "Secondary"]
            p_df1 = p_df1.loc[~p_df1.identifier.isin(aim_nonmissing_ids_final)]
            non_missing_ids = sorted(set(param_df.identifier) - set(p_df1.identifier))
            total = len(set(param_df.identifier))
            aim_missing_id_df = aim_missing_df.loc[aim_missing_df.identifier.isin(list(set(p_df1.identifier)))]
            aim_missing_id_df = aim_missing_id_df[["identifier", "Category"]].dropna().drop_duplicates()
            counts_df = pd.DataFrame(aim_missing_id_df.Category.value_counts())
            counts_df = counts_df.rename(columns = {"Category":"Counts"})
            missing_total = np.sum(counts_df["Counts"])
            assert missing_total + len(non_missing_ids) == len(set(param_df.identifier)), "Error adding up missing and non-missing ids"
            counts_df.loc["Non-missing", "Counts"] = len(non_missing_ids)
            counts_df["Total"] = int(total)
            counts_df.insert(0, "Reason", list(counts_df.index))
            counts_df.insert(0, "Parameter", [cd_dict_m[param_key]]*len(counts_df))
            counts_df["Percent"] = np.round((counts_df["Counts"]/counts_df["Total"]) * 100, 6)
            out_df_list.append(counts_df)
        return pd.concat(out_df_list).reset_index(drop = True)