"""
Utils file for running NASH DDT AV analysis
"""
import pandas as pd
import numpy as np
from scipy import stats
from functools import reduce
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
import sys
sys.path.append("../../")
from r_models import get_confintervals




class AVPreprocess():
    def __init__(self, na_list, cd_dict_m, sm_dict, special_slides, gt450_slides_df):
        self.na_list = na_list
        self.cd_dict_m = cd_dict_m
        self.sm_dict = sm_dict
        self.special_slides = special_slides
        self.gt450_slides_df = gt450_slides_df
    def _recode_na_cols(self, df, col):
        for na_val in self.na_list:
            df[col] = df[col].replace({na_val:np.nan})
        return df
    def get_gt_score(self, gs_df,gs_panel_df,deviation_df,manifest_df, param_key, drop_gt450_slides = True):
        if param_key == "f":
            stain_type = "Trichrome"
        else:
            stain_type = "H & E"
        gs_panel_df = gs_panel_df[["slide ID", "Identifier", self.cd_dict_m[param_key]]].drop_duplicates()
        gs_panel_df = gs_panel_df.dropna(subset = [self.cd_dict_m[param_key]])
        gs_panel_df = self._recode_na_cols(gs_panel_df, self.cd_dict_m[param_key])
        #print(len(gs_panel_df))
        gs_df = gs_df.merge(manifest_df[["PathAI Subject ID", "slide ID", "stain"]].drop_duplicates(),
                            left_on = ["Identifier", "slide ID"], 
                            right_on = ["PathAI Subject ID",  "slide ID"],how = "inner")
        gs_df = gs_df.loc[gs_df["stain"] == stain_type]
        gs_df= gs_df[["slide ID","Identifier", "user_name","stain", self.cd_dict_m[param_key],
                      "NASH biopsy adequacy"]].drop_duplicates()
    
        gs_df = gs_df.merge(deviation_df[["Identifier","Stain","analysis_type"]].drop_duplicates(),
                            left_on = ["Identifier","stain"], 
                            right_on = ["Identifier","Stain"], how = "left") 
        
        #print(len(set(gs_df["slide ID"])))
        all_slides = sorted(set(gs_df["slide ID"]))
        panel_slides = sorted(set.intersection(set(all_slides), set(gs_panel_df["slide ID"])))
        #print(len(panel_slides))
        panel_slides_deviation_con = sorted(set(gs_df.loc[gs_df["analysis_type"] == "Consensus"]["slide ID"]))
        panel_slides_deviation_maj = sorted(set(gs_df.loc[gs_df["analysis_type"] == "Majority"]["slide ID"]))
        panel_slides = sorted(set(panel_slides) - set(panel_slides_deviation_maj))
        #print(len(panel_slides))
        panel_slides = sorted(set.union(set(panel_slides), set(panel_slides_deviation_con)))
        if param_key in ["s", "i"]:
            for r_slide in self.special_slides:
                panel_slides.remove(r_slide)
        #print(len(panel_slides))
        nonpanel_slides = sorted(set(all_slides) - set(panel_slides))
        #print(len(nonpanel_slides))
        gt_score_df_panel = gs_panel_df.loc[gs_panel_df["slide ID"].isin(panel_slides)]
        gt_score_df1 = gt_score_df_panel[["slide ID", self.cd_dict_m[param_key]]].drop_duplicates()
        gt_score_df1["consensus_type"] = ["panel_consensus"] * len(gt_score_df1)
        panel_adeq_list = []
        for slide in list(gt_score_df1["slide ID"]):
            g_df = gs_df.loc[gs_df["slide ID"] == slide]
            adeq_list = list(g_df["NASH biopsy adequacy"])
            a_score = stats.mode(adeq_list)[0][0]
            panel_adeq_list.append(a_score)
        gt_score_df1["NASH biopsy adequacy"] = panel_adeq_list
        gs_df = gs_df.loc[gs_df["slide ID"].isin(nonpanel_slides)]
        gs_df = self._recode_na_cols(gs_df, self.cd_dict_m[param_key])
        gs_df[cd_dict_m[param_key]] = gs_df[cd_dict_m[param_key]].astype("float")  
        slide_list = sorted(set(gs_df["slide ID"]))
        gt_score = []
        for slide in slide_list:
            g_df = gs_df.loc[gs_df["slide ID"] == slide]
            score_arr = np.array(g_df[self.cd_dict_m[param_key]])
            adeq_list = list(g_df["NASH biopsy adequacy"])
            # if len(score_arr[np.isnan(score_arr)]) > 0:
            #     gt_score.append([slide, np.nan, np.nan, np.nan])
            # else:        
            if len(score_arr) == 2:
                if len(score_arr[np.isnan(score_arr)]) > 0:
                    g_score = np.nan
                    #gt_score.append([slide, np.nan, np.nan, np.nan])
                else: 
                    assert score_arr[0] == score_arr[1]
                    g_score = score_arr[0]
                if adeq_list[0] == adeq_list[1] == "No":
                    gt_score.append([slide,g_score,"two_consensus","No"])
                else:
                    gt_score.append([slide,g_score,"two_consensus", "Yes"])
            else:
                if len(score_arr[np.isnan(score_arr)]) > 1:
                    g_score = np.nan
                    #gt_score.append([slide, np.nan, np.nan, np.nan])
                else:
                    assert len(score_arr) == 3
                    g_score = stats.mode(score_arr)[0][0]
                a_score = stats.mode(adeq_list)[0][0]
                gt_score.append([slide, g_score, "three_consensus", a_score])
        gt_score_df2 = pd.DataFrame(gt_score)
        gt_score_df2.columns = list(gt_score_df1)
        gt_score_df_final = pd.concat([gt_score_df2,gt_score_df1])
        gt_score_df_final= manifest_df[["PathAI Subject ID", "slide ID"]].drop_duplicates().merge(gt_score_df_final,
                                                   on = "slide ID", how = "inner")
        if drop_gt450_slides:
            gt450_slides_df = self.gt450_slides_df
            gt450_slides = sorted(set(gt450_slides_df.loc[gt450_slides_df["Stain"] == stain_type]["Slide ID"]))
            gt_score_df_final = gt_score_df_final.loc[~gt_score_df_final["slide ID"].isin(gt450_slides)]
        gt_score_df_final =  gt_score_df_final.rename(columns = {"slide ID": stain_type + " Slide"})
        
        gt_score_df_final.loc[gt_score_df_final["NASH biopsy adequacy"] == "No", self.cd_dict_m[param_key]] = np.nan

        
        return gt_score_df_final
    def get_gt_scores_all_params(self,gs_df,gs_panel_df,deviation_df,manifest_df, drop_gt450_slides = True):
        out_list = []
        for param in list(["s", "b", "i"]):
            o_df = self.get_gt_score(gs_df,gs_panel_df,deviation_df,manifest_df, param, drop_gt450_slides = drop_gt450_slides)
            out_list.append(o_df)
        he_df = reduce(lambda  left,right: pd.merge(left,right,on=['PathAI Subject ID','H & E Slide'],
                                            how='outer'), out_list)
        
        he_df = he_df.rename(columns = {"consensus_type_x":"consensus_type_" + self.cd_dict_m["s"],
                                          "consensus_type_y":"consensus_type_" + self.cd_dict_m["b"] , 
                                          "consensus_type": "consensus_type_" + self.cd_dict_m["i"]})
        
        
        tc_df = self.get_gt_score( gs_df,gs_panel_df,deviation_df,manifest_df, "f", drop_gt450_slides = drop_gt450_slides)
        tc_df = tc_df.rename(columns = {"NASH biopsy adequacy": "NASH_biopsy_adequacy_" + self.cd_dict_m["f"]})
        out_df = he_df.merge(tc_df, on = ["PathAI Subject ID"],how = "inner")
        out_df = out_df.rename(columns = {"NASH biopsy adequacy_x":"NASH_biopsy_adequacy_" + self.cd_dict_m["s"],
                                          "NASH biopsy adequacy_y":"NASH_biopsy_adequacy_" + self.cd_dict_m["b"] , 
                                          "NASH biopsy adequacy": "NASH_biopsy_adequacy_" + self.cd_dict_m["i"]})
        
        
        out_df= out_df.rename(columns = {"consensus_type": "consensus_type_" + self.cd_dict_m["f"]})
        
        return  out_df
    def compare_sm_gs_table(self, np_df, sm_df, param_key):
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H & E Slide"
        sm_df = sm_df.loc[sm_df.use == "Yes"]
        sm_df = sm_df[["slide_ID", "identifier", self.sm_dict[param_key], "param"]].drop_duplicates()
        sm_df = sm_df.loc[sm_df.param == "Final Ground Truth"]
       
        np_df = np_df[[slide_col, "PathAI Subject ID", self.cd_dict_m[param_key],
                           "consensus_type_" + self.cd_dict_m[param_key],
                      "NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]]].drop_duplicates()
        np_df[self.cd_dict_m[param_key]] = np_df[self.cd_dict_m[param_key]].astype("float")
        merge_df = np_df.merge(sm_df, left_on = [slide_col, "PathAI Subject ID"],
                               right_on = ["slide_ID", "identifier"],how = "inner")
        merge_df["diff_score_" + self.sm_dict[param_key]] = merge_df[self.cd_dict_m[param_key]] - merge_df[self.sm_dict[param_key]]
        return merge_df
    def process_ref_reads(self, ref_reads_in,drop_gt450_slides = True):
        ref_reads_df = ref_reads_in.copy()
        param_list = self.cd_dict_m.keys()
        ref_reads_df["Identifier"] = ref_reads_df["Identifier"].str.rstrip(" ")
        for param in param_list:
            if param == "f":
                ref_reads_df[self.cd_dict_m[param]] = ref_reads_df[self.cd_dict_m[param]].replace({'1a': 1, '1b': 1, '1c': 1})
            ref_reads_df = self._recode_na_cols(ref_reads_df, self.cd_dict_m[param])
            ref_reads_df[self.cd_dict_m[param]] = ref_reads_df[self.cd_dict_m[param]].astype("float")
        if drop_gt450_slides:
            drop_slides = sorted(set(self.gt450_slides_df["Slide ID"]))
            ref_reads_df = ref_reads_df.loc[~ref_reads_df["slide ID"].isin(drop_slides)]
        return ref_reads_df
    # def recode_biopsy_cols(self, df, slide_col):
    def compare_sm_ref_table(self, np_df, sm_df):
        sm_slides = sorted(set(sm_df["slide_ID"]))
        np_slides = sorted(set(np_df["slide ID"]))
        assert len(sm_slides) == len(np_slides), "Inconsistent number of slides found"
        assert len(set(sm_slides) - set(np_slides)) == 0, "Inconsistent set of slides found"
        out_list = []
        for param in list(self.cd_dict_m.keys()):
            p_df_sm = sm_df[["slide_ID", "user_name", self.sm_dict[param]]].drop_duplicates()
            p_df_np = np_df[["slide ID", "user_name", self.cd_dict_m[param]]].drop_duplicates()
            merge_df_param = p_df_sm.merge(p_df_np, left_on = ["slide_ID", "user_name"], right_on = ["slide ID", "user_name"], how = "inner")
            merge_df_param["diff_score_" + self.sm_dict[param]] = merge_df_param[self.cd_dict_m[param]] - merge_df_param[self.sm_dict[param]]
            merge_df_param = merge_df_param.dropna(subset = [self.sm_dict[param], self.cd_dict_m[param]], how = "all")
            out_list.append(merge_df_param)
        return out_list
    def _get_reproduce_rows(self, reproducibility_df, repeatability_df, accuracy_df):
        reproducibility_df["Site"] = ["Site_" +str(1)]*len(reproducibility_df)
        pai_ids= sorted(set(reproducibility_df.Participant))
        a_participants = sorted(set.intersection(set(reproducibility_df.Participant), set(accuracy_df.Participant)))
        r_participants = sorted(set.intersection(set(reproducibility_df.Participant), set(repeatability_df.Participant)))
        repeatability_rows = []
        accuracy_rows = []
        for pai_id in r_participants:
            r_row =  repeatability_df.loc[repeatability_df.Participant == pai_id]
            r_row = r_row.loc[r_row["Time Point"] == "Day 2"]
            repeatability_rows.append(r_row)
        for pai_id in a_participants:
            accuracy_rows.append(accuracy_df.loc[accuracy_df.Participant == pai_id])
        repeatability_rows = [r_row for r_row in repeatability_rows if len(r_row) > 0]
        accuracy_rows = [a_row for a_row in accuracy_rows if len(a_row) > 0]
        r_df = pd.concat(repeatability_rows)
        a_df = pd.concat(accuracy_rows)
        r_df["Site"] = ["Site_" +str(2)]*len(r_df)
        a_df["Site"] = ["Site_" + str(3)]*len(a_df)
        r_df = r_df[list(reproducibility_df)]
        a_df = a_df[list(reproducibility_df)]
        out_df = pd.concat([reproducibility_df, r_df, a_df])
        grp_df = out_df[["Participant", "Site"]].groupby("Participant").count()
        grp_df = grp_df.loc[grp_df.Site < 3]
        non3_participants = list(grp_df.index)
        out_df = out_df.loc[~out_df.Participant.isin(non3_participants)]
        return out_df

    
class AVanalysis():
    """ 
    This class is used for NASH DDT AV analysis.
    
    Attributes:
        cd_dict_m (dict): Dictionary of NASH components short forms and corresponding colnames in GT and Manual reads tables.
        cd_dict_a (dict): Dictionary of NASH components short forms and corresponding colnames in AIM-NASH tables.
        participant_col_name (str): Colname containing participant IDs in GT and Manual reads tables. 
        participant_col_name_aim (str): Colname containing participant IDs in AIM-NASH tables.
        gs_df(pandas.core.frame.DataFrame): GT table containing ground truth scores from gold standard pathos. 
        manifest_df(pandas.core.frame.DataFrame): Manifest table containing all the metadata.
        tp_dict(dict): Dictionary with timepoint values and keys to be used for analysis. 
        special_params(list): List of special parameters (NAS >= 4, F0+F1 and F4) to be used in results.
        files_path(str): Local path to the files to be read in during the analysis. 
        exclude_participants(list): List of string participant ids to be excluded based on deviation log. 
        
    """
    def __init__(self,
                 cd_dict_m, cd_dict_a, participant_col_name, participant_col_name_aim, 
                 gs_df, manifest_df, tp_dict, labels_dict, special_params,special_params_v2, files_path, exclude_participants):
        self.cd_dict_m = cd_dict_m
        self.cd_dict_a = cd_dict_a
        self.participant_col_name= participant_col_name
        self.participant_col_name_aim = participant_col_name_aim
        self.gs_df = gs_df
        self.manifest_df = manifest_df
        self.tp_dict = tp_dict
        self.labels_dict = labels_dict
        self.special_params = special_params
        self.files_path = files_path
        self.exclude_participants = exclude_participants
        self.special_params_v2 = special_params_v2
        #self.expected_rr_length = expected_rr_length
    def _rstrip_col(self,df, col_name):
        """
        Function for striping blank spaces from a column name in a dataframe.
        
        """
        df[col_name] = df[col_name].rstrip(" ")
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
        
    def get_distribution_table_score(self, df, gt = True):
        # if df is None and gt == True:
        #     df = self.gs_df
        # df = self.gs_df
        out_df = pd.concat(self._distribution_table(df, gt = gt))
        out_df = self._round_col(out_df, "Frequency")
        return(out_df)    
    
    def get_slide_distribution_by_sponsor_score(self,sponsor_col,df,gt = True, extra_cols  = []):
        if gt:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col] + extra_cols].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"] + extra_cols,
                             left_on = [self.participant_col_name] + extra_cols,
                                      how = "inner")
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col]+ extra_cols].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"] + extra_cols,
                             left_on = [self.participant_col_name_aim] + extra_cols,
                                      how = "inner")
            cd_dict = self.cd_dict_a
            participant_col = self.participant_col_name_aim
        out_list = []
        for param in list(cd_dict.keys()):
            param_df = in_df[[participant_col, cd_dict[param], sponsor_col]].drop_duplicates().dropna()
            grp_df = param_df.groupby([cd_dict[param], sponsor_col]).count()
            grp_df.insert(0,sponsor_col,[m_idx[1] for m_idx in list(grp_df.index)])
            grp_df.insert(0,"Score",[m_idx[0] for m_idx in list(grp_df.index)])
            grp_df["Score"] = grp_df["Score"].astype("int")
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
    def _get_nas_sum_check(self,row, cd_dict,n_nas, n_sum ):
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
    
    def get_distribution_table_nas_by_sponsor(self,sponsor_col,df,  gt = True, extra_cols = []):
        if gt:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col] + extra_cols].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"] + extra_cols, 
                                     left_on = [self.participant_col_name] + extra_cols,
                                      how = "inner")
            cd_dict = self.cd_dict_m
            participant_col = self.participant_col_name
        else:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col] + extra_cols].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"] + extra_cols,
                             left_on = [self.participant_col_name_aim] + extra_cols,
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
    
    def get_slide_distribution_by_sponsor(self, sponsor_col,df, gt = True):
        if gt:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col]].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"], left_on = [self.participant_col_name],
                                      how = "inner")
        else:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", sponsor_col]].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"], left_on = [self.participant_col_name_aim],
                                      how = "inner")
        freq_df = pd.DataFrame(in_df[sponsor_col].value_counts())
        freq_df = freq_df.rename(columns = {sponsor_col: "Counts"})
        freq_df["Total"] = [freq_df["Counts"].sum()] * len(freq_df)
        freq_df.insert(0, sponsor_col, list(freq_df.index))
        freq_df["Percent"] = np.round(freq_df["Counts"]/freq_df["Total"]*100,2)
        freq_df = freq_df.reset_index(drop = True)
        return freq_df
    def get_slide_distribution_by_timepoint(self, time_col, df, gt = True):
        if gt:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", time_col]].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"], left_on = [self.participant_col_name],
                                      how = "inner")
        else:
            in_df = df.merge(self.manifest_df[["PathAI Subject ID", time_col]].drop_duplicates(),
                                      left_on = ["PathAI Subject ID"], right_on = [self.participant_col_name_aim],
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
    
    def get_per_scanner_distribution(self, df,gt = False):
        manifest_df = self.manifest_df
        if gt:
            participant_col_name = self.participant_col_name
        else:
            participant_col_name = self.participant_col_name_aim
        in_df = df.merge(manifest_df[["PathAI Subject ID", "Scanner"]].drop_duplicates(),
                                      right_on = ["PathAI Subject ID"], left_on = [participant_col_name],
                                      how = "inner")
        out_list = []
        out_list_sponsor = []
        for scanner in sorted(set(in_df["Scanner"])):
            s_df = in_df.loc[in_df.Scanner == scanner]
            scr_df = self.get_distribution_table_score(s_df, gt = gt)
            scr_df_nas = self.get_distribution_table_nas(s_df, gt = gt)
            scr_df_sponsor = self.get_slide_distribution_by_sponsor_score("dataset",df = s_df, gt = gt, extra_cols = ["Scanner"])
            scr_df_nas_sponsor = self.get_distribution_table_nas_by_sponsor( "dataset",df = s_df,gt = gt, extra_cols = ["Scanner"])
            if not gt:
                replace_dict= {}
                for param, value in self.cd_dict_a.items():
                    replace_dict[value] = self.cd_dict_m[param]
                scr_df["Feature"] = scr_df["Feature"].replace(replace_dict)
            scr_df_nas = scr_df_nas[list(scr_df)]
            scr_df_sponsor = scr_df_sponsor.rename(columns = {"Parameter":"Feature",
                                                             "total":"Total", "Counts":"Count", 
                                                             "Percent":"Frequency"})
            scr_df_nas_sponsor = scr_df_nas_sponsor[list(scr_df_sponsor)]
            scanner_df_sponsor = pd.concat([scr_df_sponsor, scr_df_nas_sponsor])
            scanner_df_sponsor.insert(0, "Scanner", [scanner]*len(scanner_df_sponsor))
            scanner_df = pd.concat([scr_df, scr_df_nas])
            scanner_df.insert(0, "Scanner", [scanner]*len(scanner_df))
            out_list.append(scanner_df)
            out_list_sponsor.append(scanner_df_sponsor)
            # out_list_sponsor.append([scr_df_sponsor, scr_df_nas_sponsor])
        # return out_list_sponsor
        # return in_df
        return pd.concat(out_list), pd.concat(out_list_sponsor)
    def get_rr_slide_distribution(self, df, cat):
        cd_dict = self.cd_dict_a
        out_list = []
        participant_col = self.participant_col_name_aim
        if cat == "Site":
            df[cat] = df[cat].replace({"Site_1":"Reproducibility", "Site_2":"Repeatability", "Site_3":"Accuracy"})
        for param in list(cd_dict.keys()):
            param_df =df.copy()
            participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param]] == "No"][self.participant_col_name])
            if cat == "Site":
                participants_to_drop = participants_to_drop + self.exclude_participants
            if param == "f":
                slide_col = "Trichrome Slide"
            else:
                slide_col = "H&E Slide"
            param_df = param_df.loc[param_df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
            param_df = param_df.loc[~param_df[self.participant_col_name_aim].isin(participants_to_drop)]
            param_df = param_df[[participant_col, cd_dict[param], cat]].drop_duplicates().dropna()
            grp_df = param_df.groupby([cd_dict[param], cat]).count()
            grp_df.insert(0,cat,[m_idx[1] for m_idx in list(grp_df.index)])
            grp_df.insert(0,"Score",[m_idx[0] for m_idx in list(grp_df.index)])
            grp_df = grp_df.rename(columns = {participant_col:"Counts"})
            grp_df = grp_df.reset_index(drop = True)
            sum_df = pd.DataFrame()
            n_cats = sorted(set(grp_df[cat]))
            n_score = sorted(set(grp_df["Score"]))
            sum_df = pd.DataFrame()
            for idx,n_cat in list(enumerate(n_cats)):
                # for idx, score in list(enumerate(n_score)):
                sum_df.loc[idx, cat] = n_cat
                sum_df.loc[idx, "total"] = grp_df.loc[grp_df[cat] == n_cat]["Counts"].sum()
            grp_df = grp_df.merge(sum_df, on = cat, how = "inner")
            grp_df["Percent"] = np.round(grp_df["Counts"]/grp_df["total"]*100,2)
            grp_df.insert(0, "Parameter", [self.cd_dict_m[param]]*len(grp_df))
            out_list.append(grp_df)
        out_df = pd.concat(out_list)
        return out_df        
    def _recode_cols(self, df, col, labels_list, alt_val = 9):
        for label in labels_list:
            df[col] = df[col].replace({label:alt_val})
        return df
    def get_bootstrap_samples_itr(self, input_df,metric,sample_col, drop_cols, itr):
        if metric in ["accuracy_aim", "accuracy_aim_pp"]:
            b_df = input_df.sample(len(input_df), replace = True)
        else:
            sample_list = sorted(set(input_df[sample_col]))
            b_samples = random.choices(sample_list, k= len(sample_list))
            b_df =  pd.concat([input_df.loc[input_df[sample_col] == b] for b in b_samples])
        b_df.insert(0, "Iteration", itr)
        b_df = b_df.drop(drop_cols, axis = 1)
        return b_df
    def get_bootstrap_samples_df(self,input_df, metric, sample_col,drop_cols, n_iterations = 2000, write = True,
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
            out_df.to_csv(self.files_path + "NASH_DDT_AV_production_data_bootstrap_" + metric + "_030523.csv", index = False)
        else:
            return out_df      
    def _get_kappa(self, arr1,arr2, labels, weights = "linear"):
        kappa = cohen_kappa_score(arr1, arr2,labels, weights= weights)
        return kappa
    def _get_p_proportion(self, in_list, endpoint = "Accuracy"):
        if endpoint == "Accuracy":
            out_p = len(np.where(np.array(in_list) < -0.1)[0])/len(in_list)
        else:
            out_p= len(np.where(np.array(in_list) <= 0.85)[0])/len(in_list)
        return out_p

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
    def get_f0_f1_accuracy_kappa(self, df,l1 = [0,1], method = "AIM", drop_duplicates = True):
        param_key = "f"
        labels_list = self.labels_dict[param_key]
        l2 = sorted(set(labels_list) - set(l1))
        if method == "AIM":
            if drop_duplicates:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]].drop_duplicates()
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]].drop_duplicates()
            else:
                a_df = df[[self.participant_col_name_aim, self.cd_dict_a[param_key]]]
                g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key]]]
            merge_df = a_df.merge(g_df, left_on = self.participant_col_name_aim,right_on = self.participant_col_name, how = "inner")
            merge_df = merge_df.dropna()
            labels_list = self.labels_dict[param_key]
            m_df = merge_df.copy()
            m_df = self._recode_cols(m_df, self.cd_dict_a[param_key], l1, 8)
            m_df = self._recode_cols(m_df, self.cd_dict_a[param_key], l2, 9)
            m_df = self._recode_cols(m_df, self.cd_dict_m[param_key], l1, 8)
            m_df = self._recode_cols(m_df, self.cd_dict_m[param_key], l2, 9)
            kappa_p = self._get_kappa(m_df[self.cd_dict_a[param_key]], m_df[self.cd_dict_m[param_key]],None, weights = None)
            num_score = len(m_df)
        else:
            gs_slide_col = "Trichrome Slide"
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
            alt_labels = sorted(set(labels_list) - set([0,1]))
            kappa_list = []
            merge_df = self._recode_cols(merge_df, self.cd_dict_m[param_key]+ "_x", l1,8)
            merge_df = self._recode_cols(merge_df, self.cd_dict_m[param_key]+ "_y", l1,8)
            merge_df = self._recode_cols(merge_df, self.cd_dict_m[param_key]+ "_x", l2, 9)
            merge_df = self._recode_cols(merge_df, self.cd_dict_m[param_key]+ "_y", l2, 9)
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df[self.cd_dict_m[param_key]+"_x"], p_df[self.cd_dict_m[param_key]+"_y"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
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
    def get_nas_kappa(self, df, method = "AIM", drop_duplicates = True, refactor = True):
        if refactor:
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
            if refactor:
                m_df = self._recode_cols(m_df, "NAS_AIM", l1,10)
                m_df = self._recode_cols(m_df, "NAS_GT", l1, 10)
                m_df = self._recode_cols(m_df, "NAS_AIM", l2, 11)
                m_df = self._recode_cols(m_df, "NAS_GT", l2, 11)
                kappa_p = self._get_kappa(m_df["NAS_AIM"], m_df["NAS_GT"],None, weights = None)
            else:
                kappa_p = self._get_kappa(m_df["NAS_AIM"], m_df["NAS_GT"],list(range(9)))
            num_score = len(m_df)
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
            if refactor:
                merge_df = self._recode_cols(merge_df, "NAS_M", l1,10)
                merge_df = self._recode_cols(merge_df, "NAS_GT",l1, 10)
                merge_df = self._recode_cols(merge_df, "NAS_M", l2, 11)
                merge_df = self._recode_cols(merge_df, "NAS_GT", l2, 11)
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                if refactor:
                    kappa_p = self._get_kappa(p_df["NAS_M"], p_df["NAS_GT"],None,
                                              weights = None)
                else:
                    kappa_p = self._get_kappa(p_df["NAS_M"], p_df["NAS_GT"],list(range(9)))
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        # if refactor:
        return ["NAS_4", np.round(kappa_p,6), num_score]
    
    def get_nas_kappa_v2(self, df, method = "AIM", drop_duplicates = True):
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
            m_df["NASH_res_AIM"] = m_df.apply(self._get_nas_1, args = (self.cd_dict_a, "NAS_AIM"), axis =1)
            m_df["NASH_res_GT"] = m_df.apply(self._get_nas_1, args = (self.cd_dict_m, "NAS_GT"), axis= 1)
            kappa_p = self._get_kappa(m_df["NASH_res_AIM"], m_df["NASH_res_GT"],None, weights = None)
            num_score = len(m_df)
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
            dict1 = {}
            dict2 = {}
            for param in list(self.cd_dict_m.keys()):
                dict1[param] = self.cd_dict_m[param] + "_x"
                dict2[param] = self.cd_dict_m[param] + "_y"
                
            merge_df["NASH_res_M"] = merge_df.apply(self._get_nas_1, args = (dict1, "NAS_M"), axis =1)
            merge_df["NASH_res_GT"] = merge_df.apply(self._get_nas_1, args = (dict2, "NAS_GT"), axis= 1)
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NASH_res_M"], p_df["NASH_res_GT"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["NAS_4", np.round(kappa_p,6), num_score]  
    
    def get_nash_res_kappa(self, df, method = "AIM", drop_duplicates = True):
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
            m_df["NASH_res_AIM"] = m_df.apply(self._get_nash_res, args = (self.cd_dict_a, "NAS_AIM"), axis =1)
            m_df["NASH_res_GT"] = m_df.apply(self._get_nash_res, args = (self.cd_dict_m, "NAS_GT"), axis= 1)
            kappa_p = self._get_kappa(m_df["NASH_res_AIM"], m_df["NASH_res_GT"],None, weights = None)
            num_score = len(m_df)
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
            dict1 = {}
            dict2 = {}
            for param in list(self.cd_dict_m.keys()):
                dict1[param] = self.cd_dict_m[param] + "_x"
                dict2[param] = self.cd_dict_m[param] + "_y"
                
            merge_df["NASH_res_M"] = merge_df.apply(self._get_nash_res, args = (dict1, "NAS_M"), axis =1)
            merge_df["NASH_res_GT"] = merge_df.apply(self._get_nash_res, args = (dict2, "NAS_GT"), axis= 1)
            for patho in patho_list:
                p_df = merge_df.loc[merge_df.user_name == patho]
                if drop_duplicates:
                    p_df = p_df.drop_duplicates()
                kappa_p = self._get_kappa(p_df["NASH_res_M"], p_df["NASH_res_GT"],None,
                                              weights = None)
                kappa_list.append(kappa_p)
            kappa_list = [k for k in kappa_list if not np.isnan(k)]
            kappa_p = np.mean(kappa_list)
            num_score = len(set(merge_df[self.participant_col_name]))
        return ["NAS_4", np.round(kappa_p,6), num_score]      
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
                num_list = list(itertools.chain.from_iterable([[a_list[i][2],m_list[i][2]] for i in range(len(a_list))]))
                diff_list = list(itertools.chain.from_iterable([[a_list[i][1] - m_list[i][1]]*2 for i in range(len(a_list))]))
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
                                     "Difference": [a_list[0] - m_list[0], np.nan],
                                     "N": [a_list[1], m_list[1]]})
            out_list.append(param_df)
        out_df = pd.concat(out_list)
        return out_df
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
        
    def get_accuracy_df_special_v2(self, accuracy_df, manual_df, drop_duplicates = True):
        special_params = self.special_params_v2
        a_list_f2_f3 = self.get_f0_f1_accuracy_kappa(accuracy_df,l1 = [2,3], method = "AIM", drop_duplicates = drop_duplicates)
        m_list_f2_f3 = self.get_f0_f1_accuracy_kappa(manual_df,l1 = [2,3], method = "Manual", drop_duplicates = drop_duplicates)
        a_list_nas_4 = self.get_nas_kappa_v2(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_nas_4 = self.get_nas_kappa_v2(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        a_list_nash_res = self.get_nash_res_kappa(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates)
        m_list_nash_res = self.get_nash_res_kappa(manual_df, method = "Manual", drop_duplicates = drop_duplicates)
        p_list = list(itertools.chain.from_iterable([[special_params[0]]*2, [special_params[1]]*2, [special_params[2]]*2]))
        d_list = list(itertools.chain.from_iterable([[a_list_f2_f3[1] - m_list_f2_f3[1]]*2, [a_list_nas_4[1] - m_list_nas_4[1]]* 2,
                                                    [a_list_nash_res[1] - m_list_nash_res[1]]*2]))
        
        out_df = pd.DataFrame({"Parameter": p_list,"Method": ["AIM-NASH", "Manual pathologists"] * 3,
                               "Kappa":[a_list_f2_f3[1], m_list_f2_f3[1], a_list_nas_4[1], m_list_nas_4[1],a_list_nash_res[1], m_list_nash_res[1]],
                                 "Difference": d_list, "N": [a_list_f2_f3[2], m_list_f2_f3[2], a_list_nas_4[2], m_list_nas_4[2],
                                                            a_list_nash_res[2],m_list_nash_res[2]]})
        return out_df
    def get_accuracy_df_nas(self, accuracy_df, manual_df, drop_duplicates = True):
        a_list_nas = self.get_nas_kappa(accuracy_df, method = "AIM", drop_duplicates = drop_duplicates, refactor = False)
        m_list_nas = self.get_nas_kappa(manual_df, method = "Manual", drop_duplicates = drop_duplicates,refactor = False)
        p_list = ["NAS"]*2
        d_list = [a_list_nas[1] - m_list_nas[1]]*2
        out_df = pd.DataFrame({"Parameter": p_list,"Method": ["AIM-NASH", "Manual pathologists"],
                               "Kappa":[a_list_nas[1],m_list_nas[1]],
                                 "Difference": d_list, "N":[a_list_nas[2],m_list_nas[2]]})
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
        elif analysis_type == "nas":
            out_df = self.get_accuracy_df_nas(iteration_accuracy_df, iteration_manual_df,drop_duplicates = False) 
        else:
            out_df = self.get_accuracy_df_special(iteration_accuracy_df, iteration_manual_df,drop_duplicates = False)
        out_df.insert(0, "Iteration", [itr] * len(out_df))
        return out_df
    def get_accuracy_df_bootstrap(self, accuracy_df_bootstrap, manual_df_bootstrap,analysis_type, n_iterations = 10):
        partial_func = partial(self.get_accuracy_df_iteration, accuracy_df_bootstrap, manual_df_bootstrap, analysis_type)
        pool = mp.Pool()
        out_list = list(pool.map(partial_func, list(range(n_iterations))))
        del(pool)
        return out_list
    def get_per_timepoint_accuracy(self, accuracy_df, manual_df, drop_duplicates = True):
        manifest_df= self.manifest_df
        tp_dict = self.tp_dict
        out_list = []
        for tp in list(tp_dict.keys()):
            tp_participants = sorted(set(manifest_df.loc[manifest_df.visit.isin(tp_dict[tp])]["PathAI Subject ID"]))
            a_df = accuracy_df.loc[accuracy_df[self.participant_col_name_aim].isin(tp_participants)]
            m_df = manual_df.loc[manual_df[self.participant_col_name].isin(tp_participants)]
            tp_acc_df = self.get_accuracy_df(a_df, m_df, drop_duplicates = drop_duplicates)
            tp_acc_df.insert(0, "Time Point", [tp]*len(tp_acc_df))
            out_list.append(tp_acc_df)
        out_df = pd.concat(out_list)
        return out_df     
    def get_per_sponsor_accuracy(self, accuracy_df, manual_df, drop_duplicates = True):
        manifest_df = self.manifest_df
        datasets = sorted(set(manifest_df.dataset))
        out_list = []
        for dataset in datasets:
            d_participants = sorted(set(manifest_df.loc[manifest_df.dataset == dataset]["PathAI Subject ID"]))
            a_df = accuracy_df.loc[accuracy_df[self.participant_col_name_aim].isin(d_participants)]
            m_df = manual_df.loc[manual_df[self.participant_col_name].isin(d_participants)]
            d_acc_df = self.get_accuracy_df(a_df, m_df, drop_duplicates = drop_duplicates)
            d_acc_df.insert(0, "dataset", [dataset]*len(d_acc_df))
            out_list.append(d_acc_df)
        out_df = pd.concat(out_list)
        return out_df
    def get_per_patho_kappa(self,df, param_key):
        m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
        if param_key == "f":
            gs_slide_col = "Trichrome Slide"
        else:
            gs_slide_col = "H & E Slide"
        m_df = df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
        g_df = self.gs_df[[self.participant_col_name, self.cd_dict_m[param_key], gs_slide_col]].drop_duplicates()
        merge_df = m_df.merge(g_df, left_on = [self.participant_col_name, "slide ID"], right_on = [self.participant_col_name,gs_slide_col],
                      how = "inner")
        merge_df = merge_df.dropna()
        patho_list = sorted(set(merge_df.user_name))
        kappa_list = []
        for patho in patho_list:
            p_df = merge_df.loc[merge_df.user_name == patho]
            p_df = p_df.drop_duplicates()
            kappa_list.append([patho, len(p_df),self._get_kappa(p_df[self.cd_dict_m[param_key]+"_x"], p_df[self.cd_dict_m[param_key]+"_y"],
                                                                self.labels_dict[param_key],
                                                                                    weights = "linear")])
        kappa_df = pd.DataFrame(kappa_list)
        kappa_df.columns = ["Pathologist", "N", "Kappa"]
        return kappa_df
    def _get_reproduce_rows(self, reproducibility_df, repeatability_df, accuracy_df):
        reproducibility_df["Site"] = ["Site_" +str(1)]*len(reproducibility_df)
        pai_ids= sorted(set(reproducibility_df.Participant))
        a_participants = sorted(set.intersection(set(reproducibility_df.Participant), set(accuracy_df.Participant)))
        r_participants = sorted(set.intersection(set(reproducibility_df.Participant), set(repeatability_df.Participant)))
        repeatability_rows = []
        accuracy_rows = []
        for pai_id in r_participants:
            r_row =  repeatability_df.loc[repeatability_df.Participant == pai_id]
            r_row = r_row.loc[r_row["Time Point"] == "Day 2"]
            repeatability_rows.append(r_row)
        for pai_id in a_participants:
            accuracy_rows.append(accuracy_df.loc[accuracy_df.Participant == pai_id])
        repeatability_rows = [r_row for r_row in repeatability_rows if len(r_row) > 0]
        accuracy_rows = [a_row for a_row in accuracy_rows if len(a_row) > 0]
        r_df = pd.concat(repeatability_rows)
        a_df = pd.concat(accuracy_rows)
        r_df["Site"] = ["Site_" +str(2)]*len(r_df)
        a_df["Site"] = ["Site_" + str(3)]*len(a_df)
        r_df = r_df[list(reproducibility_df)]
        a_df = a_df[list(reproducibility_df)]
        out_df = pd.concat([reproducibility_df, r_df, a_df])
        grp_df = out_df[["Participant", "Site"]].groupby("Participant").count()
        grp_df = grp_df.loc[grp_df.Site < 3]
        non3_participants = list(grp_df.index)
        out_df = out_df.loc[~out_df.Participant.isin(non3_participants)]
        return out_df
    def get_rr_mean_agreement(self, df,param_key,d_col = "Time Point", add_participant_suffix = False):
        labels_list = self.labels_dict[param_key]
        participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]] == "No"][self.participant_col_name])
        if d_col != "Time Point":
            participants_to_drop = participants_to_drop + self.exclude_participants
            
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H&E Slide"
        
        cd_dict = self.cd_dict_a
        df = df[[self.participant_col_name_aim,slide_col, d_col, cd_dict[param_key]]]
        df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
        df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
        if add_participant_suffix:
            df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna()
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim, values = cd_dict[param_key])
        pwise_tp = list(itertools.combinations(sorted(set(df[d_col])),2))
        agg_list = []
        for pw in pwise_tp:
            #ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
            ctab_df = confusion_matrix(pvt_df[pw[0]], pvt_df[pw[1]], labels = labels_list)
            agg_rate= np.sum(np.diag(ctab_df))/len(pvt_df)
            agg_list.append(agg_rate)
        return [np.round(np.mean(agg_list),6), len(pvt_df)]
    
    def get_f0_f1_rr_agreement(self,df,d_col = "Time Point", f4 = False,  add_participant_suffix = False):
        param_key = "f"
        labels_list = self.labels_dict[param_key]
        if f4:
            l1 = [4]
        else:
            l1 = [0,1]
        l2 = sorted(set(labels_list) - set(l1))
        participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]] == "No"][self.participant_col_name])
        if d_col != "Time Point":
            participants_to_drop = participants_to_drop + self.exclude_participants
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H&E Slide"
        
        cd_dict = self.cd_dict_a
        df = df[[self.participant_col_name_aim,slide_col, d_col, cd_dict[param_key]]]
        df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
        df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
        df = self._recode_cols(df, self.cd_dict_a[param_key], l1, 8)
        df = self._recode_cols(df, self.cd_dict_a[param_key], l2, 9)
        if add_participant_suffix:
            df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna()
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim, values = cd_dict[param_key])
        pwise_tp = list(itertools.combinations(sorted(pvt_df),2))
        agg_list = []
        for pw in pwise_tp:
            #ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
            ctab_df = confusion_matrix(pvt_df[pw[0]], pvt_df[pw[1]])
            agg_rate= np.sum(np.diag(ctab_df))/len(pvt_df)
            agg_list.append(agg_rate)
        return np.round(np.mean(agg_list),6), len(pvt_df)
    
    def get_nas_rr_agreement(self,df,d_col = "Time Point", add_participant_suffix = False):
        l1 = list(range(4))
        l2 = list(range(4,9))
        participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m["b"]] == "No"][self.participant_col_name])
        if d_col != "Time point":
            participants_to_drop = participants_to_drop + self.exclude_participants
        #slide_col = "H&E Slide"
        cd_dict = self.cd_dict_a
        # df = df[[self.participant_col_name_aim, self.cd_dict_a["s"], self.cd_dict_a["b"],d_col,
        #                   self.cd_dict_a["i"]]].drop_duplicates()
        df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
        df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
        df["NAS_AIM"] = df[self.cd_dict_a["s"]] +  df[self.cd_dict_a["b"]] + df[self.cd_dict_a["i"]]
        df = self._recode_cols(df, "NAS_AIM", l1,10)
        df = self._recode_cols(df, "NAS_AIM", l2, 11)
        if add_participant_suffix:
            df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna(subset = ["NAS_AIM"])
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim, values = "NAS_AIM")
        pwise_tp = list(itertools.combinations(sorted(pvt_df),2))
        agg_list = []
        for pw in pwise_tp:
            #ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
            ctab_df = confusion_matrix(pvt_df[pw[0]], pvt_df[pw[1]])
            agg_rate= np.sum(np.diag(ctab_df))/len(pvt_df)
            agg_list.append(agg_rate)
        return np.round(np.mean(agg_list),6), len(pvt_df)
    
    def get_rr_mean_agreement_length(self, df,param_key,d_col = "Time Point", add_participant_suffix = False):
        labels_list = self.labels_dict[param_key]
        participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]] == "No"][self.participant_col_name])
        if d_col != "Time Point":
            participants_to_drop = participants_to_drop + self.exclude_participants
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H&E Slide"
        
        cd_dict = self.cd_dict_a
        df = df[[self.participant_col_name_aim,slide_col, d_col, cd_dict[param_key]]]
        df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
        d_participants = set.intersection(set(df[self.participant_col_name_aim]),set(participants_to_drop))
        df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
        if add_participant_suffix:
            df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna()
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim, values = cd_dict[param_key])
        return [self.cd_dict_m[param_key], len(pvt_df), len(d_participants)]
    
    def get_rr_lengths_df(self,df, endpoint, add_participant_suffix = False):
        if endpoint == "Repeatability":
            d_col = "Time Point"
        else:
            d_col = "Site"
        #cd_dict = self.cd_dict_a
        out_list = []
        param_list = sorted(self.cd_dict_m.keys())
        for param in param_list:
            length_list = self.get_rr_mean_agreement_length(df, param, d_col = d_col,
                                                            add_participant_suffix = add_participant_suffix)
            length_list = [endpoint] + length_list
            out_list.append(length_list)
        out_df = pd.DataFrame(out_list)
        out_df.columns = ["Endpoint", "Parameter", "N Used", "N Dropped"]
        return out_df
    def _add_participant_suffix(self,df, expected_length):
        o_list = []
        participant_col = self.participant_col_name_aim
        for i in range(expected_length):
            o_list.append([i]*3)
        o_list = list(itertools.chain.from_iterable(o_list))
        df["extra_col"] = o_list
        df[participant_col] = df[participant_col] + "_" + df["extra_col"].astype("str")
        df = df.drop("extra_col", axis = 1)
        assert len(set(df[participant_col])) == expected_length
        return df
    def get_rr_df(self,df, endpoint, add_participant_suffix = False):
        if endpoint == "Repeatability":
            d_col = "Time Point"
        else:
            d_col = "Site"
        out_list = []
        param_list = sorted(self.cd_dict_m.keys())
        for param in param_list:
            rr_agg,rr_n = self.get_rr_mean_agreement(df, param, d_col = d_col, add_participant_suffix = add_participant_suffix)
            out_list.append([endpoint, self.cd_dict_m[param], rr_agg, rr_n])
        out_df = pd.DataFrame(out_list)
        out_df.columns = ["Endpoint", "Parameter", "Mean_Agreement_Rate", "N"]
        return out_df
    def get_per_timepoint_rr_table(self, df, endpoint, add_participant_suffix = False):
        manifest_df= self.manifest_df
        tp_dict = self.tp_dict
        out_list = []
        for tp in list(tp_dict.keys()):
            tp_participants = sorted(set(manifest_df.loc[manifest_df.visit.isin(tp_dict[tp])]["PathAI Subject ID"]))
            r_df = df.loc[df[self.participant_col_name_aim].isin(tp_participants)]
            tp_agg_df = self.get_rr_df(r_df, endpoint, add_participant_suffix = add_participant_suffix)
            tp_agg_df.insert(0, "Time Point", [tp]*len(tp_agg_df))           
            out_list.append(tp_agg_df)
        out_df = pd.concat(out_list)                
        return out_df     
    def get_per_sponsor_rr_table(self, df, endpoint, add_participant_suffix  = False):
        manifest_df = self.manifest_df
        datasets = sorted(set(manifest_df.dataset))
        out_list = []
        for dataset in datasets:
            d_participants = sorted(set(manifest_df.loc[manifest_df.dataset == dataset]["PathAI Subject ID"]))
            r_df = df.loc[df[self.participant_col_name_aim].isin(d_participants)]
            r_df = r_df.loc[~r_df[self.participant_col_name_aim].isin( self.exclude_participants)]
            d_agg_df = self.get_rr_df(r_df, endpoint, add_participant_suffix = add_participant_suffix)
            d_agg_df.insert(0, "dataset", [dataset]*len(d_agg_df))
            out_list.append(d_agg_df)
        out_df = pd.concat(out_list)
        return out_df
    def get_per_score_rr_agreement(self, df,param_key,d_col = "Time Point", add_participant_suffix = False): 
        labels_list = self.labels_dict[param_key]
        participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]] == "No"][self.participant_col_name])
        if d_col != "Time Point":
            participants_to_drop = participants_to_drop + self.exclude_participants
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H&E Slide"
        cd_dict = self.cd_dict_a
        df = df[[self.participant_col_name_aim,slide_col, d_col, cd_dict[param_key]]]
        df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
        df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
        if add_participant_suffix:
            df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna()
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim, values = cd_dict[param_key])
        pwise_tp = list(itertools.combinations(sorted(set(df[d_col])),2))
        ctab_list = []
        for pw in pwise_tp:
            #ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
            ctab_df = confusion_matrix(pvt_df[pw[0]], pvt_df[pw[1]], labels = labels_list)
            ctab_list.append(ctab_df)
        per_score_agg = self._get_per_score_agreement(ctab_list)
        n_list = []
        diag_list = [np.diagonal(c_tab) for c_tab in ctab_list]
        for i in range(ctab_list[0].shape[0]):
            n_list.append(np.min([d_list[i] for d_list in diag_list]))
        return per_score_agg, n_list
    def _get_per_score_agreement(self, ctab_list):
        out_list = []
        out_array = np.zeros((ctab_list[0].shape[0], len(ctab_list)))
        for n in range(len(ctab_list)):
            ctab_array = ctab_list[n]
            c_list = []
            for i in range(len(ctab_array)):
                r_sum = np.sum(ctab_array[i,:])
                c_sum = np.sum(ctab_array[:,i])
                r_agg = ctab_array[i,i]/r_sum
                c_agg = ctab_array[i,i]/c_sum
                mean_agg = np.mean([r_agg, c_agg])
                if np.isnan(mean_agg):
                    mean_agg = 0
                out_array[i,n] = mean_agg
        out_array = np.mean(out_array, axis = 1)
        return out_array                          
    def get_per_score_rr_df(self, df,endpoint, add_participant_suffix = False):
        if endpoint == "Repeatability":
            d_col = "Time Point"
        else:
            d_col = "Site"
        #cd_dict = self.cd_dict_a
        out_list = []
        param_list = sorted(self.cd_dict_m.keys())
        for param_key in param_list:
            labels_list = self.labels_dict[param_key]
            rr_agg_array, rr_n_list = self.get_per_score_rr_agreement(df, param_key, d_col = d_col, add_participant_suffix = add_participant_suffix)
            rr_agg_df = pd.DataFrame({"Endpoint": [endpoint] * len(rr_agg_array),
                                      "Parameter": [self.cd_dict_m[param_key]] * len(rr_agg_array), 
                                      "Score": labels_list,
                                      "Mean_Agreement_Rate":  rr_agg_array,
                                     "N": rr_n_list})
            out_list.append(rr_agg_df)
        out_df = pd.concat(out_list)
        #out_df.columns = ["Endpoint", "Parameter", "Mean_Agreement_Rate"]
        return out_df
    def get_rr_df_special(self, df, endpoint,add_participant_suffix = False):
        if endpoint == "Repeatability":
            d_col = "Time Point"
        else:
            d_col = "Site"
        special_params = self.special_params + ["F4"]
        f0_f1_agg, f0_f1_agg_n = self.get_f0_f1_rr_agreement(df, d_col = d_col, f4 = False, add_participant_suffix = add_participant_suffix)
        nas_4_agg, nas_4_agg_n = self.get_nas_rr_agreement(df, d_col = d_col, add_participant_suffix = add_participant_suffix)
        f4_agg, f4_agg_n = self.get_f0_f1_rr_agreement(df, d_col = d_col, f4 = True, add_participant_suffix = add_participant_suffix)
        out_df = pd.DataFrame({"Endpoint":[endpoint] * 3, "Parameter": special_params, "Mean_Agreement_Rate": [f0_f1_agg, nas_4_agg,f4_agg],
                              "N":[f0_f1_agg_n, nas_4_agg_n, f4_agg_n]})
        return out_df

    def get_rr_df_iteration(self,df_bootstrap,endpoint,analysis_type,itr):
        iteration_rr_df = df_bootstrap.loc[df_bootstrap.Iteration == itr]
        if analysis_type == "all":
            out_df = self.get_rr_df(iteration_rr_df, endpoint, add_participant_suffix = True)
        elif analysis_type == "per_sponsor":
            out_df =  self.get_per_sponsor_rr_table(iteration_rr_df, endpoint, add_participant_suffix = True)
        elif analysis_type == "per_timepoint":
            out_df = self.get_per_timepoint_rr_table(iteration_rr_df, endpoint, add_participant_suffix = True)
        elif analysis_type == "per_score":
            out_df = self.get_per_score_rr_df(iteration_rr_df, endpoint, add_participant_suffix = True)
        else:
            out_df = self.get_rr_df_special(iteration_rr_df, endpoint, add_participant_suffix = True)
        out_df.insert(0, "Iteration", [itr] * len(out_df))
        return out_df
    def get_rr_df_bootstrap(self, df_bootstrap, endpoint,analysis_type, n_iterations = 10):
        partial_func = partial(self.get_rr_df_iteration, df_bootstrap, endpoint, analysis_type)
        pool = mp.Pool()
        out_list = list(pool.map(partial_func, list(range(n_iterations))))
        del(pool)
        return out_list
    
    def get_ci_df_accuracy(self, bootstrap_list, s_col =None, d_col = "Difference", metric_col = "Kappa",
                           n_iterations = 10, 
                   alpha = 0.05):
        in_df = pd.concat(bootstrap_list)
        if s_col is None:
            met_df = pd.DataFrame()
            m_list = []
            for param in sorted(set(in_df.Parameter)):
                p_df = in_df.loc[in_df.Parameter == param]
                met_df.loc[param, "CI_LOW_Metric"]= p_df.dropna()[d_col].quantile(alpha/2, interpolation = "midpoint")
                met_df.loc[param, "CI_UP_Metric"]= p_df.dropna()[d_col].quantile(1-(alpha/2),interpolation = "midpoint")
                met_df.loc[param, "P_value"] = self._get_p_proportion(list(p_df.dropna()[d_col]),
                                                                      endpoint = "Accuracy")
                c_list = list(p_df.dropna()[d_col])
                met_df.loc[param, "P_values"] = len(np.where(np.array(c_list) < 0.0)[0])/len(c_list)
                method_df = pd.DataFrame()
                for method in sorted(set(in_df.Method)):
                    method_df.loc[method,"CI_LOW_Method"] = p_df.loc[p_df.Method == method][metric_col].quantile(alpha/2,interpolation = "midpoint")
                    method_df.loc[method,"CI_UP_Method"] = p_df.loc[p_df.Method == method][metric_col].quantile(1 - (alpha/2),interpolation = "midpoint")
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
                    met_df.loc[s_cat,"CI_LOW_Metric"] = s_df.dropna()[d_col].quantile(alpha/2,interpolation = "midpoint")
                    met_df.loc[s_cat,"CI_UP_Metric"] = s_df.dropna()[d_col].quantile(1 - (alpha/2),interpolation = "midpoint")
                    met_df.loc[s_cat, "P_value"] = self._get_p_proportion(list(s_df.dropna()[d_col]),
                                                                      endpoint = "Accuracy")
                    method_df = pd.DataFrame()
                    for method in sorted(set(in_df.Method)):
                        method_df.loc[method, "CI_LOW_Method"] = s_df.loc[s_df.Method == method][metric_col].quantile(alpha/2,interpolation = "midpoint")
                        method_df.loc[method, "CI_UP_Method"] = s_df.loc[s_df.Method == method][metric_col].quantile(1-(alpha/2),interpolation = "midpoint")
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
        return out_df.round(6)
    def get_ci_df_rr(self, bootstrap_list, s_col =None,
                     d_col = "Mean_Agreement_Rate", n_iterations = 10, 
                   alpha = 0.05):
        
        in_df = pd.concat(bootstrap_list)
        if s_col is None:
            met_df = pd.DataFrame()
            m_list = []
            for param in sorted(set(in_df.Parameter)):
                p_df = in_df.loc[in_df.Parameter == param]
                met_df.loc[param, "CI_LOW_Metric"]= p_df.dropna()[d_col].quantile(alpha/2,interpolation = "midpoint")
                met_df.loc[param, "CI_UP_Metric"]= p_df.dropna()[d_col].quantile(1-(alpha/2),interpolation = "midpoint")
                met_df.loc[param, "P_value"] = self._get_p_proportion(list(p_df.dropna()[d_col]),
                                                                      endpoint = "RR") 
            met_df.insert(0, "Parameter", sorted(set(in_df.Parameter)))
            out_df = met_df
        else:
            out_list = []
            met_list = []
            for param in sorted(set(in_df.Parameter)):
                met_df = pd.DataFrame()
                p_df = in_df.loc[in_df.Parameter == param]
                m_list = []
                for s_cat in sorted(set(p_df[s_col])):
                    s_df = p_df.loc[p_df[s_col] == s_cat]
                    met_df.loc[s_cat,"CI_LOW_Metric"] = s_df.dropna()[d_col].quantile(alpha/2,interpolation = "midpoint")
                    met_df.loc[s_cat,"CI_UP_Metric"] = s_df.dropna()[d_col].quantile(1 - (alpha/2),interpolation = "midpoint")
                    met_df.loc[s_cat, "P_value"] = self._get_p_proportion(list(s_df.dropna()[d_col]),
                                                                      endpoint = "RR")
                met_df.insert(0, s_col, list(met_df.index))
                met_df.insert(0, "Parameter", [param] * len(met_df))
                met_df = met_df.reset_index(drop = True)
                met_list.append(met_df)
            met_concat_df = pd.concat(met_list)
            out_df = met_concat_df
            
        return out_df.round(6) 
    def get_result_df_with_ci(self,results_df, bootstrap_list,
                              endpoint = "Accuracy",n_iterations = 10, s_col = None):
        if endpoint == "Accuracy":
            ci_df = self.get_ci_df_accuracy(bootstrap_list, s_col =s_col, d_col = "Difference",
                                            metric_col = "Kappa", n_iterations = n_iterations, 
                   alpha = 0.05)
            if s_col is None:
                out_df = results_df.merge(ci_df, on = ["Parameter", "Method"],how = "inner")
            else:
                out_df = results_df.merge(ci_df, on = ["Parameter", "Method", s_col], how ="inner")
        else:
            ci_df = self.get_ci_df_rr(bootstrap_list, s_col =s_col,
                     d_col = "Mean_Agreement_Rate", n_iterations = n_iterations, 
                   alpha = 0.05)
            if s_col is None:
                out_df = results_df.merge(ci_df, on = ["Parameter"],how = "inner")
            else:
                out_df = results_df.merge(ci_df, on = ["Parameter", s_col],how = "inner")
        return out_df.round(6)
    
class AVanalysisExp():
    def __init__(self,cd_dict_m,cd_dict_a, participant_col_name,participant_col_name_aim,
                 gs_df, manifest_df,labels_dict,  files_path):
        self.cd_dict_m = cd_dict_m
        self.cd_dict_a = cd_dict_a
        self.participant_col_name= participant_col_name
        self.participant_col_name_aim = participant_col_name_aim
        self.gs_df = gs_df
        self.manifest_df = manifest_df
        self.labels_dict = labels_dict
        self.files_path = files_path
    def get_rr_mean_agreement(self, df,param_key,d_col = "Time Point", add_participant_suffix = False,
                             rr = True):
        if param_key == "f":
            slide_col = "Trichrome Slide"
        else:
            slide_col = "H&E Slide"
        labels_list = self.labels_dict[param_key]
        cd_dict = self.cd_dict_a
        df = df[[self.participant_col_name_aim,slide_col, d_col, cd_dict[param_key]]]
        if rr:
            participants_to_drop = list(self.gs_df[self.gs_df["NASH_biopsy_adequacy_" + self.cd_dict_m[param_key]] == "No"][self.participant_col_name])
            participants_to_drop = participants_to_drop + self.exclude_participants
            df = df.loc[df[self.participant_col_name_aim].isin(self.manifest_df["PathAI Subject ID"])]
            df = df.loc[~df[self.participant_col_name_aim].isin(participants_to_drop)]
            if add_participant_suffix:
                df = self._add_participant_suffix(df, expected_length = int(len(df)/3))
        df = df.dropna()
        df = df.drop_duplicates()
        pvt_df =  pd.pivot(df, columns = d_col, index = self.participant_col_name_aim,
                           values = cd_dict[param_key])
        pwise_tp = list(itertools.combinations(sorted(set(df[d_col])),2))
        agg_list = []
        ctab_list = []
        con_mat_list = []
        for pw in pwise_tp:
            #ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
            agg_rate = len(pvt_df.loc[pvt_df[pw[0]] == pvt_df[pw[1]]])/len(pvt_df)
            # agg_rate=  self._get_agg_rate(pvt_df[pw[0]], pvt_df[pw[1]], labels_list = labels_list)
            agg_list.append(agg_rate)
            if rr:
                con_mat = confusion_matrix(pvt_df[pw[0]], pvt_df[pw[1]], normalize = "true", labels = labels_list)
                ctab_df = pd.crosstab(pvt_df[pw[0]], pvt_df[pw[1]])
                ctab_list.append(ctab_df)
        if rr:
            return agg_list, ctab_list, pvt_df
        else:
            return np.mean(agg_list), len(pvt_df)
    def compile_rr_mean_agg_df(self,df):
        agg_df = pd.DataFrame()
        for param_key in list(self.cd_dict_a.keys()):
            agg_list = self.get_rr_mean_agreement(df, param_key, rr = False)
            agg_df.loc[self.cd_dict_a[param_key], "Mean_Agreement"] = agg_list[0]
            agg_df.loc[self.cd_dict_a[param_key], "N"] = agg_list[1]
        agg_df.insert(0, "Parameter", list(agg_df.index))
        agg_df = agg_df.reset_index(drop = True)
        return agg_df
    def _get_agg_rate(arr1,arr2, labels_list = []):
        if len(labels_list) == 0:
            ctab = confusion_matrix(arr1, arr2)
        else:
            ctab = confusion_matrix(arr1, arr2, labels = labels_list)
        agg_rate = np.sum(np.diag(ctab))/np.sum(ctab)
        return agg_rate
    def _get_kappa(self, arr1,arr2, labels, weights = "linear"):
        kappa = cohen_kappa_score(arr1, arr2,labels, weights= weights)
        return kappa
    def get_accuracy_df(self, df, param_key, method = "AIM", drop_duplicates = True, per_score = False):
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
    def compute_avg_agreement(self, a_df, m_df, param_key, plot_confusion_table = False):
        a_df = self.get_accuracy_df(a_df, param_key, method = "AIM")
        m_df = self.get_accuracy_df(m_df, param_key, method = "Manual")
        t_df  = pd.pivot(m_df, index = ["slide ID"], columns = ["user_name"], values =self.cd_dict_m[param_key] + "_x")
        t_df_ceil = pd.DataFrame(t_df.apply(lambda x: np.ceil(np.median(x.dropna())), axis = 1), columns = [self.cd_dict_m[param_key] + "_x"])
        t_df_floor =  pd.DataFrame(t_df.apply(lambda x: np.floor(np.median(x.dropna())), axis = 1), columns =  [self.cd_dict_m[param_key] + "_x"])
        a_df = a_df.merge(m_df[["slide ID", "Identifier"]].drop_duplicates(),on = ["Identifier"], how = "inner")
        a_df = a_df.set_index("slide ID")
        df_ceil = a_df.merge(t_df_ceil, right_index = True, left_index = True, how = "inner")
        df_floor = a_df.merge(t_df_floor, right_index = True, left_index = True, how = "inner")
        labels_list = self.labels_dict[param_key]
        agg_rate_ceil = self._get_kappa(df_ceil[self.cd_dict_a[param_key]], df_ceil[self.cd_dict_m[param_key] + "_x"], labels = labels_list)
        agg_rate_floor = self._get_kappa(df_floor[self.cd_dict_a[param_key]], df_floor[self.cd_dict_m[param_key] + "_x"], labels = labels_list)
        return np.round(np.mean([agg_rate_ceil, agg_rate_floor]),3), len(df_floor)
    def get_avg_agreement_rate_df(self,accuracy_df,manual_df, endpoint = "AIM vs. Median Manual Reads Agreement" ):
        out_list = []
        param_list = sorted(self.cd_dict_m.keys())
        for param in param_list:
            avg_agg = self.compute_avg_agreement(accuracy_df, manual_df, param)
            out_list.append([endpoint, self.cd_dict_m[param], avg_agg[0], avg_agg[1]])
        out_df = pd.DataFrame(out_list)
        out_df.columns = ["Endpoint", "Parameter", "Kappa", "N"]
        return out_df
    def process_inter_rater_df(self, manual_df, param_key):
        m_df = manual_df[[self.participant_col_name,"user_name", "slide ID",self.cd_dict_m[param_key]]].drop_duplicates()
        m_df =m_df.dropna(subset = [self.cd_dict_m[param_key]])
        manual_df_list = []
        for idt in sorted(set(m_df[self.participant_col_name])):
            id_df = m_df.loc[m_df[self.participant_col_name] == idt]
            id_df = id_df.sort_values(by = "user_name")
            id_df["Read Number"] = [i + 1 for i in range(len(id_df))]
            manual_df_list.append(id_df)
        manual_df_processed = pd.concat(manual_df_list)
        return manual_df_processed
    def process_nas_df(self, manual_df):
        nas_params = ["b","s", "i"]
        nas_df_list = []
        for param in list(self.cd_dict_m.keys()):
            param_df = self.process_inter_rater_df(manual_df, param)
            if param in nas_params:
                nas_df_list.append(param_df)
        nas_df = reduce(lambda left, right: pd.merge(left, right, on = [self.participant_col_name,"slide ID",
                                                                        "Read Number","user_name"],
                                                     how = "inner"), nas_df_list)
        nas_cols = [self.cd_dict_m[param] for param in nas_params]
        nas_df["NAS_score"] = nas_df[nas_cols].sum(axis = 1)
        return nas_df
    def get_inter_rater_kappa(self, manual_df, param_key, NAS = False, metric = "kappa"):
        if NAS:
            manual_df_processed = self.process_nas_df(manual_df)
            param_name = "NAS_score"
        else:
            manual_df_processed = self.process_inter_rater_df(manual_df, param_key)
            param_name = self.cd_dict_m[param_key]
        pvt_df = pd.pivot(manual_df_processed, index = ["Identifier"],
                          columns = ["Read Number"], values = [param_name])
        pvt_df.columns = [col[1] for col in list(pvt_df)]
        pvt_df.columns = ["R" + str(i+1) for i in range(len(list(pvt_df)))]
        comb_list = list(itertools.combinations(list(pvt_df),2))
        comb_df = pd.DataFrame()
        if NAS:
            label_list = list(range(9))
        else:
            label_list = list(self.labels_dict[param_key])
        c_list = []
        for comb in comb_list:
            r1 = comb[0]
            r2 = comb[1]
            # str_1 = "Rater " + str(r1)
            # str_2 = "Rater " + str(r2)
            str_index = r1 + " vs. " + r2
            c_df= pvt_df[[r1,r2]].dropna()
            total = len(c_df)
            comb_df.loc[str_index, "Total"] = total
            if metric == "kappa":
                linear_kappa = self._get_kappa(c_df[r1], c_df[r2], labels = label_list)
                comb_df.loc[str_index, "Linear_Kappa"] = linear_kappa
            else:
                agg_counts = len(c_df.loc[c_df[r1] == c_df[r2]])
                agg_rate = (agg_counts/total)
                comb_df.loc[str_index, "Counts_Agree"] = agg_counts
                comb_df.loc[str_index, "Total"] = total
                comb_df.loc[str_index, "Agg_rate"] = agg_rate
        if metric == "kappa":
            comb_df.loc["Mean Kappa", "Linear_Kappa"] = np.nanmean(list(comb_df.Linear_Kappa))
            r7_idx = [c for c in list(comb_df.index) if c.find("R7") == -1 ]
            r7_idx = [c for c in r7_idx if c.find("Mean") == -1 ]
            comb_df.loc["Mean Kappa w/o Rater 7", "Linear_Kappa"] = np.nanmean(comb_df.loc[r7_idx]["Linear_Kappa"])
        comb_df.insert(0, "Parameter", [param_name] * len(comb_df))
        comb_df.insert(0, "Comparison", list(comb_df.index))
        comb_df = comb_df.reset_index(drop = True)
        return comb_df
    def compile_inter_rater_kappa(self, manual_df, metric = "kappa"):
        nas_kappa_df = self.get_inter_rater_kappa(manual_df, None, NAS = True, metric = metric)
        df_list = []
        for param in list(self.cd_dict_m.keys()):
            param_kappa_df =  self.get_inter_rater_kappa(manual_df, param, metric = metric)
            df_list.append(param_kappa_df)
        out_df = pd.concat(df_list)
        out_df = pd.concat([out_df, nas_kappa_df])
        return out_df
    def get_missing_slides_freq_rr(self, processed_df_rr, manifest_df, rr_missing_df):
        cd_dict_m = self.cd_dict_m
        out_df_list = []
        for param_key in list(cd_dict_m.keys()):
            if param_key == "f":
                stain = "Trichrome"
            else:
                stain = "H & E"
            rr_df_m = processed_df_rr.merge(manifest_df[["slide ID", "stain"]],on = ["slide ID"],
                                            how = "inner")
            rr_df_m= rr_df_m[["slide ID", "Identifier","stain","user_name", cd_dict_m[param_key]]].drop_duplicates()
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
    def get_missing_slides_freq_gt(self,processed_df_gt,gs_initial_df_final, gt_missing_df):
        cd_dict_m = self.cd_dict_m
        out_df_list = []
        for param_key in list(cd_dict_m.keys()):
            if param_key == "f":
                stain = "Trichrome"
                slide_col = "Trichrome Slide"
            else:
                stain = "H & E"
                slide_col= "H & E Slide"
            missing_ids = sorted(set(processed_df_gt.loc[processed_df_gt[cd_dict_m[param_key]].isnull()]["Identifier"]))
            non_missing_ids = sorted(set(processed_df_gt.loc[~processed_df_gt[cd_dict_m[param_key]].isnull()]["Identifier"]))
            gt_missing_id_df = gt_missing_df.loc[gt_missing_df.identifier.isin(missing_ids)]
            gt_missing_id_df = gt_missing_id_df.loc[gt_missing_id_df.stain == stain]
            gt_missing_id_df = gt_missing_id_df[["identifier", "slide_ID", "Category"]].dropna().drop_duplicates()
            gt_sample_missing_df = processed_df_gt.loc[processed_df_gt.Identifier.isin(missing_ids)]
            # # gt_sample_missing_df = processed_df_gt.loc[np.logical_and(processed_df_gt.Identifier.isin(missing_ids),
            # #                                                           processed_df_gt["consensus_type_" + cd_dict_m[param_key]] != "panel_consensus")]
            gt_sample_missing_df = gt_sample_missing_df.loc[gt_sample_missing_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "No"]
            # # gt_sample_missing_df = gt_sample_missing_df.loc[~gt_sample_missing_df.Identifier.isin(gt_missing_id_df.identifier)]
            a_df3 = pd.DataFrame({"identifier":list(gt_sample_missing_df.Identifier), "slide_ID":list(gt_sample_missing_df[slide_col]),
                                     "Category":["Sample"]*len(gt_sample_missing_df)})

            init_df_missing = gs_initial_df_final.loc[gs_initial_df_final["slide ID"].isin(a_df3["slide_ID"])][["user_name", "slide ID", "NASH biopsy adequacy"]]
            drop_slides = []
            for slide_id in sorted(set(a_df3.slide_ID)):
                s_df = init_df_missing.loc[init_df_missing["slide ID"] == slide_id]
                consensus_type = processed_df_gt.loc[processed_df_gt[slide_col] == slide_id]["consensus_type_" + cd_dict_m[param_key]].values[0]
                if 0 < len(s_df.loc[s_df["NASH biopsy adequacy"] == "No"]) < 2 and consensus_type == "panel_consensus":
                    drop_slides.append(slide_id)
            a_df3 = a_df3.loc[~a_df3.slide_ID.isin(drop_slides)]
            diff_ids = set(missing_ids) - set(gt_missing_id_df.identifier)
            if len(diff_ids) > 0:
                # diff_df = processed_df_gt.loc[processed_df_gt[cd_dict_m[param_key]].isnull()]
                diff_df = processed_df_gt.loc[processed_df_gt.Identifier.isin(diff_ids)]
                diff_df1 = diff_df.loc[diff_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "No"]
                diff_df2 = diff_df.loc[diff_df["NASH_biopsy_adequacy_" + cd_dict_m[param_key]] == "Yes"]
                a_df1 = pd.DataFrame({"identifier":list(diff_df1.Identifier), "slide_ID":list(diff_df1[slide_col]),
                                     "Category":["Sample"]*len(diff_df1)})
                a_df2 = pd.DataFrame({"identifier":list(diff_df2.Identifier), "slide_ID":list(diff_df2[slide_col]),
                                     "Category":["Other/reason not provided"]*len(diff_df2)})
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
            comma_rows = m_df.loc[m_df.Category.str.contains(",")]
            comma_missing_df = processed_df_gt.loc[processed_df_gt[slide_col].isin(comma_rows["slide_ID"])]
            counts_df = pd.DataFrame(m_df.Category.value_counts())
            counts_df = counts_df.rename(columns = {"Category":"Counts"})
            missing_total = np.sum(counts_df["Counts"])
            assert missing_total + len(non_missing_ids) == len(set(processed_df_gt.Identifier)), "Error adding up missing and non-missing ids"
            counts_df.loc["Non-missing", "Counts"] = len(non_missing_ids)
            counts_df["Total"] = int(len(set(processed_df_gt.Identifier)))
            counts_df.insert(0, "Reason", list(counts_df.index))
            counts_df.insert(0, "Parameter", [cd_dict_m[param_key]]*len(counts_df))
            counts_df["Percent"] = np.round((counts_df["Counts"]/counts_df["Total"]) * 100, 6)
            out_df_list.append(counts_df)
        return pd.concat(out_df_list).reset_index(drop = True)
    def get_wilson_ci(self,in_df, metric_col = "Mean_Agreement_Rate",  alpha = 0.05):
        for idx in list(in_df.index):
            if in_df.loc[idx,metric_col] == 1:
                wilson_ci = get_confintervals(in_df.loc[idx,"N"], in_df.loc[idx,"N"], alpha = alpha)
                in_df.loc[idx, "Wilson_CI_Lower"] = np.round(wilson_ci[0],6)
                in_df.loc[idx, "Wilson_CI_Upper"] = np.round(wilson_ci[1],6)
            else:
                in_df.loc[idx, "Wilson_CI_Lower"] = in_df.loc[idx, "Wilson_CI_Upper"] = np.nan
        return in_df


