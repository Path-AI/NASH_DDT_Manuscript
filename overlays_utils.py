"""
Utility functions for NASH DDT Overlays analysis
"""
import sys
import warnings
sys.path.append("../../")
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from functools import partial
from nash_ddt_utils import *
from r_models import get_confintervals


def recode_cols(in_df: pd.DataFrame, param_key: str, val_col: str, feature_dict_enroll: Dict[str,str]) -> pd.DataFrame:
    """
    Function for recoding enrollment columns.
    Args:
        in_df: Input dataframe containing evaluation frames
        param_key: String from ["f", "s", "b", "i","a"] indicating the evaluation parameter to be used.
        val_col: String for the column name containing values to be recoded.
        feature_dict_enroll: Dictionary containing the column names for each one of the evaluation parameters.
    Returns:
        in_df: Dataframe containing recoded columns for a given evaluation parameter.
    """
    in_df = in_df.loc[in_df["evaluation"] == feature_dict_enroll[param_key]]
    in_df[val_col] = in_df[val_col].astype("float")
    recode_vals = []
    for i in list(in_df.index):
        in_val = in_df[val_col][i]
        if param_key == "s":
            if in_val == 0:
                recode_vals.append("None")
            elif 20> in_val > 0:
                recode_vals.append("Low")
            elif 45 >= in_val >= 20:
                recode_vals.append("Medium")
            else:
                recode_vals.append("High")
        elif param_key == "f":
            if in_val == 0:
                recode_vals.append("None")
            elif 25 > in_val > 0:
                recode_vals.append("Low")
            elif 45 >= in_val >= 25:
                recode_vals.append("Medium")
            else:
                recode_vals.append("High")
        else:
            if in_val > 0:
                recode_vals.append("Present")
            else:
                recode_vals.append("None")
    in_df[val_col] = recode_vals
    return in_df

def get_frames_distribution(in_df:pd.DataFrame,param_key:str,feature_dict:Dict[str,str],stain:str,score_dict:Dict[str,str], names_dict:Dict[str,str]) -> Tuple:
    """
    Function for generating frames distribution overall, per partner and per score.
    Args:
        in_df: Input dataframe containing evaluation/enrollment frames
        param_key: String from ["f", "s", "b", "i","a"] indicating the evaluation parameter to be used.
        feature_dict: Dictionary containing the column names for each one of the evaluation parameters.
        stain: String indicating the stain column in the frames table. Could be "H & E" or "Trichrome"
        score_dict: Dictionary containing per parameter score column name.
        names_dict: Dictionary containing display names for each evaluation parameter
    Returns:
        Tuple containing distribution frequencies overall, per partner and per score.
    """
    enroll_features = feature_dict[param_key]
    feat_name = names_dict[param_key]
    if param_key != "a":
        score_col = score_dict[param_key]
    else:
        score_col = None
    param_df = in_df.loc[in_df["evaluation"] == enroll_features]
    param_df = param_df.loc[param_df["Stain"] == stain]
    if score_col is not None:
        param_df = param_df[["frame_id", "slide_id","Stain","evaluation","partner", "value", score_col]].drop_duplicates()
    else:
        param_df = param_df[["frame_id", "slide_id","Stain","evaluation","partner", "value"]].drop_duplicates()
    if param_key in ["s", "f", "a"]:
        param_df = recode_cols(param_df,param_key, "value",feature_dict)
    # Value counts
    grp_count_df = param_df[["frame_id", "value"]].groupby("value").count()
    grp_count_df = grp_count_df.rename(columns = {"frame_id":"Count"})
    grp_count_df["fraction"] = grp_count_df["Count"]/len(param_df)                          
    grp_count_df.insert(0, "Category", list(grp_count_df.index))
    grp_count_df.insert(0, "Stain", [stain]*len(grp_count_df))
    grp_count_df.insert(0, "Feature", [feat_name]*len(grp_count_df))
    # Partner counts for frames
    partner_count_df =  param_df[["frame_id", "partner"]].groupby("partner").count()
    partner_count_df = partner_count_df.rename(columns = {"frame_id":"Count"})
    partner_count_df["fraction"] = partner_count_df["Count"]/len(param_df)
    partner_count_df.insert(0, "Stain", [stain]*len(partner_count_df))
    partner_count_df.insert(0, "Feature", [feat_name]*len(partner_count_df))
    # Partner counts for slides
    slide_p_df = param_df[["slide_id", "partner"]].drop_duplicates()
    partner_count_df_slide =  slide_p_df.groupby("partner").count()
    partner_count_df_slide = partner_count_df_slide.rename(columns = {"slide_id":"Count"})
    partner_count_df_slide["fraction"] = partner_count_df_slide["Count"]/len(slide_p_df)
    partner_count_df_slide.insert(0, "Stain", [stain]*len(partner_count_df_slide))
    partner_count_df_slide.insert(0, "Feature", [feat_name]*len(partner_count_df_slide))
    
    # Slide counts
    if score_col is not None:
        param_df = param_df[["slide_id",score_col]].drop_duplicates()
        slide_group_df = param_df.groupby([score_col]).count()
        slide_group_df = slide_group_df.rename(columns = {"slide_id":"Count"})
        slide_group_df["fraction"] = slide_group_df["Count"]/len(param_df)
        slide_group_df.insert(0, "Score",list(slide_group_df.index))
        slide_group_df.insert(0, "Stain",[stain] * len(slide_group_df))
        slide_group_df.insert(0, "Feature", [feat_name]*len(slide_group_df))
        return grp_count_df.reset_index(), slide_group_df.reset_index(), partner_count_df.reset_index(), partner_count_df_slide
    else:
        return grp_count_df.reset_index(), partner_count_df.reset_index(), partner_count_df_slide
    
    
def get_per_patho_counts(in_df:pd.DataFrame, feature_dict:Dict[str,str], param_key: str,stain: str) -> pd.DataFrame:
    """
    Function for generating per pathologist feature counts frequency.
    Args:
        in_df: Input dataframe containing evaluation/enrollment frames
        feature_dict: Dictionary containing the column names for each one of the evaluation parameters.
        param_key: String from ["f", "s", "b", "i","a"] indicating the evaluation parameter to be used.
        stain: String indicating the stain column in the frames table. Could be "H & E" or "Trichrome"
    Returns:
        out_df: Dataframe containing per pathologist frames counts that were enrolled.
    """
    u_df = in_df.loc[in_df.feature == feature_dict[param_key]]
    u_df = u_df.loc[u_df.stain == stain]
    out_list = []
    patho_list = sorted(set(u_df.username))
    for patho in patho_list:
        p_df = u_df.loc[u_df.username == patho]
        p_df = p_df[["frame_id", "Visibility", "stain"]].drop_duplicates()
        grp_df = p_df[["frame_id", "Visibility"]].drop_duplicates().groupby("Visibility").count()
        grp_df = grp_df.rename(columns = {"frame_id":"Count"})
        grp_df["fraction"] = grp_df["Count"]/len(p_df)
        grp_df["N Total"] = [len(p_df)] * len(grp_df)
        grp_df.insert(0, "Pathologist", [patho] * len(grp_df))
        grp_df.insert(0, "Stain", [stain]*len(grp_df))
        grp_df.insert(0, "Feature", [feature_dict[param_key]]*len(grp_df))
        out_list.append(grp_df)
    out_df = pd.concat(out_list)
    return out_df


def get_tp_success_rate(in_df: pd.DataFrame, param_key: str, stain: str,feature_dict: Dict[str,str] , underestimate_dict: Dict[str,str]) -> pd.DataFrame:
    """
    Function for computing underestimation (true positive) success rates overall and per pathologist.
    Args:
        in_df: Input dataframe containing enrollment frames
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters.
        param_key: String from ["f", "s", "b", "i","a"] indicating the enrollment parameter to be used.
        stain: String indicating the stain column in the frames table. Could be "H & E" or "Trichrome"
        underestimate_dict: Dictionary containing per enrollment parameter values for measuring true positive rate.
    Returns:
        out_df: Dataframe containing overall and per pathologist enrollment true positive success rates per parameter
    """
    t_df = in_df.loc[in_df.feature == feature_dict[param_key]]
    u_df = t_df.loc[t_df["Visibility"] != "No"]
    patho_dict = {}
    u_df = u_df.loc[u_df.stain == stain]
    patho_list = sorted(set(u_df.username))
    tp_rate = []
    for patho in patho_list:
        p_df = u_df.loc[u_df.username == patho]
        p_df = p_df.dropna(subset = ["Underestimation"])
        n_success = len(p_df.loc[p_df["Underestimation"] == underestimate_dict[param_key]])
        n_total = len(p_df)
        tp_success = n_success/n_total
        patho_dict[patho] = [tp_success,int(n_success), int(n_total)]
        tp_rate.append(tp_success)
    out_df = pd.DataFrame(patho_dict).T
    out_df.columns = ["Success Rate","N Success", "N Total"]
    out_df.loc["All"] = [np.mean(tp_rate), np.nan, np.nan]
    out_df.insert(0,"Pathologist", list(out_df.index))
    out_df.insert(0, "Parameter", [feature_dict[param_key]] * len(out_df))
    out_df.insert(0, "Stain", [stain]*len(out_df))
    out_df = out_df.reset_index(drop = True)
    return out_df

def get_fp_success_rate(in_df: pd.DataFrame, param_key: str, stain:str,feature_dict: Dict[str,str] ) -> pd.DataFrame:
    """
    Function for computing overestimation (false positive) success rates overall and per pathologist.
    Args:
        in_df: Input dataframe containing enrollment frames
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters.
        param_key: String from ["f", "s", "b", "i","a"] indicating the enrollment parameter to be used.
        stain: String indicating the stain column in the frames table. Could be "H & E" or "Trichrome"
    Returns:
        out_df: Dataframe containing overall and per pathologist enrollment false positive success rates per parameter
    """
    t_df = in_df.loc[in_df.feature == feature_dict[param_key]]
    u_df = t_df.copy()
    u_df = u_df.loc[u_df.stain == stain]
    patho_dict = {}
    fp_rate = []
    patho_list = sorted(set(u_df.username))
    for patho in patho_list:
        p_df = u_df.loc[u_df.username == patho]
        p_df = p_df.dropna(subset = ["Overestimation"])
        n_success = len(p_df.loc[p_df["Overestimation"] != 'Greater than or equal to 20%'])
        n_total = len(p_df)
        tp_success = n_success/n_total
        patho_dict[patho] = [tp_success,int(n_success), int(n_total)]
        fp_rate.append(tp_success)
    out_df = pd.DataFrame(patho_dict).T
    out_df.columns = ["Success Rate","N Success", "N Total"]
    out_df.loc["All"] = [np.mean(fp_rate), np.nan, np.nan]
    out_df.insert(0,"Pathologist", list(out_df.index))
    out_df.insert(0, "Parameter", [feature_dict[param_key]] * len(out_df))
    out_df.insert(0, "Stain", [stain]*len(out_df))
    out_df = out_df.reset_index(drop = True)
    return out_df


def compile_results_tp(in_df: pd.DataFrame, feature_dict: Dict[str, str], underestimate_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Function for compiling underestimation success rate results across all the enrollment parameters.
    Args:
        in_df: Input dataframe containing enrollment frames
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters.
        underestimate_dict: Dictionary containing per enrollment parameter values for measuring true positive rate.
    Returns:
        out_df: Dataframe containing overall and per pathologist enrollment true positive success rates
    """
    tp_success_rate_dict = {}
    for param_key in list(feature_dict.keys()):
        if param_key == "a":
            he_name = feature_dict[param_key] + "_HE"
            tc_name = feature_dict[param_key] + "_Trichrome"
            tp_success_rate_dict[he_name] = get_tp_success_rate(in_df, param_key, 'H & E',feature_dict,underestimate_dict)
            tp_success_rate_dict[tc_name] = get_tp_success_rate(in_df, param_key,'Trichrome',feature_dict,underestimate_dict)
        elif param_key == "f":
            tp_success_rate_dict[feature_dict[param_key]] = get_tp_success_rate(in_df, param_key, 'Trichrome',feature_dict,underestimate_dict)
        else:
            tp_success_rate_dict[feature_dict[param_key]] = get_tp_success_rate(in_df, param_key, 'H & E',feature_dict,underestimate_dict)
    out_df = pd.concat(list(tp_success_rate_dict.values()))
    out_df["Parameter_actual"] = out_df["Parameter"] + "_" +out_df["Stain"]
    return out_df

def compile_results_fp(in_df: pd.DataFrame, feature_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Function for compiling overestimation success rate results across all the enrollment parameters.
    Args:
        in_df: Input dataframe containing enrollment frames
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters.
    Returns:
        out_df: Dataframe containing overall and per pathologist enrollment false positive success rates
    """
    fp_success_rate_dict = {}
    for param_key in list(feature_dict.keys()):
        if param_key == "a":
            he_name = feature_dict[param_key] + "_HE"
            tc_name = feature_dict[param_key] + "_Trichrome"
            fp_success_rate_dict[he_name] = get_fp_success_rate(in_df, param_key, 'H & E',feature_dict)
            fp_success_rate_dict[tc_name] = get_fp_success_rate(in_df, param_key, 'Trichrome',feature_dict)
        elif param_key == "f":
            fp_success_rate_dict[feature_dict[param_key]] = get_fp_success_rate(in_df, param_key, 'Trichrome',feature_dict)
        else:
            fp_success_rate_dict[feature_dict[param_key]] = get_fp_success_rate(in_df, param_key, 'H & E',feature_dict)
    out_df = pd.concat(list(fp_success_rate_dict.values()))
    out_df["Parameter_actual"] = out_df["Parameter"] + "_" +out_df["Stain"]
    return out_df

def get_success_rate_itr(bootstrap_df: pd.DataFrame,feature_dict: Dict[str,str],underestimate_dict: Dict[str,str],itr:int) -> Tuple:
    """
    Function for computing overestimation and underestimation success rates per bootstrap iteration
    Args:
        bootstrap_df: Dataframe containing bootstrap samples for the enrollment table
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters
        underestimate_dict: Dictionary containing per enrollment parameter values for measuring true positive rate
    Returns:
        Tuple containing per iteration true positive rate and false positive rate results tables.
    """
    itr_df = bootstrap_df.loc[bootstrap_df.Iteration == itr]
    itr_df["Overestimation"] = itr_df.apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x["Overestimation"]), axis = 1)
    itr_df["Overestimation"] = itr_df["Overestimation"].str.rstrip(" ")
    itr_df["Overestimation"] = itr_df["Overestimation"].str.lstrip(" ")
    fp_df = compile_results_fp(itr_df, feature_dict)
    tp_df = compile_results_tp(itr_df, feature_dict,underestimate_dict)
    fp_df.insert(0,"Iteration", [itr]*len(fp_df))
    tp_df.insert(0, "Iteration", [itr]*len(tp_df))
    return tp_df,fp_df

def compile_results_bootstrap(bootstrap_df:pd.DataFrame, feature_dict: Dict[str,str], underestimate_dict:Dict[str,str],
                              n_iterations: int = 2000, alpha: float = 0.05) -> Tuple:
    """
    Function for compiling results across all the bootstrap iterations
    Args:
        bootstrap_df: Dataframe containing bootstrap samples for the enrollment table
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters
        underestimate_dict: Dictionary containing per enrollment parameter values for measuring true positive rate
        n_iterations: Number of iterations to be used from the bootstrap data for computing confidence intervals (default = 2000)
        alpha: significance threshold for computing confidence intervals (default = 0.05)
    Returns:
        Tuple containing true positive rate and false positive rate confidence intervals computed across n_iterations using alpha significance threshold
    """
    partial_func_itr = partial(get_success_rate_itr, bootstrap_df, feature_dict,underestimate_dict)
    success_rate_list = list(map(partial_func_itr, range(n_iterations)))
    itr_tp_list = [out[0] for out in success_rate_list]
    itr_fp_list = [out[1] for out in success_rate_list]
    tp_df = pd.concat(itr_tp_list)
    fp_df = pd.concat(itr_fp_list)
    out_list_tp = []
    out_list_fp = []
    for param in sorted(set(tp_df["Parameter_actual"])):
        ci_dict_tp = {}
        ci_dict_fp = {}
        for patho in sorted(set(tp_df.Pathologist)):
            patho_df_tp = tp_df.loc[np.logical_and(tp_df.Pathologist == patho,
                                                   tp_df.Parameter_actual == param)]
            patho_df_tp = patho_df_tp.dropna(subset = ["Success Rate"])
            ci_dict_tp[patho] = patho_df_tp["Success Rate"].quantile([alpha/2, 1-(alpha/2)])
        for patho in sorted(set(fp_df.Pathologist)):
            patho_df_fp = fp_df.loc[np.logical_and(fp_df.Pathologist == patho,
                                                   fp_df.Parameter_actual == param)]
            patho_df_fp = patho_df_fp.dropna(subset = ["Success Rate"])
            ci_dict_fp[patho] = patho_df_fp["Success Rate"].quantile([alpha/2, 1-(alpha/2)])
        param_df_tp = pd.DataFrame(ci_dict_tp).T
        param_df_fp = pd.DataFrame(ci_dict_fp).T
        param_df_tp.insert(0,"Pathologist", list(param_df_tp.index))
        param_df_tp.insert(0, "Parameter_actual", [param] * len(param_df_tp))
        param_df_tp = param_df_tp.reset_index(drop = True)
        out_list_tp.append(param_df_tp)
        param_df_fp.insert(0,"Pathologist", list(param_df_fp.index))
        param_df_fp.insert(0, "Parameter_actual", [param] * len(param_df_fp))
        param_df_fp = param_df_fp.reset_index(drop = True)
        out_list_fp.append(param_df_fp)
    out_df_tp = pd.concat(out_list_tp).dropna()
    out_df_fp = pd.concat(out_list_fp).dropna()
    return out_df_tp, out_df_fp


def get_results_df(in_df: pd.DataFrame,bootstrap_df: pd.DataFrame, feature_dict: Dict[str,str],
                   underestimate_dict: Dict[str,str], n_iterations: int = 2000, alpha: float = 0.05) -> Tuple:
    """
    Function for computing all the results tables and for generating output files for overestimation and underestimation success rates
    Args:
        in_df: Input dataframe containing enrollment frames
        bootstrap_df: Dataframe containing bootstrap samples for the enrollment table
        feature_dict: Dictionary containing the column names for each one of the enrollment parameters
        underestimate_dict: Dictionary containing per enrollment parameter values for measuring true positive rate
        n_iterations: Number of iterations to be used from the bootstrap data for computing confidence intervals (default = 2000)
        alpha: significance threshold for computing confidence intervals (default = 0.05)
    Returns:
        Tuple containing final results for true positive and false positive success rates
    """
    res_df_tp = compile_results_tp(in_df, feature_dict,underestimate_dict)
    res_df_fp = compile_results_fp(in_df, feature_dict)
    df_tp, df_fp = compile_results_bootstrap(bootstrap_df, feature_dict,underestimate_dict, n_iterations = n_iterations, alpha = alpha)
    res_df_fp  = res_df_fp.merge(df_fp, on = ["Parameter_actual", "Pathologist"], how = "inner")
    res_df_tp = res_df_tp.merge(df_tp, on = ["Parameter_actual", "Pathologist"], how = "inner")
    return res_df_tp, res_df_fp

def postprocess_results_wilson_ci(in_df: pd.DataFrame, alpha: float =0.05) -> pd.DataFrame:
    """
    Generate Wilson score CIs for rows with 100% success in a post-hoc manner.
    Args:
        in_df: Dataframe containing true positive and false positive success rates.
        alpha: significance threshold for computing confidence intervals (default = 0.05)
    Returns:
        concat_df: Dataframe with Wilson score CIs for the rows with 100% success.
    """
    p_df = in_df.loc[in_df["Success Rate"] < 1]
    s_df = in_df.loc[in_df["Success Rate"] ==1]
    s_df.loc[s_df.Pathologist == "All", ["N Success", "N Total"]] = [160,160]
    wi_cis = s_df.apply(lambda x: get_confintervals(x["N Success"], x["N Total"], alpha= alpha), axis = 1)
    s_df[0.025] = [wc[0] for wc in wi_cis]
    s_df[0.975] = [wc[1] for wc in wi_cis]
    concat_df = pd.concat([p_df, s_df])
    return concat_df