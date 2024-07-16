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
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from functools import reduce
from typing import Dict, List, Optional
from CV_utils import *
random.seed(1234)
cd_dict_a1 = {
    "s":'H&E Slide - Steatosis Score (Algorithm Score)',
    "i":'H&E Slide - Lobular Inflammation Score (Algorithm Score)',
    "b":'H&E Slide - Hepatocellular Ballooning Score (Algorithm Score)',
    "f":'Trichrome Slide - CRN Fibrosis Score (Algorithm Score)'
}
cd_dict_a = {
    "s":'steatosis_final_score',
    "i":'inflam_final_score',
    "b":'balloon_final_score',
    "f":'crn_final_score'
}
cd_dict_m = {"s":"NASH steatosis",
             "i":"Lobular inflammation",
             "b":"Hepatocellular ballooning",
             "f":"NASH CRN Score"
}
util_dict= {"s":"utility_steatosis", "f":"utility_fibrosis",
            "i":"utility_inflam", "b": "utility_balloon"}
cd_dict_p = {
    "s":'steatosis',
    "i":'inflam',
    "b":'balloon',
    "f":'crn'
}
ml_suffix_aim = "_ml"
patho_suffix_aim =  "_patho"
agree_suffix_aim = "_agree"
labels_dict = {"s": [0,1,2,3], "b": [0,1,2], "i":[0,1,2,3], "f":[0,1,2,3,4]}
special_params = ["F0_F1", "NAS_4"]
names_dict = {"s": "Steatosis", "b":"Hepatocellular ballooning", "i": "Lobular inflammation", "f":"Fibrosis"}
tp_dict = {"Baseline":["Paired Screen", "Screen Eligibility", "SCREEN"], "Post Baseline":["Paired Month 18",
                                                                                         "WK48", "UNSCHED"]}
manual_drop_cols = [ "job ID", 'NASH or not NASH', 'NASH biopsy adequacy','NASH biopsy adequacy - pt 2','NASH adequacy other','Additional Comments']
aim_drop_cols = ["derivation_s", "derivation_b", "derivation_f", "derivation_i"]
special_params_v2 = ["F2_F3", "NAS>=4_with_1s", "NASH_Resolution_no_fibrosis", "NASH_Resolution_with_fibrosis"]
files_path = "Production_data/"
aim_df_base = pd.read_csv(files_path + "AIM_base_scores_032923.csv")
manifest_df = pd.read_csv(files_path + "CTS CV - Final Cumulative Manifest.csv")
aim_df = pd.read_csv(files_path + "AIM_final_scores_production_032123.csv")
rr_df = pd.read_csv(files_path + "RR_final_scores_042723.csv")
gt_df = pd.read_csv(files_path + "GT_final_scores_032123.csv")
gt_df = gt_df.rename(columns = {"H&E Slide": "H & E Slide"})
manifest_df["timepoint"] = manifest_df["timepoint"].str.replace("\n ", "")
cv_object = CVAnalysis(cd_dict_m, cd_dict_a, "identifier","identifier", gt_df, manifest_df, 
                     labels_dict, files_path, special_params, tp_dict, special_params_v2, util_dict, cd_dict_p, patho_suffix_aim,agree_suffix_aim)
aim_df =aim_df.loc[aim_df.identifier.isin(list(manifest_df.identifier))]
aim_bootstrap_df = pd.read_csv(files_path + "NASH_DDT_CV_production_data_bootstrap_accuracy_aim_032123.csv")
manual_bootstrap_df = pd.read_csv(files_path + "NASH_DDT_CV_production_data_bootstrap_accuracy_manual_042723.csv")
test = True
if test:
    n_iterations = 5
else:
    n_iterations = 2000
    
print("Generating results for Accuracy (Overall).")
accuracy_df_all = cv_object.get_accuracy_df(aim_df, rr_df)
accuracy_bootstrap_all = cv_object.get_accuracy_df_bootstrap(aim_bootstrap_df, manual_bootstrap_df,"all",n_iterations = n_iterations)
res_df_accuracy_all = cv_object.get_result_df_with_ci(accuracy_df_all, accuracy_bootstrap_all, endpoint = "Accuracy",s_col = None, n_iterations = n_iterations)
pd.concat(accuracy_bootstrap_all).to_csv(files_path + "CV_bootstrap_results_overall.csv", index = False)
res_df_accuracy_all.to_csv(files_path + "Accuracy AI-assisted vs GT.csv", index = False)
del(accuracy_bootstrap_all)

print("Generating results for Accuracy(MASH aggregate components)")
accuracy_df_special_v2 = cv_object.get_accuracy_df_special_v2(aim_df, rr_df)
accuracy_bootstrap_special_v2 = cv_object.get_accuracy_df_bootstrap(aim_bootstrap_df, manual_bootstrap_df,"special_v2",n_iterations = n_iterations)
res_df_accuracy_special_v2 = cv_object.get_result_df_with_ci(accuracy_df_special_v2, accuracy_bootstrap_special_v2, endpoint = "Accuracy",s_col = None, n_iterations = n_iterations)
# print(res_df_accuracy_special_v2)
res_df_accuracy_special_v2.to_csv(files_path + "Accuracy AI-assisted vs GT(MASH aggregate components).csv", index = False)
del(accuracy_bootstrap_special_v2)

print("Generating exploratory results")
cv_object_algo = CVAnalysis(cd_dict_m, cd_dict_a1, "identifier","identifier", gt_df, manifest_df, 
                     labels_dict, files_path, special_params, tp_dict, special_params_v2, util_dict, cd_dict_p, patho_suffix_aim,agree_suffix_aim)

accuracy_df_algo = cv_object_algo.get_accuracy_df(aim_df_base, rr_df)
aim_bootstrap_df_algo = pd.read_csv(files_path + "NASH_DDT_CV_production_data_bootstrap_accuracy_aim_v2_032123.csv")
accuracy_bootstrap_all_algo = cv_object_algo.get_accuracy_df_bootstrap(aim_bootstrap_df_algo, manual_bootstrap_df,"all",n_iterations = n_iterations)
pd.concat(accuracy_bootstrap_all_algo).to_csv(files_path + "NASH_DDT_CV_accuracy_all_algo_bootstraps.csv", index = False)
res_df_all_algo = cv_object_algo.get_result_df_with_ci(accuracy_df_algo,accuracy_bootstrap_all_algo, n_iterations = n_iterations)
res_df_all_algo.to_csv(files_path + "Accuracy AIM-NASH vs GT.csv", index = False)

print("Generating median/mode analysis results")
component_map = {
    "Hepatocellular ballooning": {
        "ground_truth": {
            "low": "Ground truth Hepatocellular ballooning lower median",
            "high": "Ground truth Hepatocellular ballooning upper median",
        },
        "manual": {
            "low": "Manual Hepatocellular ballooning lower median",
            "high": "Manual Hepatocellular ballooning upper median",
        },
        "model": "balloon_final_score",
    },
    "Lobular inflammation": {
        "ground_truth": {
            "low": "Ground truth Lobular inflammation lower median",
            "high": "Ground truth Lobular inflammation upper median",
        },
        "manual": {
            "low": "Manual Lobular inflammation lower median",
            "high": "Manual Lobular inflammation upper median",
        },
        "model": "inflam_final_score",
    },
    "NASH Steatosis": {
        "ground_truth": {
            "low": "Ground truth NASH steatosis lower median",
            "high": "Ground truth NASH steatosis upper median",
        },
        "manual": {
            "low": "Manual NASH steatosis lower median",
            "high": "Manual NASH steatosis upper median",
        },
        "model": "steatosis_final_score",
    },
    "NASH CRN Score": {
        "ground_truth": {
            "low": "Ground truth NASH CRN Score lower median",
            "high": "Ground truth NASH CRN Score upper median",
        },
        "manual": {
            "low": "Manual NASH CRN Score lower median",
            "high": "Manual NASH CRN Score upper median",
        },
        "model": "crn_final_score",
    },
}

def recode_cols(df, col_list, labels_list, alt_val = 9):
    for col in col_list:
        for label in labels_list:
            df[col] = df[col].replace({label:alt_val})
    return df
def average_metrics_for_component(median_scores: pd.DataFrame, component: str, kappa = True) -> Dict[str, float]:
    aggregated_metrics = []
    average_metrics = {}
    for ground_truth in ["low", "high"]:
        for manual in ["low", "high"]:
            aggregated_metrics.append(compute_metrics_for_component(median_scores, ground_truth, manual, component, kappa=kappa))
    for key in ["manual", "model", "difference"]:
        average_metrics[key] = np.array([metrics[key] for metrics in aggregated_metrics]).mean()
    return average_metrics
def bootstrap_average_metrics_for_component_gt_model(median_scores: pd.DataFrame, component: str, num_bootstraps: Optional[int] = 1000) -> List[Dict[str, float]]:
    bootstrap_metrics = []
    for i in range(num_bootstraps):
        bootstrap_df = median_scores.sample(frac=1, replace=True, random_state=i)
        bootstrap_metrics.append(average_metrics_for_component_gt_model(bootstrap_df, component))
    return bootstrap_metrics
def get_confidence_interval_from_bootstrap(bootstrap_metrics: List[Dict[str, float]],
                                           bounds: List[float]) -> Dict[str, Dict[str, float]]:
    confidence_intervals = {}
    for key in bootstrap_metrics[0].keys():
        aggregated_metric = [bootstrap_metric[key] for bootstrap_metric in bootstrap_metrics]
        confidence_intervals[key] = np.nanquantile(aggregated_metric, q=bounds)
    return confidence_intervals
def get_n_for_component(median_scores: pd.DataFrame, 
                        ground_truth: str, manual: str, component: str) -> List[float]:
    
    ground_truth_col = component_map[component]["ground_truth"][ground_truth]
    manual_col = component_map[component]["manual"][manual]
    model_col = component_map[component]["model"]
    manual_df = median_scores[[manual_col, ground_truth_col]].dropna()
    model_df = median_scores[[model_col, ground_truth_col]].dropna()
    return {"N model":len(model_df), "N manual": len(manual_df)}
def get_point_estimate_and_confidence_interval_for_component(median_scores: pd.DataFrame,
                                                             component: str, bounds: List[float],
                                                             num_bootstraps: Optional[int] = 1000, kappa= True):
    point_estimate = average_metrics_for_component(median_scores, component, kappa = kappa)
    bootstrap_metrics = bootstrap_average_metrics_for_component(median_scores, component, num_bootstraps, kappa=kappa)
    confidence_intervals = get_confidence_interval_from_bootstrap(bootstrap_metrics, bounds)
    num_slides_dict = get_n_for_component(median_scores, "low", "low", component)
    if kappa:
        pvalue = get_pvalue_from_bootstrap(bootstrap_metrics)
        return {"N model": num_slides_dict["N model"],
                "N manual": num_slides_dict["N manual"],
                "point_estimate": point_estimate,
                "confidence_intervals": confidence_intervals,
            "pvlaue":pvalue}
    else:
        return {"N model": num_slides_dict["N model"],
                "N manual": num_slides_dict["N manual"],
                "point_estimate": point_estimate,
                "confidence_intervals": confidence_intervals}
def get_point_estimate_and_confidence_interval_for_all_components(median_scores: List[pd.DataFrame],
component_names: List[str], bounds: List[float], num_bootstraps: Optional[int] = 1000,
kappa = True):
    complete_metrics = {}
    for median_df, component_name in zip(median_scores, component_names):
        complete_metrics[component_name] = get_point_estimate_and_confidence_interval_for_component(median_df, component_name, bounds, num_bootstraps, kappa= kappa)
        
    return complete_metrics
component_names = ["Hepatocellular ballooning", "Lobular inflammation", "NASH Steatosis", "NASH CRN Score"]

median_score_dfs_updated = [pd.read_excel("median_scores_updated.xlsx", sheet_name=sheet_name) for sheet_name in component_names]
num_bootstraps = n_iterations
bounds = [.025, .975]
complete_metrics_updated = get_point_estimate_and_confidence_interval_for_all_components(median_scores = median_score_dfs_updated,component_names=component_names,bounds=bounds,num_bootstraps=num_bootstraps)
df_list = []
for key, values in complete_metrics_updated.items():
    v_df = pd.DataFrame(values)
    v_df.insert(0, "Method", list(v_df.index))
    v_df.insert(0, "Component", [key] * len(v_df))
    df_list.append(v_df)
res_df = pd.concat(df_list)
res_df.to_csv("Accuracy AI-assisted vs. mode/median GT.csv", index = False)