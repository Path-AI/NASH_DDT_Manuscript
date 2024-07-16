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
from AV_utils import *
random.seed(1234)
cd_dict_a = {
    "s":'H&E Slide - Steatosis Score (Algorithm Score)',
    "i":'H&E Slide - Lobular Inflammation Score (Algorithm Score)',
    "b":'H&E Slide - Hepatocellular Ballooning Score (Algorithm Score)',
    "f":'Trichrome Slide - CRN Fibrosis Score (Algorithm Score)'
}
cd_dict_m = {"s":"NASH steatosis",
             "i":"Lobular inflammation",
             "b":"Hepatocellular ballooning",
             "f":"NASH CRN Score"
}
#tp_dict = {"Baseline":["Paired Screen", "Screen Eligibility"], "Month 18": ["Paired Month 18"]}
tp_dict = {"Baseline":['Paired Screen','SCREEN','Screen Eligibility'],
           "Post-Baseline": ["UNSCHED","ET", "W24", "Paired Month 18", "WK48"]}
labels_dict = {"s": [0,1,2,3], "b": [0,1,2], "i":[0,1,2,3], "f":[0,1,2,3,4]}
special_params = ["F0_F1", "NAS_4"]
exclude_participants = ["PAI_292"]
special_params_v2 = ["F2_F3", "NAS>=4_with_1s", "NASH_Resolution_no_fibrosis"]

files_path = "AV_production_data/"
print("Reading in Primary files.")
repeatability_df = pd.read_csv(files_path + "CTS_AV_Repeatability_112222.csv")
reproducibility_df = pd.read_csv(files_path + "CTS_AV_Reproducibility_mod_030323.csv")
gs_df = pd.read_csv(files_path + "CTS_AV_GT_Scores_030323.csv")
manual_df = pd.read_csv(files_path + "CTS_AV_Manual_Reads_030323.csv")
gs_df = gs_df.rename(columns = {"PathAI Subject ID": "Identifier"})
manifest_df = pd.read_csv(files_path + "CTS AV - Final Cumulative Manifest 121521 - EDC Master 112222.csv")
manifest_df_scanner = pd.read_csv(files_path + "av_manifest_scanner.csv")
manifest_df = manifest_df.merge(manifest_df_scanner[["slide ID", "Scanner", "PathAI Subject ID"]].drop_duplicates(),on = ["slide ID","PathAI Subject ID"], how = "outer")
repeatability_bootstrap_df = pd.read_csv(files_path + "NASH_DDT_AV_production_data_bootstrap_repeatability_030523.csv")
reproducibility_bootstrap_df = pd.read_csv(files_path + "NASH_DDT_AV_production_data_bootstrap_reproducibility_030523.csv")

av_object = AVanalysis(cd_dict_m, cd_dict_a, "Identifier","Participant", gs_df, manifest_df, tp_dict, 
                     labels_dict, special_params,special_params_v2, files_path, exclude_participants)

test = True
if test:
    n_iterations = 5
else:
    n_iterations = 2000

print("Generating results for Repeatability(Overall).")
repeatability_df_all = av_object.get_rr_df(repeatability_df, "Repeatability")
repeat_bootstrap_all = av_object.get_rr_df_bootstrap(repeatability_bootstrap_df, "Repeatability", "all", n_iterations = n_iterations)
res_df_repeat_all = av_object.get_result_df_with_ci(repeatability_df_all, repeat_bootstrap_all,
                              endpoint = "Repeatabilty",n_iterations = n_iterations)
res_df_repeat_all.to_csv(files_path + "Repeatability mean agreement rate.csv", index = False)
del(repeat_bootstrap_all)

print("Generating results for Reproducibility(Overall).")

repro_df_all = av_object.get_rr_df(reproducibility_df, "Reproducibility")
repro_bootstrap_all = av_object.get_rr_df_bootstrap(reproducibility_bootstrap_df, "Reproducibility", "all", n_iterations = n_iterations)
res_df_repro_all = av_object.get_result_df_with_ci(repro_df_all, repro_bootstrap_all,
                              endpoint = "Reproducibility",n_iterations = n_iterations)
res_df_repro_all.to_csv(files_path + "Reproducibility mean agreement rate.csv", index = False)
del(repro_bootstrap_all)

print("Generating exploratory results")
repeat_df_aim = pd.read_excel(files_path + "NASH AV_2022-11-21_17_17_56.xlsx")

repeat_df_aim.columns = list(repeat_df_aim.iloc[0,:].values)
repeat_df_aim = repeat_df_aim.iloc[1:, :]

repeat_df_aim["Time Point"] = repeat_df_aim["Participant"].apply(lambda x: int(x.split("-")[1]))
repeat_df_aim["Participant"] = [p_id.split("-")[0] for p_id in list(repeat_df_aim.Participant)]

av_exp_obj = AVanalysisExp( cd_dict_m,cd_dict_a,"Identifier","Participant", gs_df, manifest_df, 
                     labels_dict, files_path)
agg_df = av_exp_obj.compile_inter_rater_kappa(manual_df, metric = "Agreement")
agg_df = agg_df.loc[agg_df["Total"] >= 10]
agg_df.to_csv(files_path + "Pairwise pathologist inter-reader agreement rate.csv", index = False)
rr_aim_agg_df= av_exp_obj.compile_rr_mean_agg_df(repeat_df_aim)
rr_aim_agg_df.to_csv(files_path + "AIM-NASH Repeatability mean agreement rate.csv", index = False)
