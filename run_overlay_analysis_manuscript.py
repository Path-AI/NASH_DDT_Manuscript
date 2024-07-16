"""
Run file for generating NASH DDT Overlays results. 
"""
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import random
from overlays_utils import *

names_dict = {"a": "Artifact","f": "Fibrosis", "b":"Hepatocellular Ballooning",
               "i": "Lobular Inflammation", "s": "Macrovesicular Steatosis"}
underestimate_dict = {"s":"Less than or equal to 10%", "f":"Less than or equal to 10%",
                     "b":"Yes", "i":"Yes", "a":"Yes"}
na_dict = {"s":"No feature", "f":"No feature",
          "b":"No ballooning present in frame","i":'No lobular inflammation present in frame',
           "a":"No artifact present in frame"
}
feature_dict_enroll = {"s":"Enrollment NASH DDT Heatmap -- Macrovesicular Steatosis",
                      "i":"Enrollment NASH DDT -- Lobular Inflammation",
                       "a":"Enrollment NASH DDT Heatmap -- Artifact",
                       "b":"Enrollment NASH DDT Heatmap -- Hepatocellular Ballooning",
                       "f": "Enrollment NASH DDT Heatmap – Fibrosis",
                      }
score_dict_enroll = {"b":"Evaluate Hepatocellular Ballooning_Final",
                    "s":"Evaluate Steatosis_Final",
                    "i":"Evaluate Lobular Inflammation_Final",
                    "f" :"Evaluate Fibrosis_Final"}
pathologist_name_dict = {"Hannah Chen":"A", "Michael Torbenson":"B","Xuchen  Zhang":"C"}
# Change to False if not testing to increase the number of bootstrap iterations for computing 95% CIs
test = True
if test:
    n_iterations = 5
else:
    n_iterations = 2000
files_path = "../Production_data/" # Changeable variable based on where the user stores the input files, and wants to store results files
print("Reading files.")
in_df_eval = pd.read_csv(files_path + "012623_heatmap_eval.csv")
in_df_enroll = pd.read_csv(files_path + "burt_final_data.csv")

print("Preprocessing Evaluation file.")
in_df_eval["param_col"] = in_df_eval["feature"].replace({v: k for k, v in names_dict.items()})
in_df_eval["Overestimation"] = in_df_eval.apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x["Overestimation"]), axis = 1)
in_df_eval["Overestimation"] = in_df_eval["Overestimation"].str.rstrip(" ")
in_df_eval["Overestimation"] = in_df_eval["Overestimation"].str.lstrip(" ")
common_frames = sorted(set.intersection(set(in_df_enroll.frame_id), set(in_df_eval.frame_id)))
in_df_eval = in_df_eval.loc[in_df_eval.frame_id.isin(common_frames)]                       
print("Preprocessing Enrollment file.")

in_df_enroll["param_col"] = in_df_enroll["evaluation"].replace({v: k for k, v in feature_dict_enroll.items()})
in_df_enroll = in_df_enroll.loc[in_df_enroll.frame_id.isin(common_frames)]
eval_frames_df = in_df_eval[["param_col", "frame_id", "stain"]].drop_duplicates()
merge_df_enroll = in_df_enroll.merge(eval_frames_df, right_on = ["param_col", "frame_id", "stain"], 
                              left_on = ["param_col", "frame_id", "Stain"], how = "inner")


frames_df_art_he, partner_df_art_he,partner_df_art_he_slide = get_frames_distribution(merge_df_enroll,"a",feature_dict_enroll,"H & E",score_dict_enroll, names_dict)

frames_df_art_tc, partner_df_art_tc,partner_df_art_tc_slide = get_frames_distribution(merge_df_enroll,"a",feature_dict_enroll,"Trichrome",score_dict_enroll, names_dict)

frames_df_steat,slides_df_steat,partner_df_steat,partner_df_steat_slide = get_frames_distribution(merge_df_enroll,"s",feature_dict_enroll,"H & E",score_dict_enroll, names_dict)

frames_df_balloon,slides_df_balloon, partner_df_balloon, partner_df_balloon_slide = get_frames_distribution(merge_df_enroll,"b",feature_dict_enroll,"H & E",score_dict_enroll, names_dict)

frames_df_inflam, slides_df_inflam, partner_df_inflam, partner_df_inflam_slide = get_frames_distribution(merge_df_enroll,"i",feature_dict_enroll,"H & E",score_dict_enroll, names_dict)

frames_df_fib, slides_df_fib, partner_df_fib, partner_df_fib_slide = get_frames_distribution(merge_df_enroll,"f",feature_dict_enroll,"Trichrome",score_dict_enroll, names_dict)

patho_counts_fib = get_per_patho_counts(in_df_eval, names_dict, "f","Trichrome")
patho_counts_art_he = get_per_patho_counts(in_df_eval, names_dict, "a","H & E")
patho_counts_art_tc = get_per_patho_counts(in_df_eval, names_dict, "a","Trichrome")
patho_counts_steat = get_per_patho_counts(in_df_eval, names_dict, "s","H & E")
patho_counts_balloon = get_per_patho_counts(in_df_eval, names_dict, "b","H & E")
patho_counts_inflam = get_per_patho_counts(in_df_eval, names_dict, "i","H & E")
patho_counts_df = pd.concat([patho_counts_art_he,patho_counts_art_tc,patho_counts_steat, patho_counts_balloon,patho_counts_inflam,patho_counts_fib])

frames_df = pd.concat([frames_df_art_he,frames_df_art_tc, frames_df_steat,frames_df_balloon, frames_df_inflam,frames_df_fib ])
slides_df = pd.concat([slides_df_steat,slides_df_balloon,slides_df_inflam,  slides_df_fib ])

print("Writing distribution files.")
# Posptprocessing to add percentages
frames_df["%"] = np.round(frames_df["fraction"]*100,2)
slides_df = slides_df.drop([col for col in list(slides_df) if col.find("Evaluate") != -1], axis = 1)
slides_df["%"] = np.round(slides_df["fraction"]*100,2)
patho_counts_df["%"] = np.round(patho_counts_df["fraction"]*100, 2)
frames_df.to_csv(files_path + "Frame Distribution based on Frames Level Score for Overlay Validation .csv", index = False)
slides_df.to_csv(files_path + "Frame Distribution based on Slide Level Score for Overlay Validation.csv", index = False)
patho_counts_df["Pathologist"] = patho_counts_df["Pathologist"].replace(pathologist_name_dict)
patho_counts_df.to_csv(files_path + "Presence of Feature per Pathologist for Overlay Validation.csv")

print("Computing success rates.")
in_bootstrap_df = pd.read_csv(files_path + "NASH_DDT_Overlay_production_data_bootstrap.csv")
in_bootstrap_df["Overestimation"] = in_bootstrap_df.apply(lambda x: re.sub(r'[^\x00-\x7f]',r'', x["Overestimation"]), axis = 1)
in_bootstrap_df["Overestimation"] = in_bootstrap_df["Overestimation"].str.rstrip(" ")
in_bootstrap_df["Overestimation"] = in_bootstrap_df["Overestimation"].str.lstrip(" ")

tp_df_res,fp_df_res = get_results_df(in_df_eval, in_bootstrap_df, names_dict, underestimate_dict,n_iterations = n_iterations)
fp_df_res_wci = postprocess_results_wilson_ci(fp_df_res)
tp_df_res_patho = tp_df_res.loc[tp_df_res.Pathologist != "All"]
fp_df_res_patho = fp_df_res_wci.loc[fp_df_res_wci.Pathologist != "All"]
tp_df_res_patho["Pathologist"] = tp_df_res_patho["Pathologist"].replace(pathologist_name_dict)
fp_df_res_patho["Pathologist"] = fp_df_res_patho["Pathologist"].replace(pathologist_name_dict)
tp_df_res_patho.to_csv(files_path + "True positive success rate.csv", index = False)
fp_df_res_patho.to_csv(files_path + "False positive success rate.csv", index = False)
                        
                            
