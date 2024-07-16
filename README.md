# NASH DDT Manuscript analysis
These scripts can be used to generate results for the **Analytical and clinical validation of AIM-NASH: A Digital Pathology Tool for Artificial Intelligence-based Measurement of Nonalcoholic Steatohepatitis Histology** manuscript
- The first set of analysis is used for computing true positive and false positive success rates,per pathologist, for the overlay validation. The run script for this analysis is: `run_overlay_analysis_manuscript.py`
    - **NOTE**: The package [rpy2](https://rpy2.github.io/) is needed to run this script to compute Wilson's score CIs.
- The second set of analysis is used for computing mean agreement rates for testing repeatability and reproducibility of the AIM-NASH algorithm. Additionally, one can also compute mean inter-reader agreement for manual pathologists. The run script for this analysis is: `run_av_analysis_manuscript.py`
- The final set of analysis is used for compare accuracy of the AIM-NASH and the AI-assisted workflows to that of the manual pathologists. The run script for this analysis is: `run_cv_analysis_manuscript.py`
