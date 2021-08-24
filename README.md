# foundcog-adult-pilot-2-analysis

## s3 path
s3://foundcog-adult-pilot/foundcog-adult-pilot-2/

## s3 folder contents
**bids** : bids structured raw data that were used as input to fmriprep
**deriv-2_topup** : the outputs of fmriprep. This folder is in .gitignore and should be synced to/from s3
**models** : the pickle file results of our first level modelling with nilearn. This folder is in .gitignore and should be synced to/from s3
**results** : *whatever was put in the results from analysis 1 ... to be defined*

# logs
### COD 16/08/21
- Created single script for glm fitting which includes the load_and_estimate function from previous analyses
- Ran glm fitting for subjects 2 to 14

### COD 17/08/21
- Transferred functions from adult_fmri_analysispipeline.py into a script for MVPA
- I attempted to make this script more generalisable for future tasks, but much remains to be done
- Created a separate script for commonly used function called analysis_funcs. This is so that we can easily import functions as we go forward
-restructured the folders with results to include higher with-confounds structure. This is to coincide with first analysis which has v1 and v2, and will enable checking if we want to go back and test with no confounds

- Results for with betas calculated per hemisphere and no subtraction of the mean are present for subj 2-14. Remaining to be calculated after further preprocessing

### COD 24/08/21
- Added an option in the glm_modelling script to split movies according to a specified # seconds. This also tags the repetitions of a trial_type and has been hard coded in the mark_reps block to read the order from the expt_history file. This will need to be streamlined at a later date.

**note** all results and code is a work in progress. Code testing, debugging and thorough sanity checking is yet to be completed.

