# This script will contain functions/classes for fitting our glm to the fMRI data
# Base on load_and_estimate from other analyses pipelines
# Important first step for extracting betas which we will go on to use for MVPA

import os
import pandas as pd
import pickle
import numpy as np

from nilearn import surface
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel, run_glm

def load_and_estimate(subjind,bidsroot,derivedroot,taskname,numruns,t_r=0.656,slice_time_ref=0.5, remap_trial_types=None, elantags=False, conditions=None):
    """
    Load BIDS data, set up volume and surface regression models and estimate them
    
    args:
    subjind     for which subject to process
    bidsroot    path to bids folder
    derived root    path to preprocessed outputs
    taskname    name of the task as per the bids event descriptions
    numruns     number of runs for task as named by taskname
    
    t_r     is volume repetition time in seconds    
    slice_time_ref  point in volume slice timing is corrected to (percentage of t_r can have value between 0 and 1). Is 0.5 if fmriprep was used
    remap_trial_types - dict of form {'remap_from' : 'remap_to'} for use if finer labelling (e.g. elan tagging) is to be used
    elantags, conditions    for remapping if remap_trial_types is not None

    returns:
    fmri_glm    the volume model in MNI152NLin2009cAsym space 
    surf_glm    the surface model in fsaverage space
    """
    # This folder structure is based on fmriprep outputs
    subjpth=f'sub-{subjind}/ses-001/func'
    bidspth=os.path.join(bidsroot,subjpth)
    
    eventsuffix='_events.tsv'
    # For volume modelling
    niisuffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    # As output from fmriprep
    confoundsuffix='_desc-confounds_timeseries.tsv'

    # Get fmri images and events for each run
    fmri_img=[]
    events=[]
    confounds=[]
    texture=[[],[]]
    hemilist=['L','R']

    # State of odds and evens is remembered across runs for later MVPA
    for runind in range(numruns):
        # For each run
        basename = f'sub-{subjind}_ses-001_task-{taskname}_run-{runind+1}'
        # Files for fMRI
        fmri_img.append(os.path.join(derivedroot,'fmriprep',subjpth,basename+niisuffix))
        # Load events
        dfsess=pd.read_csv(os.path.join(bidspth,f'sub-{subjind}_ses-001_task-{taskname}_run-00{runind+1}{eventsuffix}'), sep='\t')        

        # Get rid of fixation events
        dfsess=dfsess[~(dfsess['trial_type'].str.startswith('fixation'))]

        # Apply any requested remappings to trial_type
        dfsess['trial_type'] = dfsess['trial_type'].replace(remap_trial_types)

        if elantags:
            elan = []
            # TODO: Change this to be a universal path
            if os.getcwd() == 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project_FOUNDCOG\\foundcog_adult_pilot':
                elan_pth = 'C:\\Users\\Anna\\Documents\\Research\\Projects\\ONGOING\\Project_FOUNDCOG\\foundcog_adult_pilot\\elan_emily_gk\\events_per_movie_longlist_new.pickle'
            else:
                elan_pth = '/home/CUSACKLAB/annatruzzi/foundcog_adult_pilot/elan_emily_gk/events_per_movie_longlist_new.pickle'
            with open(elan_pth,'rb') as f:
                elan_file = pickle.load(f)
            for trial in dfsess['trial_type']:
                print(trial)
                if trial in conditions:
                    #video_name = trial.split('.')[0]
                    #elan_file = pd.read_csv(os.path.join('elan_emily_gk',f'{video_name}.txt'), sep='\t',header=None) 
                    elan_tags = elan_file[trial]
                    video_onset = np.array(dfsess[dfsess['trial_type']==trial]['onset'])[0]
                    elan_tags.iloc[:,0] = elan_tags.iloc[:,0] + video_onset
                    #elan_tags.iloc[:,3] = elan_tags.iloc[:,3] + video_onset
                    elan.append(elan_tags)
            elan_df = pd.concat(elan)
            elan_df.columns =  ['onset','duration','trial_type','magnitude']
            elan_df.drop(columns=['magnitude'])
            #elan_df = elan_df[['onset', 'duration', 'trial_type']]
            dfsess = elan_df.copy()
            a = 1

        # Add this run
        events.append(dfsess)

        # Load confounds
        condf = pd.read_csv(os.path.join(derivedroot,'fmriprep',subjpth,basename+confoundsuffix),sep='\t')
        condf = condf.drop(columns = [x for x in condf.columns if x.startswith('a_comp')])
        condf = condf.drop(columns = [x for x in condf.columns if x.startswith('tcomp')])
        condf = condf.drop(columns = [x for x in condf.columns if x.startswith('cosine')])
        confounds.append(condf) 
        # Replace NaNs with zero
        confounds[runind]=confounds[runind].fillna(value=0)  

        # Also set up surface model
        # Load fMRI data on surface
        for hemiind, hemi in enumerate(hemilist):
            giisuffix = '_space-fsaverage5_hemi-L_bold.func.gii'
            texture[hemiind].append(surface.load_surf_data(os.path.join(derivedroot,'fmriprep',subjpth,basename+giisuffix)))


    # Volume modelling of all runs
    fmri_glm= FirstLevelModel(t_r=t_r, slice_time_ref=slice_time_ref)
    fmri_glm.fit(fmri_img, events=events, confounds=confounds) 


    surf_glm=[[],[]]
    for runind in range(numruns):
        for hemiind in range(2):
        # Surface modelling of each run individually
            surf_glm[hemiind].append(run_glm(texture[hemiind][runind].T,fmri_glm.design_matrices_[runind].values))
        
    return fmri_glm,surf_glm

if __name__ == '__main__':
    
    # Set paths for saving
    modelpth = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/with-confounds/models'

    # Set arguments for glm fitting
    subjects = [18]
    subjects = [f'{subjind:02}' for subjind in subjects]
    
    bidsroot = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/bids'
    derivedroot = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/deriv-2_topup'

    taskname = 'pictures'
    numruns = 3

    for subjind in subjects:

        print(f'working on subject {subjind}')
        fmri_glm, surf_glm = load_and_estimate(subjind,bidsroot,derivedroot,taskname,numruns)
        
        os.makedirs(os.path.join(modelpth,f'sub-{subjind}'), exist_ok=True)
        
        with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{taskname}_models.pickle'),'wb') as f:
                pickle.dump({'fmri_glm': fmri_glm, 'surf_glm': surf_glm},f)