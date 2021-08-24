# This script will contain functions/classes for fitting our glm to the fMRI data
# Base on load_and_estimate from other analyses pipelines
# Important first step for extracting betas which we will go on to use for MVPA

import os
from numpy.lib.shape_base import split
import pandas as pd
import pickle
import numpy as np
import random

from nilearn import surface
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel, run_glm
from scipy.linalg.special_matrices import tri
from scipy.sparse.construct import rand

def load_and_estimate(subjind,bidsroot,derivedroot,taskname,numruns,t_r=0.656,slice_time_ref=0.5, remap_trial_types=None, elantags=False, conditions=None, segment_into=None, mark_reps=False):
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
    segment_into       number of seconds to split the event into. Set to None for keeping the intact event

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
        # Get rid of dummy events
        dfsess=dfsess[~(dfsess['trial_type'].str.startswith('dummy'))]

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

        if segment_into is not None:
            segmented_df = pd.DataFrame(columns=['onset','duration','trial_type'])
            for idx, row in dfsess.iterrows():
                seg = {'onset':[], 'duration':[],'trial_type':[]}
                split_event = np.arange(row.onset, row.onset+row.duration, segment_into)
                seg['onset'] = list(split_event)

                durs = list(np.diff(split_event))
                durs.append(row.duration % segment_into)
                
                seg['duration']=durs
                
                seg['trial_type']=([f'{row.trial_type}_segment-{i}' for i in range(len(split_event))])

                segmented_df = segmented_df.append(pd.DataFrame.from_dict(seg), ignore_index=True)
            dfsess = segmented_df.copy()
        
        if mark_reps:
            trialtypecount = {}
            for idx, row in dfsess.iterrows():
                tt=row.trial_type
                if not tt in trialtypecount:
                    trialtypecount[tt]=0
                else:
                    trialtypecount[tt]+=1 
                dfsess.loc[idx,'trial_type'] += f'_rep-{trialtypecount[tt]}'
            
                # TODO: make this not crash when we generalise
                orders_on = True
                if orders_on:
                    orders = pd.read_csv('expt_history_pilot_2.csv', index_col=0)
                    orders = orders[orders['participantID'] == int(subjind)].orders.to_list()
                    orders = orders[0].split('-')

                    if trialtypecount[tt] == 0:
                        dfsess.loc[idx,'trial_type'] += f'_order-{orders[0]}'
                    elif trialtypecount[tt] == 1:
                        dfsess.loc[idx,'trial_type'] += f'_order-{orders[1]}'
                    elif trialtypecount[tt] == 2:
                        dfsess.loc[idx,'trial_type'] += f'_order-{orders[2]}'
                    else:
                        raise ValueError('the number of repetitions does not match expt history')

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
    modelpth = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/segmented/models'

    # Set arguments for glm fitting
    subjects = list(range(2,18))
    #subjects=[18]
    subjects = [f'{subjind:02}' for subjind in subjects]
    
    bidsroot = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/bids'
    derivedroot = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/deriv-2_topup'

    taskname = 'video'
    numruns = 2

    tasklist = {
                'pictures':{
                        'numruns':3,
                        'trial_types':[
                            'seabird', 'crab', 'fish', 'seashell',
                            'waiter', 'dishware', 'spoon', 'food',
                            'tree_', 'flower_', 'rock', 'squirrel',
                            'sink', 'shampoo', 'rubberduck', 'towel',
                            'shelves', 'shoppingcart', 'soda', 'car',
                            'dog', 'cat', 'ball', 'fence'],
                        'n_reps':3
                        },
                'video':{
                        'numruns':2,
                        'trial_types':[ 'bathsong.mp4', 'dog.mp4', 'new_orleans.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'piper.mp4'],
                        'n_reps':3
                        }
                }

    pic_vid_mapping = { 
                        'bathsong.mp4':['sink', 'shampoo', 'rubberduck', 'towel'],
                        'minions_supermarket.mp4':['shelves', 'shoppingcart', 'soda', 'car'], 
                        'forest.mp4':['tree_', 'flower_', 'rock', 'squirrel'],
                        'new_orleans.mp4':['waiter', 'dishware', 'spoon', 'food'], 
                        'dog.mp4':['dog','cat', 'ball', 'fence'], 
                        'piper.mp4':['seabird','crab', 'fish', 'seashell']}
    
    remap_pictures = False
    remap_shuffle = False

    # Number of seconds to segment movies into. Set to None for intact movies
    segment_into = 3.75 
    mark_reps = True

    # Pictures will just be represented by their corresponding video this after remapping
    if remap_pictures:
        tasklist['pictures']['trial_types'] = tasklist['video']['trial_types']
        remap='_remap'
        
        if remap_shuffle:
            all_imgs = ['seabird', 'crab', 'fish', 'seashell',
                        'waiter', 'dishware', 'spoon', 'food',
                        'tree_', 'flower_', 'rock', 'squirrel',
                        'sink', 'shampoo', 'rubberduck', 'towel',
                        'shelves', 'shoppingcart', 'soda', 'car',
                        'dog', 'cat', 'ball', 'fence']
            new_groups = []
            for i in range(len(tasklist['video']['trial_types'])):
                x = random.sample(all_imgs,k=4)
                [all_imgs.remove(z) for z in x]
                new_groups.append(x)

            pic_vid_mapping = dict(zip(pic_vid_mapping.keys(),new_groups))
            remap='_remap-shuffle'
        
        remap_pictures_to_video = {z:x for x,y in pic_vid_mapping.items() for z in y}
    else:
        remap=''

    for subjind in subjects:
        print(f'working on subject {subjind}')
        if remap_pictures:
            fmri_glm, surf_glm = load_and_estimate(subjind,bidsroot,derivedroot,taskname,numruns, remap_trial_types=remap_pictures_to_video, segment_into=segment_into, mark_reps=mark_reps)
        else:
            fmri_glm, surf_glm = load_and_estimate(subjind,bidsroot,derivedroot,taskname,numruns, segment_into=segment_into, mark_reps=mark_reps)
        
        os.makedirs(os.path.join(modelpth,f'sub-{subjind}'), exist_ok=True)
        
        with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{taskname}{remap}_segment-{segment_into}_models.pickle'),'wb') as f:
            pickle.dump({'fmri_glm': fmri_glm, 'surf_glm': surf_glm},f)