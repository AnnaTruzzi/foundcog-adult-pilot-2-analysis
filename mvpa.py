import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.construct import rand
import seaborn as sns

from scipy.spatial import distance

from analysis_funcs import get_rois, hierarchical_clustering

import nilearn
from nilearn import plotting

def mvpa_betas(models, subjind, task, tasklist):
    """
    Function to extract betas from the fmri models for later use in MVPA

    args:
        models - the dictionary with {'fmri_glm': , 'surf_glm': }
        subjind - the subject index
        task - the name of the task, as defined in bids folder
        tasklist - a dictionary with task information including number of runs and trial types
    
    returns:
        betas - a dictionary with one key per hemisphere and array(numvert, numruns*numconditions) of beta values
    """
    print(f'Subject {subjind}')
    params = tasklist[task]
        
    numruns = params['numruns']                    

    conditions = params['trial_types']

    numvert = len(models['surf_glm'][0][0][0])
    numcond = len(conditions)

    # Putting this straight into two numpy arrays so we don't need to convert later
    betas={'L':np.zeros((numvert, numruns * numcond))}
    betas['R'] = betas['L'].copy()

    ind=0

    # Nesting - runs on the outside, conditions on the inside 
    for runind in range(numruns):
        for trial_type in conditions:
            # Set which trial_type and get relevant cols from design matrix
            cols=models['fmri_glm'].design_matrices_[runind].columns
            colind=cols.get_loc(trial_type)

            for hemi in range(2):
                labels=models['surf_glm'][hemi][runind][0]
                regression_result=models['surf_glm'][hemi][runind][1]
                effect = np.zeros((labels.size))
                for label_ in regression_result.keys():
                    label_mask = labels == label_
                    if label_mask.any():
                        resl = regression_result[label_].theta[colind]
                        effect[label_mask]=resl
                betas[hemilist[hemi]][:,ind]=effect
            ind+=1  
    
    return betas

def mvpa_rdms(betas, roi, params, mainpth, task='pictures', randomise_columns_for_testing=False, hemilist=['L','R'], mvpa_across_hemi=False, figpth='figs', resultspth='results', tosubtract='none'):
    """
    args:
        betas - a dictionary with one key per hemisphere and array(numvert, numruns*numconditions) of beta values
        tasklist - a dictionary with task information including number of runs and trial types

        tosubtract - possible values 'voxelmean', 'none'
    """

    figpth = os.path.join(mainpth,figpth)
    resultspth = os.path.join(mainpth,resultspth)
    
    if mvpa_across_hemi:
        hemilist_mvpa = ['both']
    else:
        hemilist_mvpa = hemilist

    # Shuffle columns if specified
    if randomise_columns_for_testing:
        for hemi in hemilist:
            np.random.shuffle(np.transpose(betas[-1][hemi]))

    # set surface model
    fsaverage = nilearn.datasets.fetch_surf_fsaverage()
    
    mmp = get_rois(roi, glasserpth='./glasser_et_al_2016')
    
    # Plotting the chosen ROIs onto the surface model allow us to check 
    # if the selected areas are correct
    if not os.path.exists(os.path.join(figpth,f'roi_{roi}_mmp_right.png' )):
        plotting.plot_surf_roi(fsaverage.infl_left, np.array(mmp[0]),view='ventral',hemi='left', bg_map=fsaverage.sulc_left)
        plt.savefig(os.path.join(figpth,'roi_%s_mmp_left.png' % roi ))
        plt.close()
        plotting.plot_surf_roi(fsaverage.infl_right, np.array(mmp[1]),view='ventral',hemi='right', bg_map=fsaverage.sulc_right)
        plt.savefig(os.path.join(figpth,'roi_%s_mmp_right.png' % roi ))
        plt.close()

    rdms={key:{} for key in hemilist_mvpa}
    for hemiind, hemi in enumerate(hemilist_mvpa):
        if hemi == 'both':
            # Stack voxels across hemispheres before MVPA
            nvert = sum(mmp[0]) + sum(mmp[1])
            betas_roi = np.vstack([betas[h][mmp[hi], :] for hi, h in enumerate(hemilist) ]) 
        else:
            # For each hemisphere
            nvert = sum(mmp[hemiind])
            # Get betas in ROI - for x in betas syntax removed because writing for one task only
            # TODO: fix for multiple tasks, do hstack. Should stack before doing distance calc
            betas_roi = betas[hemi][mmp[hemiind], :]

        if tosubtract == 'voxelmean':
            betas_roi = betas_roi - np.mean(betas_roi,axis = 1, keepdims=True)
        elif tosubtract == 'none':
            pass
        else:
            raise (f'Unknown to subtract {tosubtract}')
        
        # Calculate RDM                        
        rdm = distance.squareform(distance.pdist(betas_roi.T, metric='correlation'))
        
        # Save RDMS with each run separate
        if not os.path.exists(os.path.join(resultspth,f'sub-{subjind}')):
            os.makedirs(os.path.join(resultspth,f'sub-{subjind}'))
        with open(os.path.join(resultspth,f'sub-{subjind}',f'sub-{subjind}_task-{task}_hemi-{hemi}_roi-{roi}_rdms{randomise_for_testing_flag}_subtract-{tosubtract}.pickle'),'wb') as f:
            pickle.dump(rdm,f)
            
        # Structure as dataframe for visualisation
        condbyrun = [f'{task}_{cond}_{run}' for run in range(params['numruns']) for cond in params['trial_types']]
        rdm_df = pd.DataFrame(rdm, index=condbyrun, columns=condbyrun)
        
        fig, ax =plt.subplots(figsize=(8,8))
        sns.heatmap(rdm_df, ax=ax)
        plt.title(f'RDM for task {task} in {hemi}HS {roi}')
        if not os.path.exists(os.path.join(figpth,f'sub-{subjind}')):
            os.makedirs(os.path.join(figpth,f'sub-{subjind}'))
        plt.tight_layout()
        plt.savefig(os.path.join(figpth,f'sub-{subjind}',f'sub-{subjind}_task-combined_hemi-{hemi}_roi-{roi}_subtract-{tosubtract}_rdm.png'))
        plt.close()

        # Calculate average of between-run RDMs
        ## Two tasks can have different numbers of conditions (if pictures are not collapsed down) so need to take some care to work out where the relevant parts of the RDM are
        blocktask =[task for run in range(params['numruns'])]
        blocklen =[len(params['trial_types']) for run in range(params['numruns'])]
        blockstart = np.cumsum(blocklen)
        blockstart = np.insert(blockstart, 0, 0)
        
        #rdm_summaries is dict with keys "pics",'vids','picvid' and idx lists w run numbers [0,1,2] [3,4]
        rdm_summary = 'picpic'
        runlists = [[0,1,2],[0,1,2]]
        
        # Loop over type of summary (e.g., pics vs pics, vid vs vid, vid vs pics)
        rdms_roi_betweenrunaverage = np.zeros((blocklen[runlists[0][0]], blocklen[runlists[1][0]]))
        count=0

        # Find all possible pairs of between-run comparisons for this summary (e.g., pic block 1 vs. vid block 1; or pic block 1 vs pic block 2)
        for run0 in runlists[0]:
            for run1 in runlists[1]:
                if not run0==run1:
                    rdms_roi_betweenrunaverage+=rdm[blockstart[run0]:blockstart[run0+1], blockstart[run1]:blockstart[run1+1]]
                    count+=1
        rdms_roi_betweenrunaverage/=count    # make average

        # Put into a dataframe for storage and figures
        rdm_df = pd.DataFrame(rdms_roi_betweenrunaverage, index=[tt for tt in tasklist[blocktask[runlists[0][0]]]['trial_types']], columns=[tt for tt in tasklist[blocktask[runlists[1][0]]]['trial_types']])
        rdms[hemi][rdm_summary]=rdm_df

        # Plot between-run-average
        fig, ax =plt.subplots(figsize=(12,8.5))
        sns.heatmap(rdm_df, ax=ax)
        plt.title(f'Between run average RDM for task {task} in {hemi}HS {roi}')
        if not os.path.exists(os.path.join(figpth,f'sub-{subjind}')):
            os.makedirs(os.path.join(figpth,f'sub-{subjind}'))
        plt.savefig(os.path.join(figpth,f'sub-{subjind}',f'sub-{subjind}_comparison-{rdm_summary}_hemi-{hemi}_roi-{roi}_subtract-{tosubtract}_betweenrunrdm.png'))
        plt.close()

    return rdms

def mvpa_acrosssubj(subjlist, mainpth, roi_list, randomise_for_testing_flag='' , remap='', across_subject_average_type='mean', tosubtract='none',figpth='figs', modelpth='models', mvpa_across_hemi=False):
    """
    args:
        across_subject_average_type - which centre to use, either mean or median
    """

    figpth = os.path.join(mainpth,figpth)
    modelpth = os.path.join(mainpth,modelpth)

    if mvpa_across_hemi:
        hemilist_mvpa = ['both']
    else:
        hemilist_mvpa = hemilist
    
    with open(os.path.join(mainpth,'results', f'mvpa_contrast_results_subtract-{tosubtract}{remap}_average-{across_subject_average_type}.txt'), 'w') as conf:                        
        conf.write(f'*** MVPA contrasts using {across_subject_average_type}\n')

        numsubj=len(subjlist)

        # How should we average RDMs across subjects for summary measure and during bootstrapping            
        if across_subject_average_type=='mean':
            avfunc=np.mean
        elif across_subject_average_type=='median':
            avfunc=np.median

        with open(os.path.join(figpth, f'rdmsummary_rdms{randomise_for_testing_flag}{remap}_subtract-{tosubtract}.html'),'w') as fhtml:

            for roi in roi_list:
                fhtml.write(f'<h1>analysis with confounds: ROI {roi} Subtract {tosubtract}</h1>')
                rdms = []
                for subjind in subjlist:
                    with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_roi-{roi}_across-runs-reps_rdms{randomise_for_testing_flag}_subtract-{tosubtract}.pickle'), 'rb') as f:
                        rdms.append(pickle.load(f))

                # Order using videos
                rdms_across_subj=np.stack([rdm[hemi]['picpic'] for rdm in rdms for hemi in hemilist_mvpa], axis=2).transpose([2,0,1]) 
                order_rdm = avfunc(rdms_across_subj, axis=0) 

                dendrofn=os.path.join('mvpa_acrosssubj', f'dendrogram_roi-{roi}_orderby-videos_average-{across_subject_average_type}_subtract-{tosubtract}.png')
                os.makedirs(os.path.join(figpth,'mvpa_acrosssubj'), exist_ok=True)
                
                order = hierarchical_clustering(order_rdm, tasklist['pictures']['trial_types'], outpath = os.path.join(figpth, dendrofn))
                # Bootstrapping parameters
                numboot = 1000
                alpha = 0.5
                
                # Define MVPA contrasts
                # These are summary contrasts in the RDM space - i.e., average of leading diagonal minus average of off diagonal cells
                
                conditions = tasklist['pictures']['trial_types']
                numcond = len(conditions)
                mvpa_contrasts_labels=['identity']   ## Use numpy array not dict for mvpa_contrasts for ease of later broadcasting. So labels must be in same order as mvpa_contrasts
                nummvpa_contrasts = len(mvpa_contrasts_labels)
                mvpa_contrasts = np.zeros((nummvpa_contrasts,numcond,numcond))
                mvpa_contrasts[0] = -np.eye(numcond)/numcond + (1-np.eye(numcond))/(numcond*(numcond-1)) # identity contrast for RDM

                conf.write(f' Task {task} in {roi}\n')

                for hemiind, hemi in enumerate(hemilist_mvpa):
                    fhtml.write(f'<h2>hemisphere {hemi}</h2>')
                    fhtml.write(f'<table><tr>')
                    fhtml.write(f'<td><img src="{dendrofn}" width="600"/></td>')

                    rdm_summary = 'picpic'
                    # Make numsubjects x numcond x numcond stack of rdms for this roi, hemisphere and rdm_summary measure
                    rdms_across_subj=np.stack([rdm[hemi][rdm_summary] for rdm in rdms], axis=2).transpose([2,0,1]) # subjects in axis 0                 
                            
                    # Average across subjects
                    rdms_across_subj_average=avfunc(rdms_across_subj, axis=0)

                    # Calculate contrast values for all MVPA contrasts 
                    conmean = np.sum(np.sum(rdms_across_subj_average * mvpa_contrasts, axis=2), axis=1)

                    # Bootstrapping for CI on these contrast values
                    cons = np.zeros((numboot, nummvpa_contrasts))
                    for bootind in range(numboot):
                        subj_sample = random.choices(range(numsubj), k=numsubj)                                
                        ras_sample = avfunc(rdms_across_subj[subj_sample, :,:], axis=0)
                        cons[bootind,:] = np.sum(np.sum(ras_sample * mvpa_contrasts, axis=2), axis=1)
                    lb=np.percentile(cons, (alpha/2)*100, axis=0)
                    ub=np.percentile(cons, 100 - (alpha/2)*100, axis=0)

                    # Write results
                    for conind in range(nummvpa_contrasts):
                        conf.write(f'{mvpa_contrasts_labels[conind]:12} {hemi} ')
                        conf.write(f'{rdm_summary}: {conmean[conind]:6.3} CI=[{lb[conind]:6.3}, {ub[conind]:6.3}]\t')

                    # Graph up
                    df = pd.DataFrame(rdms_across_subj_average, columns=conditions, index=conditions)
                    df = df.reindex(index=order,columns=order)

                    df = df.rename(columns = {key:key.replace('.mp4','') for key in df.columns}, index=  {key:key.replace('.mp4','') for key in df.index}) # Remove .mp4
                    df.to_csv(os.path.join('./with-confounds/results', f'across-subj_comparison-{rdm_summary}_hemi-{hemi}_roi-{roi}_average-{across_subject_average_type}_subtract-{tosubtract}_rdm.csv'))
                    
                    fig, ax =plt.subplots(figsize=(8,8))
                    sns.heatmap(df, ax=ax)
                    plt.title(f'{rdm_summary} hemi-{hemi} subtract-{tosubtract} {roi}')
                    ax.set_aspect('equal', 'box')
                    plt.tight_layout()
                    os.makedirs(os.path.join(figpth, 'mvpa_acrosssubj'), exist_ok=True)
                    rdmname = f'across-subj_comparison-{rdm_summary}_hemi-{hemi}_roi-{roi}_average-{across_subject_average_type}_subtract-{tosubtract}_rdm.png'
                    plt.savefig(os.path.join(figpth, 'mvpa_acrosssubj',rdmname))
                    fhtml.write(f'<td><img src="mvpa_acrosssubj/{rdmname}" width="600"/></td>')
                    plt.close()
                    fhtml.write(f'</tr></table>')
                    conf.write('\n')
                conf.write('\n')

if __name__ == '__main__':
    hemilist = ['L','R']
    subjects = list(range(2,19))
    #subjects=[18]
    subjects = [f'{subjind:02}' for subjind in subjects]

    mainpth = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/with-confounds/'
    modelpth = os.path.join(mainpth,'models')

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
                        'n_reps':1
                        }
                }

    pic_vid_mapping = { 
                        'bathsong.mp4':['sink', 'shampoo', 'rubberduck', 'towel'],
                        'minions_supermarket.mp4':['shelves', 'shoppingcart', 'soda', 'car'], 
                        'forest.mp4':['tree_', 'flower_', 'rock', 'squirrel'],
                        'new_orleans.mp4':['waiter', 'dishware', 'spoon', 'food'], 
                        'dog.mp4':['dog','cat', 'ball', 'fence'], 
                        'piper.mp4':['seabird','crab', 'fish', 'seashell']}
    
    randomise_columns_for_testing = False
    if randomise_columns_for_testing:
        print('******WARNING RANDOMISING COLUMNS FOR TESTING*******')
        randomise_for_testing_flag = '_random'
    else:
        randomise_for_testing_flag = ''
    
    remap_pictures = False
    if remap_pictures:
        remap='_remap'
    else:
        remap=''
    
    remap_shuffle = False
    if remap_shuffle:
        remap='_remap-shuffle'
    
    tosubtract='voxelmean'
    across_hemi=True
    roi_list = ['ventralvisual','earlyvisual','scene-occipital']

    task = 'pictures'
    params = tasklist[task]
    if remap_pictures:
        params['trial_types'] = tasklist['video']['trial_types']

    get_rdms = False
    if get_rdms:
        for subjind in subjects:
            
            if not os.path.exists(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_betas.pickle')):
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_models.pickle'),'rb') as f:
                    models=pickle.load(f)
                
                betas = mvpa_betas(models, subjind, task='pictures',tasklist=tasklist)
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_betas.pickle'),'wb') as f:
                    pickle.dump(betas,f)
            else:
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_betas.pickle'),'rb') as f:
                    betas = pickle.load(f)
            
            for roi in roi_list:
                #if not os.path.exists(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_roi-{roi}_across-runs-reps_rdms{randomise_for_testing_flag}_subtract-{tosubtract}.pickle')):
                rdms = mvpa_rdms(betas, roi, params, mainpth, randomise_columns_for_testing=randomise_columns_for_testing, tosubtract=tosubtract, mvpa_across_hemi=across_hemi)
                # Save between-run RDMS values per visual area/subject (in each rdm file both hemisphere are included)
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_roi-{roi}_across-runs-reps_rdms{randomise_for_testing_flag}_subtract-{tosubtract}.pickle'), 'wb') as f:
                    pickle.dump(rdms, f)
    
    mvpa_acrosssubj(subjects,mainpth, roi_list, across_subject_average_type='median', tosubtract=tosubtract , mvpa_across_hemi=across_hemi)