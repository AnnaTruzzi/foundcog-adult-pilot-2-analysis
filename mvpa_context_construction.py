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

import boto3
from sklearn.preprocessing import normalize

def mvpa_betas_4context(models, subjind, task, tasklist,segment_list, conditions):
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
    numrep = params['n_reps']

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

def mvpa_rdms(betas, roi, params, mainpth, task, segment_len, conditions, randomise_columns_for_testing=False, hemilist=['L','R'], mvpa_across_hemi=False, figpth='figs', resultspth='results', tosubtract='none'):
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
    
    if os.getlogin()=='CUSACKLAB/clionaodoherty':
        glasserpth = './glasser_et_al_2016'
    if os.getlogin()=='CUSACKLAB\\annatruzzi':
        glasserpth  = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/glasser_et_al_2016'
    
    mmp = get_rois(roi, glasserpth=glasserpth)
    
    # Plotting the chosen ROIs onto the surface model allow us to check 
    # if the selected areas are correct
    if not os.path.exists(os.path.join(figpth)):
        os.makedirs(os.path.join(figpth))
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
        if not os.path.exists(os.path.join(resultspth,)):
            os.makedirs(os.path.join(resultspth))
        with open(os.path.join(resultspth,f'allsubj_task-{task}_hemi-{hemi}_roi-{roi}_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'),'wb') as f:
            pickle.dump(rdm,f)
            
        # Structure as dataframe for visualisation
        condbyrun = [f'{cond}' for cond in conditions]
        rdm_df = pd.DataFrame(rdm, index=condbyrun, columns=condbyrun)
        
        fig, ax =plt.subplots(figsize=(8,8))
        sns.heatmap(rdm_df, ax=ax)
        plt.title(f'RDM for task {task} in {hemi}HS {roi} - allsubj')
        if not os.path.exists(os.path.join(figpth,f'sub-{subjind}')):
            os.makedirs(os.path.join(figpth,f'sub-{subjind}'))
        plt.tight_layout()
        plt.savefig(os.path.join(figpth,f'allsubj_task-combined_hemi-{hemi}_roi-{roi}_subtract-{tosubtract}_rdm_segment-{segment_len}.png'))
        plt.close()
        s3.upload_file(os.path.join(figpth, f'allsubj_task-combined_hemi-{hemi}_roi-{roi}_subtract-{tosubtract}_rdm_segment-{segment_len}.png'), bucket, f'foundcog-adult-pilot-2/pilot-2-full-pipeline/figs/allsubj_task-combined_hemi-{hemi}_roi-{roi}_subtract-{tosubtract}_rdm_segment-{segment_len}.png')

        '''
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
        plt.close()'''

    return rdm


def merge_rdms_acrosssubj():
    for segment in segment_list:
        seg_rdm_list = []
        for sub in subjects:
            with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_roi-{roi}_across-runs-reps_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'),'rb') as f:
                fullsub_rdm = pickle.load(f)
            seg_idx = [i for i, cond in enumerate(conditions) if str(segment) in cond.split('_')[1]] 
            seg_rdm_list.append(fullsub_rdm[np.ix_(seg_idx, seg_idx)])
        seg_rdm = np.concatenate()


def makecontrast(rdm, conditions, forget = False):
    # Structure as dataframe for visualisation
    condbyrun = [f'{cond}' for cond in conditions]
    rdm_contrast = pd.DataFrame(rdm, index=condbyrun, columns=condbyrun)
    for r,row in enumerate(rdm_contrast.iterrows()):
        row_name = row[0]
        row_subj = row_name.split('_')[0]
        row_order = row_name.split('_')[-1]
        row_movie = row_name.split('_')[2]
        if 'new' in row_movie:
            row_movie = 'new_orleans.mp4'
        if 'minions' in row_movie:
            row_movie = 'minions_supermarket.mp4'
        if forget:
            row_movie_idx = np.where(np.array(allorders[int(row_order.split('-')[-1])]) == row_movie)
            row_movie = allorders[int(row_order.split('-')[-1])][int(row_movie_idx[0])-1]
        for c,col in enumerate(list(rdm_contrast.keys())):
            col_subj = col.split('_')[0]
            col_movie = col.split('_')[2]
            if 'new' in col_movie:
                col_movie = 'new_orleans.mp4'
            if 'minions' in row_movie:
                col_movie = 'minions_supermarket.mp4'
            col_order = col.split('_')[-1]   

            if row_subj == col_subj:
                rdm_contrast.loc[row[0]][col] = 0
            elif row_movie == col_movie and row_order == col_order:
                print(row_movie,row_order)
                rdm_contrast.loc[row[0]][col] = 1
            elif row_movie == col_movie and row_order != col_order:
                rdm_contrast.loc[row[0]][col] = -1
            else:
                rdm_contrast.loc[row[0]][col] = 0

    fig, ax =plt.subplots(figsize=(8,8))
    sns.heatmap(rdm_contrast, ax=ax)
    if forget:
        contrast_type = 'forget'
    else:
        contrast_type = 'construct'
    plt.title(f'{contrast_type} context contrast')
    if not os.path.exists(figpth):
        os.makedirs(figpth)
    plt.tight_layout()
    plt.savefig(os.path.join(figpth,f'fig_{contrast_type}context_contrast_segmentlen-{segment_len}.png'))
    plt.show()
    plt.close()
    s3.upload_file(os.path.join(figpth, f'fig_{contrast_type}context_contrast_segmentlen-{segment_len}.png'), bucket, f'foundcog-adult-pilot-2/pilot-2-full-pipeline/figs/fig_{contrast_type}context_contrast_segmentlen-{segment_len}.png')

    return rdm_contrast


def calculate_similarity(roi,rdm,seg,contrast):
    seg_index = [value for value in list(np.where(np.array([key.split('_')[-3] for key in list(contrast.keys())[1:]]) == f'segment-{seg}'))]
    labels = [key for key in list(contrast.keys())[1:] if key.split('_')[-3]==f'segment-{seg}']
    contrast = contrast.drop('Unnamed: 0', axis=1)
    contrast_seg = contrast.to_numpy()[seg_index[0],:][:,seg_index[0]]
    one_idx = np.where(contrast_seg==1)
    minus_one_idx = np.where(contrast_seg==-1)
    contrast_seg[one_idx] = 1/len(one_idx[0])
    contrast_seg[minus_one_idx] = -1/len(minus_one_idx[0])
    rdm_seg = rdm[seg_index[0],:][:,seg_index[0]]

    #construct_rdm = np.sum(np.dot(contrast_seg,rdm_seg))
    similarity_value = np.sum(contrast_seg*rdm_seg)

    return similarity_value


def makeplot(contrast_type):
    contrast = pd.read_csv(os.path.join(modelpth,f'{contrast_type}_context_contrast.csv'),sep=',')
    dot_product = []
    roi_plot = []
    seg_plot= []
    for roi in roi_list:
        print(roi)
        with open(os.path.join(modelpth, f'roi-{roi}_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'), 'rb') as f:
                rdm = pickle.load(f)
        for seg in segment_list:
            print(seg)
            roi_plot.append(roi)
            seg_plot.append(seg)
            construct = calculate_similarity(roi,rdm,seg,contrast)
            dot_product.append(construct)
    
    plot_dict = {'roi': roi_plot,
                 'segment': seg_plot,
                 'dot_product': dot_product}
    plot_df = pd.DataFrame(plot_dict)
    sns.barplot(x='segment', y='dot_product', hue='roi', data=plot_df)
    plt.savefig(os.path.join(figpth,f'barplot_{contrast_type}_allsubj_segment{segment_len}.png'))
    plt.close()
    s3.upload_file(os.path.join(figpth, f'barplot_{contrast_type}_allsubj_segment{segment_len}.png'), bucket, f'foundcog-adult-pilot-2/pilot-2-full-pipeline/figs/barplot_{contrast_type}_allsubj_segment{segment_len}.png')



if __name__ == '__main__':
    hemilist = ['L','R']
    subjects = list(range(2,19))
    #subjects=[2]
    subjects = [f'{subjind:02}' for subjind in subjects]
    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    movie_len = 22.5
    segment_len = 3.75
    segment_list = np.arange(0,int(movie_len//segment_len))


    if os.getlogin()=='CUSACKLAB\\annatruzzi':
        mainpth = '/home/CUSACKLAB/annatruzzi/foundcog-adult-pilot-2-analysis/'
        modelpth = os.path.join(mainpth,'models')
    if os.getlogin()=='CUSACKLAB/clionaodoherty':
        mainpth = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/with-confounds/'
        modelpth = os.path.join(mainpth,'models')

    figpth = os.path.join(mainpth,'figs')
    resultspth = os.path.join(mainpth,'results')

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
    
    allorders = {
        1: ['bathsong.mp4', 'dog.mp4', 'new_orleans.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'piper.mp4'], 
        2: ['minions_supermarket.mp4', 'piper.mp4', 'forest.mp4', 'dog.mp4', 'bathsong.mp4', 'new_orleans.mp4'], 
        3: ['forest.mp4', 'minions_supermarket.mp4', 'piper.mp4', 'bathsong.mp4', 'new_orleans.mp4', 'dog.mp4'], 
        4: ['dog.mp4', 'new_orleans.mp4', 'bathsong.mp4', 'piper.mp4', 'minions_supermarket.mp4', 'forest.mp4'], 
        5: ['new_orleans.mp4', 'bathsong.mp4', 'dog.mp4', 'forest.mp4', 'piper.mp4', 'minions_supermarket.mp4'], 
        6: ['piper.mp4', 'forest.mp4', 'minions_supermarket.mp4', 'new_orleans.mp4', 'dog.mp4', 'bathsong.mp4']
    }

    expinfo = pd.read_csv('expt_history_pilot_2.csv', index_col=0)

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

    task = 'video'
    params = tasklist[task]
#    if remap_pictures:
#        params['trial_types'] = tasklist['video']['trial_types']


    get_betas = False
    if get_betas:
        for subjind in subjects:
            orders = expinfo[expinfo['participantID'] == int(subjind)].orders.to_list()
            orders = orders[0].split('-')
            numrep = tasklist[task]['n_reps']
            conditions = []
            for trial in params['trial_types']:
                for segment in segment_list:
                    for repidx,rep in enumerate(range(0,numrep)):
                        order = orders[repidx]
                        conditions.append(f'{trial}_segment-{segment}_rep-{rep}_order-{order}')            
            if not os.path.exists(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle')):
                s3.download_file(bucket, f'foundcog-adult-pilot-2/segmented/sub-{subjind}/sub-{subjind}_task-{task}_segment-{segment_len}_models.pickle', f'sub-{subjind}_task-{task}_segment-{segment_len}_models.pickle')
                os.system(f'mv sub-{subjind}_task-{task}_segment-{segment_len}_models.pickle ./temp')
                with open(os.path.join(mainpth,'temp',f'sub-{subjind}_task-{task}_segment-3.75_models.pickle'),'rb') as f:
                    models=pickle.load(f)
                
                betas = mvpa_betas_4context(models, subjind,task,tasklist,segment_list, conditions)
                if not os.path.exists(os.path.join(modelpth,f'sub-{subjind}')):
                    os.makedirs(os.path.join(modelpth,f'sub-{subjind}'))
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle'),'wb') as f:
                    pickle.dump(betas,f)
                os.system(f'rm ./temp/sub-{subjind}_task-{task}_segment-{segment_len}_models.pickle')
                s3.upload_file(os.path.join(modelpth, f'sub-{subjind}', f'sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle'), bucket, f'foundcog-adult-pilot-2/segmented/sub-{subjind}/sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle')

            else:
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle'),'rb') as f:
                    betas = pickle.load(f)
    
    full_conditions = []
    for subjind in subjects:
        orders = expinfo[expinfo['participantID'] == int(subjind)].orders.to_list()
        orders = orders[0].split('-')
        numrep = tasklist[task]['n_reps']
        numruns = tasklist[task]['numruns']
        for run in range(0,numruns):
            for trial in params['trial_types']:
                for segment in segment_list:
                    for repidx,rep in enumerate(range(0,numrep)):
                        order = orders[repidx]
                        full_conditions.append(f'subj-{subjind}_run-{run}_{trial}_segment-{segment}_rep-{rep}_order-{order}')            

    get_rdms = False
    if get_rdms:
        for roi in roi_list: 
            fullbetas = {'L':[], 'R': []}
            for subjind in subjects:
                with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}{remap}_segment-{segment_len}_betas.pickle'),'rb') as f:
                    betas = pickle.load(f)
                    fullbetas['L'].append(betas['L'])
                    fullbetas['R'].append(betas['R'])
            fullbetas['L'] = np.concatenate(fullbetas['L'], axis=1)
            fullbetas['R'] = np.concatenate(fullbetas['R'], axis=1)
            rdms = mvpa_rdms(fullbetas, roi, params, mainpth, task, segment_len, full_conditions, randomise_columns_for_testing=randomise_columns_for_testing, tosubtract=tosubtract, mvpa_across_hemi=across_hemi)
        # Save between-run RDMS values per visual area/subject (in each rdm file both hemisphere are included)
            with open(os.path.join(modelpth, f'roi-{roi}_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'), 'wb') as f:
                pickle.dump(rdms, f)
            s3.upload_file(os.path.join(modelpth, f'roi-{roi}_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'), bucket, f'foundcog-adult-pilot-2/pilot-2-full-pipeline/results/roi-{roi}_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle')

    get_construct_contrast = False
    if get_construct_contrast:
        with open(os.path.join(modelpth, f'roi-ventralvisual_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'), 'rb') as f:
                rdm = pickle.load(f)
        construct_contrast = makecontrast(rdm,full_conditions)
        construct_contrast.to_csv(os.path.join(modelpth,'construct_context_contrast.csv'),sep=',')

    get_forget_contrast = False
    if get_forget_contrast:
        with open(os.path.join(modelpth, f'roi-ventralvisual_allsubj_rdms{randomise_for_testing_flag}_subtract-{tosubtract}_segment-{segment_len}.pickle'), 'rb') as f:
                rdm = pickle.load(f)
        forget_contrast = makecontrast(rdm,full_conditions,forget=True)
        forget_contrast.to_csv(os.path.join(modelpth,'forget_context_contrast.csv'),sep=',')

    contrast_list = ['construct','forget']
    for contrast in contrast_list:
        makeplot(contrast_type=contrast)

