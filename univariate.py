from re import sub
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import nilearn
from nilearn.glm import compute_contrast
from nilearn import plotting

from scipy.stats import norm


def get_contrasts(contrasts,desmat,allcons=None):
    """
    Given a design and a list of filters for contrasts, build a set of contrasts for use with nistats
    Hard coded to create separate even and odd contrasts 
    Parameters: 
        contrasts - a dictionary of contrasts created so far (or None)
        desmat - design matrix as output by nistats FirstLevelModel
        allcons - list of strings to search for in design matrix column names. All columns that match will be included in contrast
    """
    if not contrasts:
        contrasts={}
        
    # Conditions vary by run in this expriment
    contrast_matrix = np.eye(desmat.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(desmat.columns)])
    # Make contrasts from sums of matching basic_contrasts
    bc=basic_contrasts.keys()
    
    # Each contrast, odd or even
    for con, conelements in allcons.items():
        if not con in contrasts:
            contrasts[con]=[]

        contrasts[con].append(sum([basic_contrasts[x] for x in set(conelements).intersection(set(bc))]))

    return contrasts

def get_all_contrasts(fmri_glm,task, params):
    contrasts=None  
    for runind in range(params['numruns']):         
        # Get contrasts    
        #  For main effects
        contrasts=get_contrasts(contrasts,fmri_glm.design_matrices_[runind], allcons={task:params['trial_types']}) 
    
    return contrasts

def show_contrasts(subjind, numruns,fmri_glm,surf_glm,contrasts,fsaverage,derivedroot,figpth):
    
    subjpth=f'sub-{subjind}/func'

    hemilist=['left','right']
    # Iterate over contrasts
    z=None
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % i out of %i: %s, right hemisphere' %
            (index + 1, len(contrasts), contrast_id))

        # Surface analysis

        z=None

        for hemiind,hemi in enumerate(hemilist):
            for runind in range(numruns):
                # compute contrast-related statistics
                contrast = compute_contrast(surf_glm[hemiind][runind][0], surf_glm[hemiind][runind][1], contrast_val[runind],
                                            contrast_type='t')
                # we present the Z-transform of the t map
                if z is None:
                    z=np.array(contrast.z_score())
                else:
                    z=np.vstack([z,contrast.z_score()])
                # we plot it on the surface, on the inflated fsaverage mesh,
                # together with a suitable background to give an impression
                # of the cortex folding.

            z_score=np.mean(z,axis=0)

            plotting.plot_surf_stat_map(
                fsaverage.infl_left if hemi=='left' else fsaverage.infl_right, z_score, view='ventral',hemi=hemi,
                title=contrast_id, colorbar=True,
                threshold=1.5, bg_map=fsaverage.sulc_left if hemi=='left' else fsaverage.sulc_right)

            plt.savefig(os.path.join(figpth,'sub-%s_surf_%s_%s.png' % (subjind, hemi, contrast_id)))

            plt.close()

        # Volume analysis
        z_map = fmri_glm.compute_contrast(contrast_val, output_type='z_score')

        z_image_path=os.path.join(derivedroot,'fmriprep',subjpth,'%s_z_map.nii.gz' % contrast_id)
        os.makedirs(os.path.dirname(z_image_path), exist_ok=True)
        z_map.to_filename(z_image_path)
        # we plot it on the surface, on the inflated fsaverage mesh,
        # together with a suitable background to give an impression
        # of the cortex folding.
        plotting.plot_glass_brain(z_map, colorbar=True, threshold=norm.isf(0.001),
                            title='Nistats Z map of "%s" (unc p<0.001)' % contrast_id,
                            plot_abs=False, display_mode='ortho',vmin=-30,vmax=30)

        plt.savefig(os.path.join(figpth,'sub-%s_vol_%s.png' % (subjind, contrast_id)))

        plt.close()

if __name__ == '__main__':
    subjects = list(range(3,19))
    subjects = [f'{subjind:02}' for subjind in subjects]

    mainpth = '/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis/with-confounds/'
    modelpth = os.path.join(mainpth,'models')
    derivedpth = os.path.join('/home/CUSACKLAB/clionaodoherty/foundcog-adult-pilot-2-analysis','deriv-2_topup')
    figpth = os.path.join(mainpth,'univar_results')

    fsaverage = nilearn.datasets.fetch_surf_fsaverage()

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
                        'trial_types':[ 'bathsong.mp4', 'dog.mp4', 'new_orelans.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'piper.mp4'],
                        'n_reps':1
                        }
                }
    
    task='pictures'
    params=tasklist[task]
    for subjind in subjects:
        with open(os.path.join(modelpth,f'sub-{subjind}',f'sub-{subjind}_task-{task}_models.pickle'),'rb') as f:
            models=pickle.load(f)
        
        contrasts=get_all_contrasts(models['fmri_glm'],task, params)
        show_contrasts(subjind, params['numruns'], models['fmri_glm'],models['surf_glm'],contrasts,fsaverage,derivedpth,figpth=figpth)
