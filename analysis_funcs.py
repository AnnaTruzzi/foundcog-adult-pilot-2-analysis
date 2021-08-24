import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from nilearn import surface

def get_rois(roi, glasserpth):
    """
    Extracts visual ROIs from the Glasser et al. 2016 parcellation
    
    args:
        roi - either 'ventralvisual' or 'earlyvisual'
    """
    # Check if ROI is a viable option
    roi_options = ['ventralvisual', 'earlyvisual', 'scene-occipital']
    if not roi in roi_options:
        raise NotImplementedError('the specified ROI is not defined. Please select either ventralvisual or earlyvisual. \n Alternatively, implement new ROIs in the get_rois function')

    # Regions as in Glasser et al 2016, Supplementary Neuroanatomical info
    if roi=='ventralvisual':
        visualrois=set([7,18,22, 153,154,160,163])
    elif roi=='earlyvisual':
        visualrois=set([1,4,5,6])
    elif roi=='scene-occipital':
        visualrois=set([155,126,127,14,31,20,21,22,159])

    # Load Glasser et al 2006 labels
    mmp=[]
    mmp.append(surface.load_surf_data(os.path.join(glasserpth, 'lh.HCP-MMP1.fsaverage5.gii')))
    mmp.append(surface.load_surf_data(os.path.join(glasserpth, 'rh.HCP-MMP1.fsaverage5.gii')))

    # Get regions from MMP atlas
    mmp[0]=[(x in visualrois) for x in mmp[0]]
    mmp[1]=[ (x in visualrois) for x in mmp[1]]

    print('Vertices in left ROI %d' % np.sum(mmp[0]))
    print('Vertices in right ROI %d' % np.sum(mmp[1]))

    return mmp

def hierarchical_clustering(matrix, label_list, outpath=None):
    fig,ax = plt.subplots(figsize=(10,10))
    dend = sch.dendrogram(sch.linkage(matrix, method='ward'), 
        ax=ax, 
        labels=label_list, 
        orientation='left'
    )
    ax.tick_params(axis='x', labelsize=4)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
    plt.close()

    cluster_order = dend['ivl']

    return cluster_order