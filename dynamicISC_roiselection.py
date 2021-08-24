from operator import index
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import pickle
import pandas as pd
import glob
import os
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import zscore
import nibabel as nib


def roi_render(index_list,atlas,outname):
    rois_data = atlas.get_fdata()
    outvolume = np.zeros((182, 218, 182))
    for roi in index_list:
        roi_index = np.where(rois_data==roi+1)
        outvolume[roi_index] = 1
    outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
    nib.save(outimage,outname)


if __name__ == '__main__':
    network_file = pd.read_csv(os.path.join('Data','Schaefer2018_400Parcels_17Networks_order.txt'),sep = '\t', header = None)
    roi_net_list = np.array([i.split("_")[2] for i in network_file[1]])
    network_names = list(set(roi_net_list))
    atlas = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz')
    roi_names = np.array(network_file[1])
    earlyvis_index = []
    ventralvis_index = []
    sceneareas_index = []
    for i,roi in enumerate(roi_names):
        if 'VisCent' in roi or 'VisPeri' in roi:
            earlyvis_index.append(i)
        elif 'TempOcc_2' in roi or 'TempOcc_4' in roi or "TempPole" in roi:
            ventralvis_index.append(i)
        elif 'Temp_' in roi or 'TempOcc_1' in roi or 'Rsp' in roi or 'ParOcc' in roi or 'PHC' in roi:
            sceneareas_index.append(i)
    roi_render(earlyvis_index,atlas,os.path.join('Results','ISC','earlyvisual_selection.nii.gz'))
    roi_render(ventralvis_index,atlas,os.path.join('Results','ISC','ventralvisual_selection.nii.gz'))
    roi_render(sceneareas_index,atlas,os.path.join('Results','ISC','sceneareas_selection.nii.gz'))
