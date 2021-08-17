import pandas as pd
import numpy as np
import glob
import os
from scipy.optimize import curve_fit
import nibabel as nib
from scipy.stats import zscore

if __name__ == '__main__':
    atlas = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz')
    rois_data = atlas.get_fdata()
    filename = "./Results/intrinsic_timescales/estimatedtau_foundcog-pilot-2_resting_run-001.txt"
    ac_data = np.nanmedian(np.loadtxt(filename),axis=0)
    ac_data_zscore = zscore(ac_data,nan_policy='omit')
    outvolume = np.zeros((182, 218, 182))
    outvolume_zscore = np.zeros((182, 218, 182))
    for roi in range(0,400):
        roi_index = np.where(rois_data==roi+1)
        outvolume[roi_index] = ac_data[roi] * 0.656
        outvolume_zscore[roi_index] = ac_data_zscore[roi] * 0.656
        print('roi',roi)
    outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
    outname = ('./Results/intrinsic_timescales/tau_brainrender_resting_run-001.nii.gz')
    nib.save(outimage,outname)

    outimage_zscore = nib.Nifti1Image(outvolume_zscore, affine=atlas.affine)
    outname_zscore = ('./Results/intrinsic_timescales/tau_brainrender_resting_run-001_zscore.nii.gz')
    nib.save(outimage_zscore,outname_zscore)
