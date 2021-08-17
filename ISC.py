import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import random
import boto3
from scipy.stats import spearmanr
import pickle
import nibabel as nib

def average_2runs(movie_list, run_list, sub, TR, movie_len):
    betweenrun = []
    for run in run_list:
        withinrun = []
        events = pd.read_csv(f'temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv', delimiter='\t')
        timecourse = np.loadtxt(f'temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
        for movie in movie_list:
            onset_list = list(events.loc[events['trial_type'] == str(movie+'.mp4')]['onset'])
            withinrep = []
            if sub == 11:
                onset_list.pop(-1)
            for onset in onset_list:
                start = int(round(onset/TR)-1)
                stop = int(round(start + (movie_len/TR)))
                withinrep.append(timecourse[start:stop,:])
            withinrun.append(np.mean(np.array(withinrep),axis=0))
        betweenrun.append(np.array(withinrun))
    return np.mean(np.array(betweenrun),axis=0)




if __name__ == '__main__':
    sub_list = [2,3,4,5,6,7,8,9,10,11,12,13,14]
    movie_list = ['piper', 'bathsong', 'dog', 'forest', 'minions_supermarket', 'new_orleans']
    run_list= ['video_run-001','video_run-002']
    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    movie_len = 22.6
    TR = 0.656
    ROInum = 400

    timecourses_mean = []
    output_correlations = {}

    task_list = ['run_ISC', 'brain_render_file', 'plot_by_movie']

    for task in task_list:
        if task == 'run_ISC':
            for sub in sub_list:
                for run in run_list:
                    if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt'):
                        s3.download_file(bucket, f'foundcog-adult-pilot-2/volumetric_preprocessing/timecourses/sub-{sub:02d}/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt', f'sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
                        os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt ./temp')
                    if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv'):
                        s3.download_file(bucket, f'foundcog-adult-pilot-2/bids/sub-{sub:02d}/ses-001/func/sub-{sub:02d}_ses-001_task-{run}_events.tsv', f'sub-{sub:02d}_ses-001_task-{run}_events.tsv')
                        os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_events.tsv ./temp')
                timecourses_mean.append(average_2runs(movie_list, run_list, sub, TR, movie_len))

            for movie_ind,movie in enumerate(movie_list):
                allcorr = np.zeros((ROInum,len(sub_list)))
                for ROI in range(0,ROInum): 
                    for sub_ind,sub in enumerate(sub_list):
                        allsub = np.array(timecourses_mean)
                        curr_sub = allsub[sub_ind,movie_ind,:,ROI]
                        othersub = np.delete(allsub,sub_ind,axis=0)
                        othersub_average = np.mean(othersub[:,movie_ind,:,ROI],axis = 0)
                        corr,p = spearmanr(curr_sub, othersub_average)
                        if np.isnan(corr):
                            print(sub_ind,ROI)
                            print(curr_sub)
                            print(othersub_average)
                        allcorr[ROI,sub_ind] = corr
                output_correlations[movie] = allcorr
            with open('Results/ISC/ISC_allcorr.pickle', 'wb') as f:
                pickle.dump(output_correlations ,f)
        
        elif task == 'brain_render_file':
            atlas = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz')
            rois_data = atlas.get_fdata()
            with open('Results/ISC/ISC_allcorr.pickle','rb') as f:
                allcorr = pickle.load(f)
            for movie in movie_list:
                print(movie)
                corrdata = np.nanmean(allcorr[movie],axis=1)
                outvolume = np.zeros((182, 218, 182))
                for roi in range(0,400):
                    roi_index = np.where(rois_data==roi+1)
                    outvolume[roi_index] = corrdata[roi]
                    print('roi',roi)
                outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
                outname = (f'./Results/ISC/{movie}_corr_brainrender.nii.gz')
                nib.save(outimage,outname)


        #elif task == 'plot_by_movie':


    

