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
from scipy.stats import mannwhitneyu


def find_couples(movie_list,orders):
    couples_tot = {}
    for movie in movie_list:
        couples_per_movie = []
        for order in orders:
            movie_idx = np.where(order == movie)
            precedent = movie_idx - 1
            couples_per_movie.append([movie_list[precedent],movie_list[movie_idx]])
        couples_tot[movie] = couples_per_movie
    return couples_tot


if __name__ == '__main__':
    sub_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    movie_list = ['piper', 'bathsong', 'dog', 'forest', 'minions_supermarket', 'new_orleans']
    run_list= ['video_run-001','video_run-002']
    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    movie_len = 22.6
    TR = 0.656
    ROInum = 400

    orders =[['bathsong.mp4', 'dog.mp4', 'new_orleans.mp4', 'minions_supermarket.mp4', 'forest.mp4', 'piper.mp4'], 
        ['minions_supermarket.mp4', 'piper.mp4', 'forest.mp4', 'dog.mp4', 'bathsong.mp4', 'new_orleans.mp4'], 
        ['forest.mp4', 'minions_supermarket.mp4', 'piper.mp4', 'bathsong.mp4', 'new_orleans.mp4', 'dog.mp4'], 
        ['dog.mp4', 'new_orleans.mp4', 'bathsong.mp4', 'piper.mp4', 'minions_supermarket.mp4', 'forest.mp4'], 
        ['new_orleans.mp4', 'bathsong.mp4', 'dog.mp4', 'forest.mp4', 'piper.mp4', 'minions_supermarket.mp4'], 
        ['piper.mp4', 'forest.mp4', 'minions_supermarket.mp4', 'new_orleans.mp4', 'dog.mp4', 'bathsong.mp4']
    ]

    network_file = pd.read_csv('/Data/Schaefer2018_400Parcels_17Networks_order.txt',sep = '\t', header = None)
    atlas = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz')
    roi_net_list = np.array([i.split("_")[2] for i in network_file[1]])
    network_names = list(set(roi_net_list))
    windows = np.arange(0,35,5)
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
    

    timecourses_mean = []
    for sub in sub_list:
        for run in run_list:
            if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt'):
                s3.download_file(bucket, f'foundcog-adult-pilot-2/volumetric_preprocessing/timecourses/sub-{sub:02d}/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt', f'sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
                os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt ./temp')
            if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv'):
                s3.download_file(bucket, f'foundcog-adult-pilot-2/bids/sub-{sub:02d}/ses-001/func/sub-{sub:02d}_ses-001_task-{run}_events.tsv', f'sub-{sub:02d}_ses-001_task-{run}_events.tsv')
                os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_events.tsv ./temp')

    couples = find_couples(movie_list,orders)
