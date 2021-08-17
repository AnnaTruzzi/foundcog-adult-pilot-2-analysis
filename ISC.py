import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import random
import boto3

def average_runs(movie_list, run_list, sub, TR, movie_len):
    for run in run_list:
        events = pd.read_csv(f'temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv')
        timecourse = np.loadtxt(f'temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
        for movie in movie_list:
            onset_list = events.loc[events['trial_type'] == str(movie+'.mp4')]['onset']
            withinrun = []
            for onset in onset_list:
                start = onset/TR
                stop = start + (movie_len/TR)
                withinrun.append(timecourse[start:stop,:])
            betweenrun.append(np.mean(np.array(withinrun),axis=0))
            betweenrun_average =     



if __name__ == '__main__':
    sub_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    movie_list = ['piper', 'bathsong', 'dog', 'forest', 'minions_supermarket', 'new_orleans']
    run_list= ['video_run-001','video_run-002']
    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    movie_len = 22.6
    TR = 0.656

    timecourses_list = []

    for sub in sub_list:
        for run in run_list:
            if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt'):
                s3.download(bucket, f'foundcog-adult-pilot-2/volumetric_preprocessing/timecourses/sub-{sub:02d}/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt', f'sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
                os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt ./temp')
            if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv'):
                s3.download(bucket, f'foundcog-adult-pilot-2/bids/sub-{sub:02d}/ses-001/func//sub-{sub:02d}_ses-001_task-{run}_events.tsv', f'sub-{sub:02d}_ses-001_task-{run}_events.tsv')
                os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_events.tsv ./temp')
        mean_timecourse = average_runs(movie_list, run_list, sub, TR, movie_len)

