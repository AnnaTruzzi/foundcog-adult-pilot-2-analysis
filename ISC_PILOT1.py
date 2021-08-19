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


def average_2runs(movie_list, run_list, sub, TR, movie_len):
    betweenrun = []
    for run in run_list:
        withinrun = []
        events = pd.read_csv(f'temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv', delimiter='\t')
        timecourse = np.loadtxt(f'temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
        for movie in movie_list:
            onset_list = list(events.loc[events['trial_type'] == str(movie+'.mp4')]['onset'])
            withinrep = []
            if sub == 9:
                print(onset_list)
            for onset in onset_list:
                start = int(round(onset/TR)-1)
                stop = int(round(start + (movie_len/TR)))
                withinrep.append(timecourse[start:stop,:])
                print(timecourse[start:stop,:])
            withinrun.append(np.mean(np.array(withinrep),axis=0))
        betweenrun.append(np.array(withinrun))
    return np.mean(np.array(betweenrun),axis=0)


if __name__ == '__main__':
    sub_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    movie_list = ['moana','piper','bathsong', 'dog', 'forest', 'minions_supermarket', 'new_orleans']
    run_list= ['video_run-001','video_run-002']
    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    movie_len = 22.6
    TR = 0.656
    ROInum = 400

    timecourses_mean = []
    task_list = ['run_ISC','brain_render_file', 'plot_and_stats_ISC']  # Possible values : 'run_ISC', 'brain_render_file', 'plot_and_stats_ISC'

    for task in task_list:
        if task == 'run_ISC':
            for sub in sub_list:
                for run in run_list:
                    if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt'):
                        s3.download_file(bucket, f'volumetric_preprocessing/timecourses/sub-{sub:02d}/sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt', f'sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt')
                        os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_Schaefer400_timecourses.txt ./temp')
                    if not os.path.isfile(f'./temp/sub-{sub:02d}_ses-001_task-{run}_events.tsv'):
                        s3.download_file(bucket, f'bids/sub-{sub:02d}/ses-001/func/sub-{sub:02d}_ses-001_task-{run}_events.tsv', f'sub-{sub:02d}_ses-001_task-{run}_events.tsv')
                        os.system(f'mv sub-{sub:02d}_ses-001_task-{run}_events.tsv ./temp')
                timecourses_mean.append(average_2runs(movie_list, run_list, sub, TR, movie_len))
                with open('Results/Timecourses_average_across_runs_PILOT1.pickle','wb') as f:
                    pickle.dump(timecourses_mean,f)

            output_correlations = {}
            for movie_ind,movie in enumerate(movie_list):
                allcorr = np.zeros((ROInum,len(sub_list)))
                for ROI in range(0,ROInum): 
                    for sub_ind,sub in enumerate(sub_list):
                        allsub = np.array(timecourses_mean)
                        curr_sub = allsub[sub_ind,movie_ind,:,ROI]
                        othersub = np.delete(allsub,sub_ind,axis=0)
                        othersub_average = np.mean(othersub[:,movie_ind,:,ROI],axis = 0)
                        corr,p = spearmanr(curr_sub, othersub_average)
                        #if np.isnan(corr):
                            #print(sub_ind,ROI)
                            #print(curr_sub)
                            #print(othersub_average)
                        allcorr[ROI,sub_ind] = corr
                output_correlations[movie] = allcorr
            with open('Results/ISC/ISC_allcorr_PILOT1.pickle', 'wb') as f:
                pickle.dump(output_correlations ,f)

            allcorr_nulldist = []
            permutations = 10000
            print('working on permutations')
            for perm in range(0,permutations):
                ROI = random.choice(list(range(0,ROInum)))
                movie_ind = np.where(np.array(movie_list) == random.choice(movie_list))[0][0]
                movie_random = np.where(np.array(movie_list) == random.choice(movie_list))[0][0]
                print(movie_ind)
                print(movie_random)
                sub_ind = random.choice(list(range(0,len(sub_list))))
                allsub = np.array(timecourses_mean)
                curr_sub = allsub[sub_ind,movie_ind,:,ROI]
                othersub = np.delete(allsub,sub_ind,axis=0)
                othersub_average = np.nanmean(othersub[:,movie_random,:,ROI],axis = 0)
                corr,p = spearmanr(curr_sub, othersub_average)
                print(corr)
                allcorr_nulldist.append(corr)
            with open('Results/ISC/ISC_nulldist_PILOT1.pickle', 'wb') as f:
                pickle.dump(allcorr_nulldist ,f)

        elif task == 'brain_render_file':
            atlas = nib.load('Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz')
            rois_data = atlas.get_fdata()
            with open('Results/ISC/ISC_allcorr_PILOT1.pickle','rb') as f:
                allcorr = pickle.load(f)
            for movie in movie_list:
                if movie == 'moana':
                    print(movie)
                    corrdata = np.nanmean(allcorr[movie],axis=1)
                    outvolume = np.zeros((182, 218, 182))
                    for roi in range(0,400):
                        roi_index = np.where(rois_data==roi+1)
                        outvolume[roi_index] = corrdata[roi]
                        print('roi',roi)
                    outimage = nib.Nifti1Image(outvolume, affine=atlas.affine)
                    outname = (f'./Results/ISC/{movie}_corr_brainrender_PILOT1.nii.gz')
                    nib.save(outimage,outname)


        elif task == 'plot_and_stats_ISC':
            with open('Results/ISC/ISC_allcorr_PILOT1.pickle','rb') as f:
                allcorr = pickle.load(f)
            with open('Results/ISC/ISC_nulldist_PILOT1.pickle','rb') as f:
                nulldist = pickle.load(f)
            movie_names = []
            movie_corrs = []
            for movie in movie_list:
                name = np.repeat(movie, allcorr[movie].shape[0])
                corr = np.nanmean(np.array(allcorr[movie]),axis=1)
                movie_names.extend(name)
                movie_corrs.extend(corr)
            plot_dic = {'movie_name' : movie_names,
                        'movie_corr' : movie_corrs}
            plot_df = pd.DataFrame(plot_dic)
            sns.barplot(x = 'movie_name', y = 'movie_corr', data = plot_df)     
            plt.savefig('Results/ISC/ISC_barplot_bymovie_PILOT1.png')
            plt.close()
            
            with open('Results/ISC/ISC_stats_PILOT1.txt','w') as f:
                for movie in movie_list:
                    movie_dist = np.array(plot_df.loc[plot_df['movie_name'] == movie]['movie_corr'])
                    u, p = mannwhitneyu(np.array(nulldist),movie_dist)
                    print(movie)
                    print(u,p)
                    f.write(f'### {movie} vs null distribution - PILOT1### \n')
                    f.write(f'u = {u} - p = {round(p,2)} \n')
                    f.write('\n \n')

                    distplot_dict = {'type': np.concatenate((np.repeat(movie,len(movie_dist)),np.repeat('null',len(nulldist)))),
                                    'values': np.concatenate((movie_dist,np.array(nulldist)))}
                    distplot_df = pd.DataFrame(distplot_dict)
                    if movie == 'moana':
                        sns.displot(distplot_df, x = 'values', hue = 'type')
                        plt.savefig(f'Results/ISC/{movie}_and_null_distplot_PILOT1.png')
                        plt.close()
                f.close()