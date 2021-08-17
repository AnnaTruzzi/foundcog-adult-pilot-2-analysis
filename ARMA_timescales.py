import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import pickle
import pandas as pd
import glob
import os
import random
import boto3


def calculate_ACF(timecourse,nlags):
    xc=timecourse-np.mean(timecourse)
    fullcorr=np.correlate(xc, xc, mode='full')
    fullcorr=fullcorr / np.max(fullcorr)
    start=len(fullcorr) // 2
    stop=start+nlags
    return fullcorr[start:stop]

def calculate_tau(timecourse, order):
    mod = ARIMA(endog=ts_df[9:,ROI], order=order, enforce_stationarity=False)
    res=mod.fit()
    ar=res.arparams
    rho0 = 1
    try:
        rho1 = ar[0]
        rho1 = rho1
        tau=-1/np.log(rho1)
    except:
        rho1 = np.nan
        tau= np.nan
    return rho1,tau


if __name__ == '__main__':
    sublist = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    s3=boto3.client('s3')
    bucket='foundcog-adult-pilot'
    tasks_list = ['resting_run-001']
    nlags = 10

    for task in tasks_list:
        alltau = np.zeros((len(sublist),400))
        allrho1 = np.zeros((len(sublist),400))

        for i,sub in enumerate(sublist):
            s3.download_file(bucket, f'foundcog-adult-pilot-2/volumetric_preprocessing/timecourses/sub-{sub:02d}/sub-{sub:02d}_ses-001_task-{task}_Schaefer400_timecourses.txt', f'sub-{sub:02d}_ses-001_task-{task}_Schaefer400_timecourses.txt')
            os.system(f'mv sub-{sub:02d}_ses-001_task-{task}_Schaefer400_timecourses.txt ./temp')
            ts_df = np.loadtxt(os.path.join(f'./temp/sub-{sub:02d}_ses-001_task-{task}_Schaefer400_timecourses.txt'))
            timescale = np.zeros((ts_df.shape[1], nlags))
            for ROI in range(0,ts_df.shape[1]):
                    timecourse=ts_df[9:,ROI]
                    timescale[ROI,:]=calculate_ACF(timecourse,nlags)
            plt.plot(timescale.T, color='blue', alpha=0.01)
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')
            plt.suptitle(f'Sub-{sub:02d}')
            plt.savefig(f'./Results/intrinsic_timescales/ACF_sub-{sub:02d}.png')
            plt.show()
            plt.close()

            
            for ROI in range(0,ts_df.shape[1]):
                timecourse=ts_df[9:,ROI]
                order = (1,0,1)
                allrho1[i,ROI], alltau[i,ROI] = calculate_tau(timecourse,order)


        tau_df=pd.DataFrame(data=alltau[0:,0:],
                index=[i for i in range(alltau.shape[0])],
                columns=[i for i in range(alltau.shape[1])])
        tau_df.insert(loc=0, column='subj', value=sublist)

        allrho1_df=pd.DataFrame(data=allrho1[0:,0:],
                index=[i for i in range(allrho1.shape[0])],
                columns=[i for i in range(allrho1.shape[1])])
        allrho1_df.insert(loc=0, column='subj', value=sublist)

        tau_df.to_csv(f'./Results/intrinsic_timescales/estimatedtau_foundcog-pilot-2_{task}.csv')
        np.savetxt(f'./Results/intrinsic_timescales/estimatedtau_foundcog-pilot-2_{task}.txt',alltau)

        allrho1_df.to_csv(f'./Results/intrinsic_timescales/estimatedrho1_foundcog-pilot-2_{task}.csv')
        np.savetxt(f'./Results/intrinsic_timescales/estimatedrho1_foundcog-pilot-2_{task}.txt',allrho1)



