#!/usr/bin/env python
# coding: utf-8

import pathlib
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

lof = []
def list_of_files(path, shuffle=True):
    lof =  sorted(str(p) for p in pathlib.Path(path).glob("*.csv"))
    if shuffle:
        random.shuffle(lof)
    return lof

np_list=[]
def list_to_array(input_list):
    for elem in input_list:
        df = pd.read_csv(elem, header=None).fillna(0)
        np_array = df.to_numpy()
        np_list.append(np.nan_to_num(np_array))
    return np_list

zmn_list=[]
def list_to_zmn(input_list):
    for elem in input_list:
        zmn_data = (elem - elem.mean(axis=0))/elem.std(axis=0)
        zmn_list.append(np.nan_to_num(zmn_data))
    return zmn_list
        
def interpolate_signal(insignal, len1, len2):
    x1 = np.linspace(0, len2-1, len1)
    f = interp1d(x1, insignal, axis=0, kind='linear')
    x2 = np.linspace(0, len2-1, len2)
    return f(x2)
    
inter_list = []
def list_to_interpolate(input_list, output_size=365):
    for elem in input_list:
        np_int = interpolate_signal(elem, elem.shape[0], output_size)
        inter_list.append(np.nan_to_num(np_int))
    return inter_list

def mov_avg(values, window):
    smas_list_temp = []
    weights = np.repeat(1.0, window)/window
    r, c = values.shape
    for i in range(c):
        smas_temp = np.convolve(values[:, i], weights, 'valid')
        smas_list_temp.append(smas_temp)
    smas = np.array(smas_list_temp)
    return smas.T

smas_list = []
def list_to_smas(input_list, window=3):
    for elem in input_list:
        smas = mov_avg(elem, window)
        smas_list.append(smas)
    return smas_list

def denoised_signal(insignal, fs, fc, filters):
    axis = np.where(insignal.shape == np.max(insignal.shape))[0][0]
    w = fc / (fs / 2)
    b1, b2 = signal.butter(filters, w, 'low')
    return signal.filtfilt(b1, b2, insignal, axis=axis)

denoised_list = []
def list_to_denoise(input_list, fs=100, fc=10, filters=4):
    for elem in input_list:
        denoised = denoised_signal(elem, fs, fc, filters)
        denoised_list.append(np.nan_to_num(denoised))
    return denoised_list

dP0_list=[]
def ref_to_dP0(ref_list):
    for elem in ref_list:
        df = pd.read_csv(elem, usecols=range(0,315), header=None).fillna(0)
        P = df.to_numpy()
        num_it = len(P)
        num_cols_P = P[0].size
        num_cols_d = int(num_cols_P / 3.)
        d = np.zeros((num_it-1, num_cols_d))
        pa = np.zeros((0,3))
        for j in range(num_cols_d):
            for i in range(num_it-1):
                s = 3*j
                e = 3*j+3
                pa = P[i, s:e]
                p0 = P[0,s:e]
                d[i][j] = np.linalg.norm(pa - p0)
        dP0_list.append(d)
    return dP0_list
    
PCA_list = []
def ref_to_PCA(ref_list):
    for elem in ref_list:
        df = pd.read_csv(elem, usecols=range(0,315), header=None).fillna(0)
        scaler = StandardScaler()
        ScaleDf = scaler.fit_transform(df)
        num_it = len(ScaleDf)
        num_cols = 315
        num_cols_d = int(num_cols / 3)
        d = np.zeros((len(ScaleDf), num_cols_d))
        newDF = pd.DataFrame(d)
        for j in range(num_cols_d):
            s = 3*j
            e = 3*j+3
            p = ScaleDf[:, s:e]
            pca=PCA(n_components=1)
            model = pca.fit_transform(p)
            model = np.reshape(model, (num_it,1))
            mo = pd.DataFrame(model)
            newDF.iloc[:, j] = mo
            data = newDF.to_numpy()
        PCA_list.append(data)
    return PCA_list

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

