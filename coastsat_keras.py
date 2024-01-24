import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import geopandas as gpd
from datetime import datetime, timedelta
import scipy.io as sio
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from keras.backend import clear_session
import pyfes
import os

def read_shore_data_new(beach_id,site_id,shore_data_new,time_sla,sla_transect):
    
    if beach_id in shore_data_new.columns:
        
        fold = 'data_coastsat/shoreline_data/USA_CA/' + beach_id[0:11]
        data_file = fold + '/time_series_raw.csv'
        shore_data_ = pd.read_csv(data_file)
        shore_data = pd.DataFrame()
        shore_data['dates'] = shore_data_['dates']
        shore_data[beach_id] = shore_data_[beach_id]
        data_file = fold + '/tide_levels_fes2014.csv'
        tide_heights = pd.read_csv(data_file)
        tide_heights['dates'] = pd.to_datetime(tide_heights['dates'])

        shore_data_time = pd.to_datetime(shore_data['dates'])
        median_2021 = np.nanmedian(shore_data[beach_id][shore_data_time.dt.year==2021])

        shore_data_new_ = pd.DataFrame()
        idxnan = ~np.isnan(shore_data_new[beach_id])
        shore_data_new_['dates'] = shore_data_new['dates'][idxnan]
        shore_data_new_[beach_id] = shore_data_new[beach_id][idxnan]
        shore_data_new_.reset_index(drop=True, inplace=True)
        
        shore_data_new_time = pd.to_datetime(shore_data_new_['dates'])
        
        if shore_data_new_time.empty:
            test = 0
        else:
            median_2021_new = np.nanmedian(shore_data_new_[beach_id][shore_data_new_time.dt.year==2021])
            shore_data_new_[beach_id] = shore_data_new_[beach_id] - median_2021_new + median_2021

            shore_data = pd.concat([shore_data,shore_data_new_])

            filepath = r'/Users/suadusumilli/sioglaciology Dropbox/Susheel Adusumilli/data/fes2014'
            config_ocean = os.path.join(filepath, 'ocean_tide.ini') # change to ocean_tide.ini
            config_load =  os.path.join(filepath, 'load_tide.ini')  # change to load_tide.ini
            ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
            load_tide = pyfes.Handler("radial", "io", config_load)
            pts = transect_data[transect_data['id'].str.contains(beach_id)]['points'].to_numpy()
            for pts_ in pts:
                llon_1 = pts_[0]
                llon_2 = pts_[1]
            coords = [llon_2[0],llon_2[1]]
            # get tide time-series with 15 minutes intervals
            tide_heights_new = pd.DataFrame()
            tide_heights_new['tide levels'] = SDS_slope.compute_tide_dates(coords, shore_data_new_time, ocean_tide, load_tide)
            tide_heights_new['dates'] = shore_data_new_time
            tide_heights = pd.concat([tide_heights,tide_heights_new])

        wave_data = pd.DataFrame()
        ind_mop = str(np.squeeze(np.argwhere(site_id==beach_id)) + 1)
        mop_fold = '/Volumes/nox2/MOP_coastsat/MOP_coastsat_'
        mop_data = sio.loadmat(mop_fold + ind_mop + '.mat')    

        dist_mop = np.squeeze(mop_data['dist_mop'])
        mop_time = np.squeeze(mop_data['time_mop'])
        spec1d = np.squeeze(mop_data['spec1d'])
        spec1d[np.isnan(spec1d)] = 0.0
        spec1d[spec1d>1.e3] = 0.0
        Hs = np.squeeze(mop_data['Hs'])
        Tp = np.squeeze(mop_data['Tp'])
        freqz = np.squeeze(mop_data['freqz'])
        Hs[Hs>1.e2] = np.NaN
        Tp[Tp>1.e5] = np.NaN
        mop_time = pd.to_datetime(mop_time-719529., unit='D')
        Hs = np.interp(tide_heights['dates'],mop_time,Hs)
        Tp = np.interp(tide_heights['dates'],mop_time,Tp)
        
        time_sla_ = pd.to_datetime(time_sla-719529., unit='D')
        sla_ = np.interp(tide_heights['dates'],time_sla_,np.squeeze(sla_transect[np.argwhere(site_id==beach_id),:]))
        tide_heights['tide levels'] = tide_heights['tide levels'] + sla_
        
        spec1d_interp = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        
        for i in range(20):
            spec1d_interp[:,i] = np.interp(tide_heights['dates'],mop_time,spec1d[:,i])
            
        spec1d_interp_lag_14d = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_1mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_3mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_6mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
        spec1d_interp_lag_1yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
        spec1d_interp_lag_2yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))   
        
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 14*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_14d[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_1mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 3*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_3mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 6*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_6mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 12*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_1yr[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 2*12*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_2yr[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        wave_data['dates'] = shore_data['dates']
        wave_data['Tp'] = Tp
        wave_data['Hs'] = Hs

        shore_data.reset_index(drop=True, inplace=True)
        tide_heights.reset_index(drop=True, inplace=True)
        wave_data.reset_index(drop=True, inplace=True)
        return shore_data,tide_heights,wave_data,dist_mop,spec1d_interp,spec1d_interp_lag_14d,spec1d_interp_lag_1mo,spec1d_interp_lag_3mo,spec1d_interp_lag_6mo,spec1d_interp_lag_1yr,spec1d_interp_lag_2yr,freqz
    else:
        fold = 'data_coastsat/shoreline_data/USA_CA/' + beach_id[0:11]
        data_file = fold + '/time_series_raw.csv'
        shore_data_ = pd.read_csv(data_file)
        shore_data = pd.DataFrame()
        shore_data['dates'] = shore_data_['dates']
        shore_data[beach_id] = shore_data_[beach_id]
        data_file = fold + '/tide_levels_fes2014.csv'
        tide_heights = pd.read_csv(data_file)
        tide_heights['dates'] = pd.to_datetime(tide_heights['dates'])

        shore_data_time = pd.to_datetime(shore_data['dates'])
        median_2021 = np.nanmedian(shore_data[beach_id][shore_data_time.dt.year==2021])
        
        wave_data = pd.DataFrame()
        ind_mop = str(np.squeeze(np.argwhere(site_id==beach_id)) + 1)
        mop_fold = '/Volumes/nox2/MOP_coastsat/MOP_coastsat_'
        mop_data = sio.loadmat(mop_fold + ind_mop + '.mat')    

        dist_mop = np.squeeze(mop_data['dist_mop'])
        mop_time = np.squeeze(mop_data['time_mop'])
        spec1d = np.squeeze(mop_data['spec1d'])
        spec1d[spec1d>1.e3] = 0.0
        spec1d[np.isnan(spec1d)] = 0.0
        Hs = np.squeeze(mop_data['Hs'])
        Tp = np.squeeze(mop_data['Tp'])
        freqz = np.squeeze(mop_data['freqz'])
        Hs[Hs>1.e2] = np.NaN
        Tp[Tp>1.e5] = np.NaN
        mop_time = pd.to_datetime(mop_time-719529., unit='D')
        Hs = np.interp(tide_heights['dates'],mop_time,Hs)
        Tp = np.interp(tide_heights['dates'],mop_time,Tp)
        
        time_sla_ = pd.to_datetime(time_sla-719529., unit='D')
        sla_ = np.interp(tide_heights['dates'],time_sla_,np.squeeze(sla_transect[np.argwhere(site_id==beach_id),:]))
        tide_heights['tide levels'] = tide_heights['tide levels'] + sla_

        spec1d_interp = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        
        for i in range(20):
            spec1d_interp[:,i] = np.interp(tide_heights['dates'],mop_time,spec1d[:,i])
            
        spec1d_interp_lag_14d = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_1mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_3mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
        spec1d_interp_lag_6mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
        spec1d_interp_lag_1yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
        spec1d_interp_lag_2yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))   
        
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 14*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_14d[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_1mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 3*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_3mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 6*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_6mo[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)
            
        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 12*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_1yr[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        for i in range(20):
            spec_1d_lag_ = spec1d[:,i]
            window_size = 2*12*30*24
            cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
            moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
            # Padding with NaNs at the beginning for the first 14 days
            padded_avg = np.full(spec_1d_lag_.shape, np.nan)
            padded_avg[window_size - 1:] = moving_avg
            spec1d_interp_lag_2yr[:,i] = np.interp(tide_heights['dates'],mop_time,padded_avg)

        wave_data['dates'] = shore_data['dates']
        wave_data['Tp'] = Tp
        wave_data['Hs'] = Hs

        shore_data.reset_index(drop=True, inplace=True)
        tide_heights.reset_index(drop=True, inplace=True)
        wave_data.reset_index(drop=True, inplace=True)
        return shore_data,tide_heights,wave_data,dist_mop,spec1d_interp,spec1d_interp_lag_14d,spec1d_interp_lag_1mo,spec1d_interp_lag_3mo,spec1d_interp_lag_6mo,spec1d_interp_lag_1yr,spec1d_interp_lag_2yr,freqz

def read_keras_model(beach_id,site_id):

    ind_keras = str(np.squeeze(np.argwhere(site_id==beach_id)))
    keras_fold = '/Users/suadusumilli/sioglaciology Dropbox/Susheel Adusumilli/code/coastsat/coastsat_keras_adam_mse/keras_hmodel_'
    model = load_model(keras_fold + ind_keras + '.h5')    
   
    return model

def create_df_for_keras(shore_data,tide_heights,wave_data,spec1d_interp,spec1d_interp_lag_14d,spec1d_interp_lag_1mo,spec1d_interp_lag_3mo,spec1d_interp_lag_6mo,spec1d_interp_lag_1yr,spec1d_interp_lag_2yr,freqz,beach_id):
    d1 = pytz.utc.localize(datetime(2002,1,1))
    d2 = pytz.utc.localize(datetime(2023,4,1))
    idx_dates = [np.logical_and(_>d1,_<d2) for _ in tide_heights['dates']]
    shore_data[beach_id] = shore_data[beach_id][idx_dates]
    for key in tide_heights.keys():
        tide_heights[key] = tide_heights[key][idx_dates]
    for key in wave_data.keys():
        wave_data[key] = wave_data[key][idx_dates]

    idx_nan = np.logical_or(np.isnan(wave_data['Hs']),np.isnan(wave_data['Tp']*shore_data[beach_id])) 

    df = pd.DataFrame()
    df['tide'] = tide_heights['tide levels'][~idx_nan].to_numpy()
    timestamp_dt = pd.to_datetime(wave_data['dates']) - datetime(2000,1,1,tzinfo=pytz.UTC)
    timestamp_s = timestamp_dt.dt.total_seconds()[~idx_nan].to_numpy()
    day = 24.*60*60
    year = (365.2425)*day
    # df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    # df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # df['Hs'] = wave_data['Hs'][~idx_nan].to_numpy()
    # df['Tp'] = wave_data['Tp'][~idx_nan].to_numpy()

    m_num = 5
    n_num = 5
    m = np.linspace(0.2,2,m_num)
    n = np.linspace(-4,4,n_num)
    
    e_ipa = np.empty((np.shape(spec1d_interp)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('A_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_14d)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_14d,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('B_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_1mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_1mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('C_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_3mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_3mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('D_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_6mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_6mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('E_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_1yr)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_1yr,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('F_')
    df = pd.concat([df, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_interp_lag_2yr)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_interp_lag_2yr,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa[~idx_nan,:]))
    df__ = df__.add_prefix('G_')
    df = pd.concat([df, df__], axis=1)

    df['shore_change'] = shore_data[beach_id][~idx_nan].to_numpy()
    df_time = pd.to_datetime(shore_data['dates'][~idx_nan].to_numpy())
    return df, df_time

def create_proj_data(beach_id,site_id):
    
    df_proj = pd.DataFrame()
    
    ind_mop = str(np.squeeze(np.argwhere(site_id==beach_id)) + 1)
    mop_fold = '/Volumes/nox2/MOP_coastsat/MOP_coastsat_'
    mop_data = sio.loadmat(mop_fold + ind_mop + '.mat')    
    
    dist_mop = np.squeeze(mop_data['dist_mop'])
    mop_time = np.squeeze(mop_data['time_mop'])
    spec1d = np.squeeze(mop_data['spec1d'])
    spec1d[spec1d>1.e3] = 0.0
    spec1d[np.isnan(spec1d)] = 0.0
    
    Hs = np.squeeze(mop_data['Hs'])
    Tp = np.squeeze(mop_data['Tp'])
    freqz = np.squeeze(mop_data['freqz'])
    Hs[Hs>1.e2] = np.NaN
    Tp[Tp>1.e5] = np.NaN
    mop_time = pd.to_datetime(mop_time-719529., unit='D')
    
    filepath = r'/Users/suadusumilli/sioglaciology Dropbox/Susheel Adusumilli/data/fes2014'
    config_ocean = os.path.join(filepath, 'ocean_tide.ini') # change to ocean_tide.ini
    config_load =  os.path.join(filepath, 'load_tide.ini')  # change to load_tide.ini
    ocean_tide = pyfes.Handler("ocean", "io", config_ocean)
    load_tide = pyfes.Handler("radial", "io", config_load)
    pts = transect_data[transect_data['id'].str.contains(beach_id)]['points'].to_numpy()
    for pts_ in pts:
        llon_1 = pts_[0]
        llon_2 = pts_[1]
    coords = [llon_2[0],llon_2[1]]
    # get tide time-series with 15 minutes intervals
    df_proj['tide'] = SDS_slope.compute_tide_dates(coords, mop_time, ocean_tide, load_tide)
    timestamp_s = mop_time - datetime(2000,1,1)
    timestamp_s = np.array(timestamp_s.total_seconds())
    day = 24.*60*60
    year = (365.2425)*day

    spec1d_lag_14d = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
    spec1d_lag_1mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
    spec1d_lag_3mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))
    spec1d_lag_6mo = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
    spec1d_lag_1yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))       
    spec1d_lag_2yr = np.empty((np.shape(Hs)[0],np.shape(spec1d)[1]))   

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 14*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_14d[:,i] = padded_avg

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 30*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_1mo[:,i] = padded_avg

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 3*30*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_3mo[:,i] = padded_avg

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 6*30*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_6mo[:,i] = padded_avg

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 12*30*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_1yr[:,i] = padded_avg

    for i in range(20):
        spec_1d_lag_ = spec1d[:,i]
        window_size = 2*12*30*24
        cumsum_vec = np.cumsum(np.insert(spec_1d_lag_, 0, 0))
        moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        # Padding with NaNs at the beginning for the first 14 days
        padded_avg = np.full(spec_1d_lag_.shape, np.nan)
        padded_avg[window_size - 1:] = moving_avg
        spec1d_lag_2yr[:,i] = padded_avg
    
    m_num = 5
    n_num = 5
    m = np.linspace(0.2,2,m_num)
    n = np.linspace(-4,4,n_num)
    
    e_ipa = np.empty((np.shape(spec1d)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('A_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_14d)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_14d,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('B_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_1mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_1mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('C_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_3mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_3mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('D_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_6mo)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_6mo,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('E_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_1yr)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_1yr,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('F_')
    df_proj = pd.concat([df_proj, df__], axis=1)
    
    e_ipa = np.empty((np.shape(spec1d_lag_2yr)[0],m_num*n_num))
    mn_ = 0
    for m_ in m:
        for n_ in n:
            e_ipa_ = np.power(spec1d_lag_2yr,m_)*np.power(freqz,n_)
            e_ipa[:,mn_] = np.sum(e_ipa_,axis=1)
            mn_+=1
    df__ = pd.DataFrame(np.sqrt(e_ipa))
    df__ = df__.add_prefix('G_')
    df_proj = pd.concat([df_proj, df__], axis=1)
            
    df_proj['shore_change'] = df_proj['tide']*0
    
    return df_proj,mop_time

def create_train_val_test_data(df,df_proj):
    # df = df.diff()[1:]
    n = len(df)
    
    df_mean = df.mean()
    df_std = df.std()
    
    df = (df-df_mean)/df_std
    
    train_df, remaining_data = train_test_split(df, test_size=0.2, random_state=42)

    # Splitting the remaining data into test and validation
    test_df, val_df = train_test_split(remaining_data, test_size=0.5, random_state=42)
    
    df_proj_scaled = (df_proj-df_mean)/df_std
    df_proj_scaled['shore_change'] = df_proj_scaled['shore_change']*0.
    
    X_all, y_all = df.values[:, :-1], df.values[:, -1]
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]
    X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]
    X_proj = df_proj_scaled.values[:, :-1]
    
    return X_all,X_train,X_val,X_test,y_all,y_train,y_val,y_test,df_mean,df_std,X_proj

def create_train_val_test_data_sequential(df,df_proj):
    # df = df.diff()[1:]
    n = len(df)
    
    df_mean = df.mean()
    df_std = df.std()
    
    df = (df-df_mean)/df_std
    
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.8)]
    test_df = df[int(n*0.8):]
    
    df_proj_scaled = (df_proj-df_mean)/df_std
    df_proj_scaled['shore_change'] = df_proj_scaled['shore_change']*0.
    
    X_all, y_all = df.values[:, :-1], df.values[:, -1]
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]
    X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]
    X_proj = df_proj_scaled.values[:, :-1]
    
    return X_all,X_train,X_val,X_test,y_all,y_train,y_val,y_test,df_mean,df_std,X_proj

def create_train_val_test_data_no_proj(df):
    n = len(df)
    
    df_mean = df.mean()
    df_std = df.std()
    
    df = (df-df_mean)/df_std
    
    train_df, remaining_data = train_test_split(df, test_size=0.2, random_state=42)

    # Splitting the remaining data into test and validation
    test_df, val_df = train_test_split(remaining_data, test_size=0.5, random_state=42)
    
    X_all, y_all = df.values[:, :-1], df.values[:, -1]
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]
    X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]
    
    return X_all,X_train,X_val,X_test,y_all,y_train,y_val,y_test,df_mean,df_std

def create_train_val_test_data_no_proj_sequential(df):
    n = len(df)
    
    df_mean = df.mean()
    df_std = df.std()
    
    df = (df-df_mean)/df_std
    
    train_df = df[0:int(n*0.8)]
    test_df = df[int(n*0.8):int(n*0.9)]
    val_df = df[int(n*0.9):]
    train_df, remaining_data = train_test_split(df, test_size=0.2, random_state=42)

    # Splitting the remaining data into test and validation
    test_df, val_df = train_test_split(remaining_data, test_size=0.5, random_state=42)
    
    X_all, y_all = df.values[:, :-1], df.values[:, -1]
    X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
    X_val, y_val = val_df.values[:, :-1], val_df.values[:, -1]
    X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]
    
    return X_all,X_train,X_val,X_test,y_all,y_train,y_val,y_test,df_mean,df_std

def rmse_calc(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
