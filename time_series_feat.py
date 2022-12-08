#Wrote by Subham Swastik Samal

#this code contains various functions required in feature extraction

import numpy
import scipy
from statsmodels.robust import mad as median_deviation


#fast fourier transformation
def fast_fourier_transform(df):
    complex_f = scipy.fftpack.fft(df, axis=0)
    amplitude_f = numpy.abs(complex_f)
    return amplitude_f

def col_means(df): #mean
    return list(numpy.mean(df,axis =0))

def col_range(df): #range
    return list(numpy.max(df, axis=0) - numpy.min(df, axis=0))

def stdev(df): #st_dev
    return list(numpy.std(df, axis=0))

def med_dev(df):
    return list(median_deviation(df, axis=0))  # calculate the median absolute deviation value of each column

def col_max(df): #max
    return list(numpy.max(df, axis=0))

def col_min(df): #min
    return list(numpy.min(df, axis=0))

def col_harmonic_mean(df): #harmonic mean
    divide_v = numpy.zeros(df.shape[1])
    for i in range(df.shape[1]):
        for j, v in enumerate(df[:, i]):
            if v != 0:
                divide_v[i] += 1 / v
    result = []
    for i in range(df.shape[1]):
        if divide_v[i] != 0:
            result.append(df.shape[0] / divide_v[i])
        else:
            result.append(0)  # NAN replaced by 0
    return result

def col_waveform_abs_length(df): #waveform absolute length
    result = numpy.zeros(df.shape[1])
    for i in range(1, df.shape[0]):
        result += numpy.abs(df[i, :] - df[i-1, :])
    return list(result)

def col_iq_range(df):   #inter-quartile range
    return list(scipy.stats.iqr(df, axis=0))

def col_sum_absolute(df):
    return list(numpy.sum(abs(df), axis=0))

def col_mean_energy(df):    #mean energy
    return list(numpy.sum(numpy.power(df, 2), axis=0) / len(df))

def col_skewness(df):       #skewness
    return list(scipy.stats.skew(df, axis=0))

def col_kurtosis(df):   #kurtosis
    return list(scipy.stats.kurtosis(df, axis=0))

def col_max_freq(df, sample_freq):  #max frequency
    if len(df.shape) == 1:
        n_col = 1
    else:
        n_col = df.shape[1]
    results = []
    freqs = scipy.fftpack.fftfreq(len(df), d=1 / float(sample_freq))
    for i in range(n_col):
        results.append(freqs[df[:, i].argmax()])
    return results

def col_f_mean_freq(df, sample_freq):   #mean frequency
    freqs = scipy.fftpack.fftfreq(len(df), d=1 / float(sample_freq))

    f_mean_freq = list(numpy.sum(df * numpy.array(freqs).reshape((-1, 1)), axis=0) / numpy.sum(df, axis=0))

    # NAN is replaced by 0
    for i in range(len(f_mean_freq)):
        if numpy.isnan(f_mean_freq[i]):
            f_mean_freq[i] = 0

    return f_mean_freq


def time_features(df):
    df = df.to_numpy()
    t_mean = col_means(df)
    t_range = col_range(df)
    t_stdev = stdev(df)
    t_mad = med_dev(df)
    t_max = col_max(df)
    t_min = col_min(df)
    t_iqr = col_iq_range(df)
    t_abs_sum = col_sum_absolute(df)
    t_mean_energy = col_mean_energy(df)
    t_skew = col_skewness(df)
    t_kurt = col_kurtosis(df)
    t_wave_len = col_waveform_abs_length(df)
    t_har_mean = col_harmonic_mean(df)

    t_features_vector = t_mean + t_range + t_stdev + t_mad + t_max + t_min + \
                        t_iqr + t_abs_sum + t_mean_energy + t_skew + t_kurt + t_wave_len + t_har_mean

    return t_features_vector


def time_feat_labels():
    T_features = ['Mean', 'Range', 'St_Dev', 'Mean_Abs_Dev', 'Max', 'Min',
                  'Int_Quar_Range', 'Abs_sum', 'Mean_energy', 'Skewness', 'Kurtosis',
                  'Abs_wavelength','Harmonic_Mean']
    Location = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc16', 'Acc', 'Gyro', 'Mag']
    Axis = ['X', 'Y', 'Z']

    features_names = []
    label = 't_'
    for feat in T_features:
        hb_feat = label+feat+'heartbeat'
        features_names.append(hb_feat)
        for loc in Location:
            temp_feat = label + feat + loc + 'temp'
            features_names.append(temp_feat)
            for se in Sensor:
                for ax in Axis:
                    feature_name = label + feat + loc + se + ax
                    features_names.append(feature_name)
    return features_names

def time_mag_labels():
    T_features = ['Mean', 'Range', 'St_Dev', 'Mean_Abs_Dev', 'Max', 'Min',
                  'Int_Quar_Range', 'Abs_sum', 'Mean_energy', 'Skewness', 'Kurtosis',
                  'Abs_wavelength','Harmonic_Mean']

    loc2 = ['Ankle','Chest','Hand']
    sens2 = ['Acc6', 'Acc16']
    features_names = []
    label = 't_'
    for feat in T_features:
        for loc in loc2:
            for se in sens2:
                feature_name = label + feat + loc + se
                features_names.append(feature_name)
    return features_names


def freq_features(df, obs_freq):
    f_mean = col_means(df)
    f_range = col_range(df)
    f_stdev = stdev(df)
    f_mad = med_dev(df)
    f_max = col_max(df)
    f_min = col_min(df)
    f_iqr = col_iq_range(df)
    f_abs_sum = col_sum_absolute(df)
    f_mean_energy = col_mean_energy(df)
    f_skew = col_skewness(df)
    f_kurt = col_kurtosis(df)
    f_mean_freq = col_f_mean_freq(df,obs_freq)
    f_har_mean = col_harmonic_mean(df)
    f_wave_len = col_waveform_abs_length(df)
    f_max_freq = col_max_freq(df, obs_freq)

    f_features_vector = f_mean + f_range + f_stdev + f_mad + f_max + f_min + \
                        f_iqr + f_abs_sum + f_mean_energy + f_skew + f_kurt + f_mean_freq + f_har_mean + f_wave_len + \
                        f_max_freq

    return f_features_vector

def freq_feat_labels():
    F_features = ['Mean', 'Range', 'St_Dev', 'Mean_Abs_Dev', 'Max', 'Min',
                  'Int_Quar_Range', 'Abs_sum', 'Mean_energy', 'Skewness', 'Kurtosis','Mean_freq',
                  'Harmonic_Mean','Abs_wavelength','Max_Freq']
    Location = ['Hand', 'Chest', 'Ankle']
    Sensor = ['Acc16', 'Acc', 'Gyro', 'Mag']
    Axis = ['X', 'Y', 'Z']

    features_names = []
    label = 'f_'
    for feat in F_features:
        hb_feat = label+feat+'heartbeat'
        features_names.append(hb_feat)
        for loc in Location:
            temp_feat = label + feat + loc + 'temp'
            features_names.append(temp_feat)
            for se in Sensor:
                for ax in Axis:
                    feature_name = label + feat + loc + se + ax
                    features_names.append(feature_name)
    return features_names


def freq_mag_labels():
    F_features = ['Mean', 'Range', 'St_Dev', 'Mean_Abs_Dev', 'Max', 'Min',
                  'Int_Quar_Range', 'Abs_sum', 'Mean_energy', 'Skewness', 'Kurtosis', 'Mean_freq',
                  'Harmonic_Mean','Abs_wavelength','Max_Freq']
    loc2 = ['Ankle', 'Chest', 'Hand']
    sens2 = ['Acc6', 'Acc16']
    features_names = []
    label = 'f_'
    for feat in F_features:
        for loc in loc2:
            for se in sens2:
                feature_name = label + feat + loc + se
                features_names.append(feature_name)
    return features_names


#acceleration correlation features

def acc_correlations(df):
    acc_corr_features = []
    acc_h_x_y = (df['handAcc16_x'].corr(df['handAcc16_y']))
    acc_corr_features.append(acc_h_x_y)
    acc_h_y_z = (df['handAcc16_y'].corr(df['handAcc16_z']))
    acc_corr_features.append(acc_h_y_z)
    acc_h_z_x = (df['handAcc16_z'].corr(df['handAcc16_x']))
    acc_corr_features.append(acc_h_z_x)
    acc_c_x_y = (df['chestAcc16_x'].corr(df['chestAcc16_y']))
    acc_corr_features.append(acc_c_x_y)
    acc_c_y_z = (df['chestAcc16_y'].corr(df['chestAcc16_z']))
    acc_corr_features.append(acc_c_y_z)
    acc_c_z_x = (df['chestAcc16_z'].corr(df['chestAcc16_x']))
    acc_corr_features.append(acc_c_z_x)
    acc_a_x_y = (df['ankleAcc16_x'].corr(df['ankleAcc16_y']))
    acc_corr_features.append(acc_a_x_y)
    acc_a_y_z = (df['ankleAcc16_y'].corr(df['ankleAcc16_z']))
    acc_corr_features.append(acc_a_y_z)
    acc_a_z_x = (df['ankleAcc16_z'].corr(df['ankleAcc16_x']))
    acc_corr_features.append(acc_a_z_x)
    acc_a_h = (df['ankleAcc16'].corr(df['handAcc16']))
    acc_corr_features.append(acc_a_h)
    acc_a_c = (df['ankleAcc16'].corr(df['chestAcc16']))
    acc_corr_features.append(acc_a_c)
    acc_c_h = (df['chestAcc16'].corr(df['handAcc16']))
    acc_corr_features.append(acc_c_h)

    return acc_corr_features

def corr_feat_labels():
    locn = ['Hand','Chest','Ankle']
    comp = ['x_y','y_z','z_x']
    features_names = []
    label = 'corr_'
    for loc in locn:
        for c in comp:
            feat_name = label+loc+c
            features_names.append(feat_name)
    abs_corr = ['corr_ankle_hand','corr_ankle_chest','corr_hand_chest']
    features_names.extend(abs_corr)
    return features_names


