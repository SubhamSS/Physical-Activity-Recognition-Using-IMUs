#Wrote by Subham Swastik Samal

#This code takes the time series data, divides the dataset into small windows,
# and then generates features (statistical, correlation, energy and frequency based)
#Finally returns the time-series processed dataset

#dont run this code, instaed run the 'main_ts_l2,py' file

import pandas
from time_series_feat import *

def timeseries_prep(df3, overlap,i, path):

    df3.sort_values(by=['id', 'time_stamp'], ascending=[True, True])
    df_x = df3.drop(['time_stamp','activity_id','IN_predict'],axis =1)
    df_y = df3[['activity_id','id']]

    #windowing parameters
    sample_freq = 100
    window_len = 169

    ids = df3['id'].unique()

    for ii in ids:
        dfid = df_x[df_x["id"] == ii]
        df_ac = df_y[df_y["id"] == ii]
        dfid = dfid.drop(['id'],axis=1)
        df_ac = df_ac.drop(['id'],axis=1)
        df5 = dfid
        ts_sets = int(1 + (len(df5) - window_len) // (window_len *(1- overlap)))

        for jj in range(ts_sets):

            w_start = int(jj * window_len * (1-overlap))
            w_end = int(w_start + window_len)
            wind_df = df5[w_start:w_end]

            y_set = df_ac[w_start:w_end]

            temp_y = numpy.floor(numpy.median(y_set,axis=0))
            temp_y = list(temp_y)

            # transformations
            t_data = wind_df
            df_fft = fast_fourier_transform(wind_df)

            # features extract
            t_features = time_features(t_data)
            f_features = freq_features(df_fft, sample_freq)
            corr_features = acc_correlations(t_data)
            feature_vector = t_features + f_features + corr_features + temp_y
            feature_vector.append(ii)

            if jj == 0 and ii == min(ids) :
                feat_np = numpy.array(feature_vector).reshape((1, -1))
            else:
                feat_np = numpy.vstack((feat_np, numpy.array(feature_vector).reshape((1, -1))))

        #extracted feature names
        feature_titles = time_feat_labels() + time_mag_labels() + \
                         freq_feat_labels() + freq_mag_labels() + corr_feat_labels()
        feature_titles.extend(['activity_id','subject_id'])

    feat_df = pandas.DataFrame(feat_np)
    feat_df.columns = feature_titles
    feat_df.to_csv(path+'ts_prep2_'+str(i)+'.csv') #returns time series processed csv for each intensity
    return feat_df

if __name__ == '__main__':
    timeseries_prep()