import numpy
import openpyxl
import pandas as pd
import scipy
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.impute as impute
import sklearn.preprocessing as preprocess
import sklearn.model_selection as modelselect
import sklearn.metrics as metrics
import pydotplus
#import collections

def generate_cols_IMU(name):
    cols_name = [name+'Temperature',
                 name+'Acc16_x',name+'Acc16_y', name+'Acc16_z',
                 name+'Acc6_x', name+'Acc6_y', name+'Acc6_z',
                 name+'Gyro_x', name+'Gyro_y', name+'Gyro_z',
                 name+'Magne_x', name+'Magne_y', name+'Magne_z',
                 name+'Orientation_x', name+'Orientation_y', name+'Orientation_z', name+'Orientation_w']
    return cols_name
def load_colnames():
    output = ['time_stamp','activity_id','heart_rate']
    hand = "hand"
    chest = "chest"
    ankle = "ankle"
    IMU_hand = generate_cols_IMU(hand)
    output.extend(IMU_hand)
    IMU_chest = generate_cols_IMU(chest)
    output.extend(IMU_chest)
    IMU_ankle = generate_cols_IMU(ankle)
    output.extend(IMU_ankle)
    return output

def prep():
    # load dataset from dat file
    df = pd.DataFrame()
    cols = load_colnames()
    dirname = 'C:/Personal_Data/VT SEM 1/Adv ML/PAMAP2_Dataset/PAMAP2_Dataset/Protocol/subject'
    dirname2 = 'C:/Personal_Data/VT SEM 1/Adv ML/PAMAP2_Dataset/PAMAP2_Dataset/Optional/subject'
    for i in range(101, 110):
        sub_file = dirname + str(i) + '.dat'
        subject = pd.read_table(sub_file, header=None, sep='\s+')
        subject.columns = cols
        subject['id'] = i
        df = df.append(subject, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    for i in [101, 105, 106, 108, 109]:
        sub_file = dirname2 + str(i) + '.dat'
        subject = pd.read_table(sub_file, header=None, sep='\s+')
        subject.columns = cols
        subject['id'] = i
        df = df.append(subject, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    df = df.drop(['handOrientation_x','handOrientation_y','handOrientation_z','handOrientation_w',
                  'chestOrientation_x','chestOrientation_y','chestOrientation_z','chestOrientation_w',
                  'ankleOrientation_x','ankleOrientation_y','ankleOrientation_z','ankleOrientation_w'],axis=1)
    df[['heart_rate']] = df[['heart_rate']].fillna(method='bfill')

    #map intensity
    intensity = {0: 999, 1: 0, 2: 0, 3: 1, 4: 1, 5: 3, 6: 3, 7: 2, 9: 0,
                 10: 1,11: 1, 12: 2, 13: 2, 16: 2, 17: 1, 18: 1, 19: 2, 20: 3,
                 24: 3}

    df['intensity'] = df['activity_id'].map(intensity)
    df2 = df.drop(df[df['activity_id']==0].index)
    #dfsort = df2.sort_values(by=['id','time_stamp'], ascending=[True, True])


    jj = df2.isna().sum()
    nan_values = df2[df2.isna().any(axis=1)]
    result = df2.dtypes

    df2 = df2.drop(['intensity'],axis=1)
    df3 = df2.fillna(df2.median()).to_numpy()

    df_s = df3[0:999,:]

    complex_f = scipy.fftpack.fft(df_s, axis=0)
    amplitude_f = numpy.abs(complex_f)
    print('hi')


if __name__ == '__main__':
    prep()