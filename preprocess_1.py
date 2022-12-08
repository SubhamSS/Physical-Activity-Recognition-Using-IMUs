#Wrote by Subham Swastik Samal

#This code loads the raw dataset, performs these steps:
#add column names, data cleaning, assigning intensity,
#add absolute acceleration columns, imputaion, and finally
#reducing the dataset size

#change lines 39, 40 for input path and line 96 for output path

import numpy
import openpyxl
import pandas as pd

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
    del df
    df2 = df2.fillna(df2.median())

    hand_acc16 = numpy.sqrt(df2["handAcc16_x"]**2 + df2["handAcc16_y"]**2 + df2["handAcc16_z"]**2)
    df2.insert(loc=42, column='handAcc16', value=hand_acc16)
    chest_acc16 = numpy.sqrt(df2["chestAcc16_x"]**2 + df2["chestAcc16_y"]**2 + df2["handAcc16_z"]**2)
    df2.insert(loc=42, column='chestAcc16', value=chest_acc16)
    ankle_acc16 = numpy.sqrt(df2["ankleAcc16_x"]**2 + df2["ankleAcc16_y"]**2 + df2["ankleAcc16_z"]**2)
    df2.insert(loc=42, column='ankleAcc16', value=ankle_acc16)
    hand_acc6 = numpy.sqrt(df2["handAcc6_x"]**2 + df2["handAcc6_y"]**2 + df2["handAcc6_z"]**2)
    df2.insert(loc=42, column='handAcc6', value=hand_acc6)
    chest_acc6 = numpy.sqrt(df2["chestAcc6_x"]**2 + df2["chestAcc6_y"]**2 + df2["handAcc6_z"]**2)
    df2.insert(loc=42, column='chestAcc6', value=chest_acc6)
    ankle_acc6 = numpy.sqrt(df2["ankleAcc6_x"]**2 + df2["ankleAcc6_y"]**2 + df2["ankleAcc6_z"]**2)
    df2.insert(loc=42, column='ankleAcc6', value=ankle_acc6)

    jj = df2.isna().sum() #check no of NA
    nan_values = df2[df2.isna().any(axis=1)]
    result = df2.dtypes #Check Datatype

    df_med = df2[['id','activity_id','intensity']]
    df_mean = df2.drop(['id','activity_id','intensity'],axis =1)
    df_med = df_med.rolling(3).median()
    df_mean = df_mean.rolling(3).mean()
    df_med = df_med.iloc[3::3, :]
    df_med = df_med.astype(int)
    df_mean = df_mean.iloc[3::3, :]
    dfinal= pd.concat([df_mean,df_med],axis=1)

    dfinal.to_csv('C:/Personal_Data/VT SEM 1/Adv ML/imu_files/preprocessedv2.csv')

if __name__ == '__main__':
    prep()