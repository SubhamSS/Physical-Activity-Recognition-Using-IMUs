#Wrote by Subham Swastik Samal and Venkata Adithya Chekuri

#This is the main function for Activity evaluation, run this code to get results
#This calls functions from other files "ts_eval.py", "time_series_feat.py",
# and "ts_prep.py" to run
#Only change the from and to paths, classifier and overlap in the start function
#Results: For each iteration, Confusion Martices will be generated and saved, metrics will be printed


from ts_prep import *
from ts_eval import *

def start():

    #change the below line to your file generated after activity classification
    df = pandas.read_csv('C:/Personal_Data/VT SEM 1/Adv ML/imu_files/dataFrame_act.csv')
    df.drop(['Unnamed: 0', 'intensity'], axis=1, inplace=True)

    #change to your preferred classifier
    classifier = es.RandomForestClassifier(criterion="log_loss", bootstrap="False",
                                           n_estimators=20, min_samples_leaf=30)
    #classifier = svm.LinearSVC(max_iter=1000, C = 1)

    #change overlap ratio (30%, 50%, 70% considered for our project)
    window_overlap = 0.3

    #this is where every confusion matrix image would be stored...
    #create a new folder every time u run and change the path accordingly
    path = "C:/Personal_Data/VT SEM 1/Adv ML/imu_files/knn_rf_0.3/"

    CA_LOSO_ov = []
    CA_WSE_ov = []
    ca_ov = []
    ts_windows = []
    for cl_int in [3,2,1,0]:
        print("intensity %d start"%cl_int)
        df_int = df[df["IN_predict"] == cl_int] #separate the dataset based on predicted intensities
        df_int_ts = timeseries_prep(df_int, window_overlap, cl_int, path) #prepare time-series preprocessed dataset
        print("time series features extracted")

        # Run the algorithm (includes feature selection, LOSO and WSE evaluation)
        ca, CA_LOSO, CA_WSE = run_act_clf(df_int_ts, classifier,path,cl_int)
        print("intensity %d complete" % cl_int)
        CA_LOSO_ov.append(CA_LOSO * df_int_ts.shape[0])
        CA_WSE_ov.append(CA_WSE * df_int_ts.shape[0])
        ca_ov.append(ca * df_int_ts.shape[0])
        ts_windows.append(df_int_ts.shape[0])

    CA_OV_LOSO = sum(CA_LOSO_ov)/sum(ts_windows)
    CA_OV_WSE = sum(CA_WSE_ov)/sum(ts_windows)
    ca_oov = sum(ca_ov)/sum(ts_windows)

    #print final metrics
    print("The overall metrics are:\n"
          "For Within Subject evaluation:Classification Accuracy = %f\n"
          "For LOSO evaluation:Classification Accuracy = %f\n"
          "Overall evaluation:Classification Accuracy = %f\n" % (CA_OV_WSE, CA_OV_LOSO,ca_oov))

if __name__ == '__main__':
    start()