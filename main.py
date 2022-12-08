import pandas
from sklearn import model_selection as ms
import pandas as pd
import  numpy as np
import sklearn.preprocessing as pp
import sklearn.impute as impute
from sklearn.neighbors import KNeighborsClassifier as knc
import sklearn.feature_selection as fs
from sklearn import tree
from sklearn import metrics
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import preprocess_1


####### Module coded by Shahwar Atiq Khursheed
def dataprep(X, Y):

  Xscalar_1 = pp.MinMaxScaler(feature_range=(-1,1))
  Yscalar_1 = pp.MinMaxScaler(feature_range=(-1,1))
  X = Xscalar_1.fit_transform(X)
  return X, Y

def feature_select(clf, X, Y, X_t):
    selector = fs.RFECV(clf, step = 1, cv = 5)  # parameters dependent on the final dataset that we get
    selector.fit(X, Y.ravel())

    newxtrain = selector.transform(X)
    newxtest = selector.transform(X_t)

    clf.fit(newxtrain, Y.ravel())
    print("Features selected: ", selector.n_features_)
    return newxtrain, newxtest, selector, selector.n_features_

def evaluation(model, Y_train, Y_test, Y_pred , mode):
    if mode == 0:
        R2 = metrics.r2_score(Y_test, Y_pred)
        MSE = metrics.mean_squared_error(Y_test, Y_pred)
        CA = metrics.accuracy_score(Y_test, Y_pred)
        print("R2 = %f, MSE = %f, Classification Accuracy = %f" %
              (metrics.r2_score(Y_test, Y_pred), metrics.mean_squared_error(Y_test, Y_pred),
               metrics.accuracy_score(Y_test, Y_pred)))
        conf1 = metrics.confusion_matrix(Y_test, Y_pred)
        print(conf1)
    elif mode == 1:
        R2 = metrics.r2_score(Y_test, Y_pred)
        MSE = metrics.mean_squared_error(Y_test, Y_pred)
        CA = 0
        print("R2 = %f, MSE = %f" %
              (metrics.r2_score(Y_test, Y_pred), metrics.mean_squared_error(Y_test, Y_pred)))

    return R2, MSE, CA

def between_subject(model, data):
    print("LOOCV start")
    R2 = []
    MSE = []
    CA = []

    for i in range(101, 110):
        # leave one subject out
        LOOCV_O = str(i)
        data['id'] = data['id'].apply(str)
        data_train = data[data['id'] != LOOCV_O]
        data_test= data[data['id'] == LOOCV_O]

        # Test data - the person left out of training
        # Use this for classification
        X_test = data_test.drop(columns = ['intensity', 'id'])
        # Use this for the regression model
        # X_test = data_test.drop(columns = ['heart_rate', 'id'])

        Y_test = data_test.intensity.to_numpy()  # This is the outcome variable

        # Train data - all other people in dataframe
        # Use this for classification
        X_train = data_train.drop(columns = ['intensity','id'])

        # Use this for the regression model
        # X_train = data_train.drop(columns = ['heart_rate','id'])

        Y_train = data_train.intensity.to_numpy()  # This is the outcome variable

        X_train, Y_train = dataprep(X_train, Y_train)
        X_test, Y_test = dataprep(X_test, Y_test)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        # put 0 for classification and 1 for regression
        r2, mse, ca = evaluation(model, Y_train, Y_test, Y_pred, 0)

        R2.append(r2)
        MSE.append(mse)
        CA.append(ca)

    R_2 = st.mean(R2)
    M_S_E = st.mean(MSE)
    C_A = st.mean(CA)

    print("For Between Subjects evaluation,")
    print("R2 = %f, MSE = %f, Classification Accuracy = %f" %
          (R_2, M_S_E, C_A))

    return R_2, M_S_E, C_A

def within_subject(data,model):
    print("wse start")
    # target_var = 'intensity'
    target_var = 'heart_rate'
    R2 = []
    MSE = []
    CA = []
    data['id'] = data['id'].apply(str)
    for ii in range(101, 110):
        ind = str(ii)
        data_ws = data[data['id'] == ind]
        # Use this for classification
        X_ws = data_ws.drop(['intensity', 'id'], axis=1)
        # Use this for the regression model
        # X_ws = data_ws.drop(['heart_rate', 'id'], axis=1)
        Y_ws = data_ws[target_var].to_numpy()
        X_train, X_test, Y_train, Y_test = ms.train_test_split(X_ws, Y_ws, test_size=0.35, random_state=44565)

        X_train, Y_train = dataprep(X_train, Y_train)
        X_test, Y_test = dataprep(X_test, Y_test)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        Y_test = Y_test.reshape(-1,1)
        Y_pred = Y_pred.reshape(-1,1)

        # put 0 for classification and 1 for regression
        r2, mse, ca = evaluation(model, Y_train, Y_test, Y_pred, 0)

        R2.append(r2)
        MSE.append(mse)
        CA.append(ca)

        #comment for the regression model
        conf1 = metrics.confusion_matrix(Y_test, Y_pred)
        df_cfm = pandas.DataFrame(conf1)
        plt.figure(figsize=(10, 4))
        cfm_plot = sn.heatmap(df_cfm, annot=True)
        cfm_plot.figure.savefig("C:\\Users\\pc\\Downloads\\HR" + str(ii) + ".png")
    R2_WSE = st.mean(R2)
    MSE_WSE = st.mean(MSE)
    CA_WSE = st.mean(CA)
    print("For Within Subject evaluation, the metrics are:\nR2 Score = %f \nMSE = %f "
          "\nClassification Accuracy = %f" % (R2_WSE, MSE_WSE, CA_WSE))

def get_model(a): #Contributions by Venkata and Subham
    if a == 1:
        clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state= 1000)
    elif a == 2:
        clf = SVC(C=10.0, kernel='linear', degree=3, gamma=1.0, coef0=0.0, shrinking=True, probability=False,
    tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
    break_ties=False, random_state=None)
    elif a == 3:
        clf = knc(n_neighbors=5, metric='minkowski', p=2)
    elif a ==4:
        clf = LinearRegression()
    else:
        print('Incorrect argument for model selection')
    return clf

if __name__ == '__main__':
    # load dataset if saved
    # pathName = "C:\\Users\\pc\\Downloads\\"
    # fileName = "preprocessedv2.csv"
    # dataFrame_OG = pd.read_csv(pathName + fileName)
    # dataFrame_OG = dataFrame_OG.drop(columns = ['Unnamed: 0'])

    #loads datset from the Prep module
    dataFrame_OG = preprocess_1.prep()

    # ataFrame_OG.to_pickle("my_data.pkl")
    # dataFrame_OG = pd.read_pickle('my_data.pkl')

    Activity_IDs = dataFrame_OG.activity_id
    Activity_IDs = Activity_IDs.transpose().reset_index()
    Activity_IDs = Activity_IDs.drop(columns = ['index'])
    dataFrame = dataFrame_OG.drop(columns = ['activity_id']).reset_index()

# Arguments: 1 for a Random Forest Classifier, 2 for a SVM, 3 for KNN and 4 for Linear Regression
    clf = get_model(1)

# #feature selection
    #Use for the regression model
#     X_full = dataFrame.drop(columns = ["heart_rate", 'id'])

    #Use for the classifiers
    X_full = dataFrame.drop(columns = ["intensity", 'id'])
    IDs = dataFrame.id.to_numpy()
    #Use for the classifiers
    Y_full = dataFrame.intensity.to_numpy()
    #Use for the regression model
    # Y_full = dataFrame.heart_rate.to_numpy()
    Y_full = Y_full.reshape(-1,1)
    IDs = IDs.reshape(-1,1)
    X_use = X_full
    Y_use = Y_full

    X_train, X_test, Y_train, Y_test = ms.train_test_split(X_use, Y_use, test_size=0.3, random_state= 10)
    X_train, Y_train = dataprep(X_train, Y_train)
    Y_train = Y_train.reshape(-1,1)
    X_test, Y_test = dataprep(X_test, Y_test)
    Y_test = Y_test.reshape(-1,1)
    X_train, X_test, selection, selected_f = feature_select(clf,X_train, Y_train, X_test)

    X_full = selection.transform(X_full)
#
    dataFrame_int = np.concatenate([X_full, IDs, Y_full], axis = 1)
    dataFrame_int.to_pickle("my_data_fs.pkl")
# #
    dataFrame_int = pd.DataFrame(dataFrame_int, columns = ['A','B','C','D','E','F','G','H','I',
                                                           'J','K','L','M','N','O','P',
                                                           'id', 'intensity']) # needs to be changed according to the feature selection result
#     dataFrame_int.to_pickle("my_data_fs.pkl")
#
#     #feature selected csv - use if dataframe or .csv file is saved
#     # dataFrame_int = pd.read_pickle('my_data_fs.pkl')
#     pathName = "C:\\Users\\pc\\Downloads\\"
#     fileName = "dataFrame_int.csv"
#     dataFrame_int = pd.read_csv(pathName + fileName)
#     dataFrame_int = dataFrame_int.drop(columns=['Unnamed: 0'])

    #Uncomment for the regression model
    # dataFrame_int = dataFrame
    dataFrame_int['id'] = dataFrame_int['id'].apply(np.int64)

    #Uncomment for 1 fold random split evaluation
##################################
#     X = dataFrame_int.drop(['intensity', 'id'], axis=1)
#     Y = dataFrame_int.intensity.to_numpy()
#     X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.3, random_state=44565)
#
#     X_train, Y_train = dataprep(X_train, Y_train)
#     X_test, Y_test = dataprep(X_test, Y_test)
#
#     clf.fit(X_train, Y_train)
#     Y_pred = clf.predict(X_test)
#     R2, MSE, CA = evaluation(clf, Y_train, Y_test, Y_pred)
#
#     conf1 = metrics.confusion_matrix(Y_test, Y_pred)
#     df_cfm = pandas.DataFrame(conf1)
#     plt.figure(figsize=(10, 4))
#     cfm_plot = sn.heatmap(df_cfm, annot=True)
#     cfm_plot.figure.savefig("C:\\Users\\pc\\Downloads\\SVM_comp" + str(1000) + ".png")
#
#     print("For Within Subject evaluation, the metrics are:\nR2 Score = %f \nMSE = %f "
#           "\nClassification Accuracy = %f" % (R2, MSE, CA))
#############################
    within_subject(dataFrame_int, clf)
    between_subject(clf, dataFrame_int)
# #
# #dataset for activity based classification
    X_full = dataFrame_int.drop(columns = ["intensity", 'id'])
    Y_full = dataFrame_int.intensity.to_numpy()
    X_full, Y_full = dataprep(X_full, Y_full)
    Y_full = Y_full.reshape(-1,1)
    IN_predict = clf.predict(X_full)
    IN_predict = pd.DataFrame(IN_predict, columns=['IN_predict'])
# #
    dataFrame = pd.concat([dataFrame, IN_predict], axis = 1, ignore_index= False)
    dataFrame = pd.concat([dataFrame, Activity_IDs], axis = 1, ignore_index= False)

    CA_full = metrics.accuracy_score(dataFrame.intensity.to_numpy(), IN_predict.to_numpy())
    R2_full = metrics.r2_score(dataFrame.intensity.to_numpy(), IN_predict.to_numpy())
    MSE_full = metrics.mean_squared_error(dataFrame.intensity.to_numpy(), IN_predict.to_numpy())
    print("For intensity prediction on the full dataset, the metrics are:\nR2 Score = %f \nMSE = %f "
          "\nClassification Accuracy = %f" % (R2_full, MSE_full, CA_full))

    #change path here
    dataFrame_act = dataFrame.drop(columns = ['index'])
    dataFrame_act.to_pickle("my_data_act.pkl")
    dataFrame_act.to_csv("C:\\Users\\pc\\Downloads\\dataFrame_act.csv")

    pass