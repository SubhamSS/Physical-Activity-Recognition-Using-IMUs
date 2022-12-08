#Wrote by Subham Swastik Samal

#evaluation code for activity classifier (layer 2) with feature selection; evaluated @ LOSO
#(Leave one subject out) and WSE (Within subject evaluation)

#change nothing here, and don't run this.

import sklearn.model_selection as modelselect
import pandas
import numpy
import sklearn.svm as svm
import sklearn.ensemble as es
import sklearn.preprocessing as pp
import sklearn.feature_selection as fs
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

#Returns scores after feature selection and evaluations
def run_act_clf(df_int, classifier,path,cl_int):
    filepath = path
    ac_id = df_int[['activity_id']]
    sub_id = df_int[['subject_id']]

    X = df_int.drop(['activity_id','subject_id'], axis=1)

    #normalization
    X_n = (X - X.min()) / (X.max() - X.min())
    X_n.insert(loc=0, column='activity_id', value=ac_id)
    X_n.insert(loc=0, column='subject_id', value=sub_id)

    #remove features with NA
    a = X_n.isna().sum()
    variables = X_n.columns
    variable = []
    for i in range(0, len(X_n.columns)):
        if a[i] == 0:  # setting the threshold as 20%
            variable.append(variables[i])
    X_n = X_n[variable]

    print("Xn done")

    #Random Forest based RFECV for feature selection
    clf = es.RandomForestClassifier(criterion="log_loss",bootstrap = "False",
                                   n_estimators=20,min_samples_leaf=30)
    CA_WSE = within_sub_eval(X_n, clf, classifier, filepath, cl_int)
    CA = normal(X_n, clf, classifier, filepath, cl_int)
    CA_LOSO = loso_cv(X_n,clf, classifier,filepath, cl_int)
    return CA, CA_LOSO, CA_WSE

#does feature selection, returns transformed dataset for evaluation
def feature_select(clf,classifier, X, Y, X_t):

    #Remove features with high correlation
    correlated_features = set()
    correlation_matrix = X.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    len(correlated_features)
    print(len(correlated_features))
    print("removed high corr")

    newxtrain = X.drop(labels=correlated_features, axis=1)
    newxtest = X_t.drop(labels=correlated_features, axis=1)

    #remove more features using RFECV
    selector = fs.RFECV(clf, step = 20, cv = 5, min_features_to_select=50)
    selector.fit(newxtrain, Y.ravel())
    print("rfecv done")

    ne2train = selector.transform(newxtrain)
    ne2test = selector.transform(newxtest)
    print("transformed")

    classifier.fit(ne2train, Y.reshape(-1,1))
    print("Features selected: ", selector.n_features_)
    return ne2train, ne2test

#LOSO Evaluation
def loso_cv(data,clf,classifier,filepath, cl_int):
    print("loso start")
    target_var = 'activity_id'
    CA = []
    obs = []
    for ii in range(101, 110):  #corner case if very less data
        print(ii)
        data_test = data[data["subject_id"] == ii]

        if(data_test.shape[0]) <10:
            continue

        #test-train split
        data_train = data[data["subject_id"] != ii]
        X_test = data_test.drop(['activity_id','subject_id'], axis=1)
        Y_test = data_test[target_var].to_numpy()
        X_train = data_train.drop(['activity_id', 'subject_id'], axis=1)
        Y_train = data_train[target_var].to_numpy()

        #feature selection based on train
        X_tr, X_te = feature_select(clf, classifier, X_train, Y_train, X_test)

        # run learning model on transformed test set after feature selection
        Y_pred = classifier.predict(X_te)
        CA.append(metrics.accuracy_score(Y_test, Y_pred)*len(Y_test))
        obs.append(len(Y_test))
        conf1 = metrics.confusion_matrix(Y_test, Y_pred)
        df_cfm = pandas.DataFrame(conf1)
        plt.figure(figsize=(10, 7))
        cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='g')
        cfm_plot.figure.savefig(filepath+"int_"+str(cl_int)+"cfm_loso_"+str(ii)+".png") #save conf matrix
    CA_LOSO = sum(CA)/sum(obs)

    #print classification accuracy
    print("For Leave one Subject Out evaluation of intensity %d, the Classification Accuracy = %f"%(cl_int, CA_LOSO))
    return CA_LOSO

# Within Subject Evaluation
def within_sub_eval(data,clf, classifier, filepath, cl_int):
    print("wse start")
    target_var = 'activity_id'
    CA = []
    obs = []
    for ii in range(101,110):
        print(ii)
        data_ws = data[data["subject_id"] == ii]
        if (data_ws.shape[0]) < 10: #corner case if very less data
            continue
        if (len(pandas.unique(data_ws['activity_id']))) == 1: #corner case if all data have same label
            obs.append(data_ws.shape[0]*0.35)
            CA.append(data_ws.shape[0]*0.35)
            continue

        #test-train split
        X_ws = data_ws.drop(['activity_id','subject_id'], axis=1)
        Y_ws = data_ws[target_var].to_numpy()
        X_train, X_test, Y_train, Y_test = modelselect.train_test_split(X_ws, Y_ws, test_size=0.35, random_state=44565)

        if (len(numpy.unique(Y_train))) == 1:
            obs.append(data_ws.shape[0]*0.35)
            CA.append(data_ws.shape[0]*0.35)
            continue

        X_tr, X_te = feature_select(clf,classifier, X_train, Y_train, X_test) #feature selection
        Y_pred = classifier.predict(X_te)   #run learning model on transformed test set after feature selection
        CA.append(metrics.accuracy_score(Y_test, Y_pred)*len(Y_test))
        obs.append(len(Y_test))
        conf1 = metrics.confusion_matrix(Y_test, Y_pred)
        df_cfm = pandas.DataFrame(conf1)
        plt.figure(figsize=(10, 7))
        cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='g')
        cfm_plot.figure.savefig(filepath+"int_"+str(cl_int)+"cfm_within_"+str(ii)+".png") #save confusion matrix
    CA_WSE = sum(CA)/sum(obs)

    # print classification accuracy
    print("For Within Subject evaluation of intensity %d, the Classification Accuracy = %f" % (cl_int, CA_WSE))
    return CA_WSE

#overall evaluation
def normal(data,clf, classifier, filepath, cl_int):
    print("all start")
    target_var = 'activity_id'

    X_ws = data.drop(['activity_id','subject_id'], axis=1)
    Y_ws = data[target_var].to_numpy()
    X_train, X_test, Y_train, Y_test = modelselect.train_test_split(X_ws, Y_ws, test_size=0.35, random_state=44565)

    X_tr, X_te = feature_select(clf,classifier, X_train, Y_train, X_test)
    Y_pred = classifier.predict(X_te)
    CA = metrics.accuracy_score(Y_test, Y_pred)
    conf1 = metrics.confusion_matrix(Y_test, Y_pred)
    df_cfm = pandas.DataFrame(conf1)
    plt.figure(figsize=(10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='g')
    cfm_plot.figure.savefig(filepath+"int_"+str(cl_int)+"cfm_all.png")

    # print classification accuracy
    print("For all evaluation of intensity %d, the Classification Accuracy = %f" % (cl_int, CA))
    return CA

if __name__ == '__main__':
    run_act_clf()