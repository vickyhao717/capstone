import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
from plot_cm import plot_confusion_matrix
from data_processing import data_processing

import warnings
warnings.filterwarnings("ignore")

def best_para(X_train, y_train):

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']}

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X_train, y_train)

    bestmodel = CV_rfc.best_estimator_

    return bestmodel

def train(X_train, X_test, y_train, y_test, text_file):

    bst_model = best_para(X_train,y_train)

    print("After tuning the model, the best one is: ", file=text_file)
    print(bst_model,file=text_file)
    text_file.write("\n")

    bst_model.fit(X_train, y_train)
    print("Training Accuracy : " + str(bst_model.score(X_train, y_train)), file=text_file)
    y_pred = bst_model.predict(X_test)

    print("Classification Report: ",file=text_file)
    print(classification_report(y_test, y_pred),file=text_file)

    print("Tseting Accuracy : " + str(accuracy_score(y_test, y_pred)),file= text_file)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    return cnf_matrix

if __name__ == "__main__":
    df = pd.read_csv("dataSet_business.csv")
    text_file = open("./result/model_result.txt", "a+")
    text_file.write("\n")
    text_file.write(".This is RandomForest Classifier model")
    text_file.write("\n")


    X_train, X_test, y_train, y_test =  data_processing()
    cnf_matrix = train(X_train, X_test, y_train, y_test,text_file)

    print("Confusion matrix", file=text_file)
    print(cnf_matrix, file=text_file)

    text_file.close()

    class_names = ["bad", "good"]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix__RandomForest Classifier')