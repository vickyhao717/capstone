from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from plot_cm import plot_confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_processing import data_processing


def best_para(X_train, y_train):

    clf = LogisticRegression(penalty='l2')
    param_grid = {'C': np.arange(0.01, 100, 10)}
    lg = GridSearchCV(clf, param_grid, cv=5, return_train_score=True)
    lg.fit(X_train,y_train)
    bestmodel = lg.best_estimator_

    return bestmodel



def train(X_train, X_test, y_train, y_test, text_file):

    bst_model = best_para(X_train,y_train)

    print("After tuning the model, the best one is: ", file=text_file)
    print(bst_model,file=text_file)
    text_file.write("\n")

    bst_model.fit(X_train, y_train)
    print("Training Accuracy : " + str(bst_model.score(X_train, y_train)), file=text_file)
    y_pred = bst_model.predict(X_test)
    print("Tseting Accuracy : " + str(accuracy_score(y_test, y_pred)),file= text_file)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    print("Classification Report: ", file=text_file)
    print(classification_report(y_test, y_pred), file=text_file)

    return cnf_matrix


if __name__ == "__main__":
    df = pd.read_csv("dataSet_business.csv")
    text_file = open("./result/model_result.txt", "a+")
    text_file.write("\n")
    text_file.write(".This is Logistic Regression model")
    text_file.write("\n")


    X_train, X_test, y_train, y_test =  data_processing()
    cnf_matrix = train(X_train, X_test, y_train, y_test,text_file)

    print("Confusion matrix", file=text_file)
    print(cnf_matrix, file=text_file)

    text_file.close()

    class_names = ["bad", "good"]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix__Logistic Regression')