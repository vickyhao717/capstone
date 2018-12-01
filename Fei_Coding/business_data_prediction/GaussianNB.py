from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from plot_cm import plot_confusion_matrix
from sklearn.metrics import classification_report
from data_processing import data_processing


def train(X_train, X_test, y_train, y_test, text_file):

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    print("Training Accuracy : " + str(clf.score(X_train,y_train)), file=text_file)

    text_file.write("\n")
    y_pred = clf.predict(X_test)
    print("Tseting Accuracy : " + str(accuracy_score(y_test, y_pred)),file= text_file)
    print("\n", file=text_file)

    print("Classification Report: ", file=text_file)
    print(classification_report(y_test, y_pred), file=text_file)
    # print("\n",file=text_file)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    return cnf_matrix


if __name__ == "__main__":
    df = pd.read_csv("dataSet_business.csv")
    text_file = open("./result/model_result.txt", "a+")
    text_file.write("\n")
    text_file.write(".This is GussianNB model")
    text_file.write("\n")


    X_train, X_test, y_train, y_test =  data_processing()
    cnf_matrix = train(X_train, X_test, y_train, y_test,text_file)

    print("Confusion matrix", file=text_file)
    print(cnf_matrix,file=text_file)

    text_file.close()

    class_names = ["bad", "good"]
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix__GussianNB')