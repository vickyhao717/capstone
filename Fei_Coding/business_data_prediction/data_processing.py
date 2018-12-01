from sklearn.model_selection import train_test_split
import pandas as pd

def data_processing():
    df = pd.read_csv("dataSet_business.csv")

    df["stars"].replace([1.0, 1.5, 2.0,2.5], 0, inplace=True)
    df["stars"].replace([5.0,4.5], 1, inplace=True)
    df1 = df[df["stars"].isin({1.0, 0})]

    X = df1.drop("stars", axis=1)
    Y = df1["stars"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test
