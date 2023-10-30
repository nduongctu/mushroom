import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def get_y_pred(sample):
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', header=None)

    df = df.drop(16, axis=1)
    df = df.drop(11, axis=1)

    df.replace('?', np.nan, inplace=True)

    le = LabelEncoder()
    for column in df.columns:
        df[column] = le.fit_transform(df[column])

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    x1 = pd.DataFrame([sample])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(x1)
    if y_pred == 0:
        return "không ăn được"
    else:
        return "ăn được"
