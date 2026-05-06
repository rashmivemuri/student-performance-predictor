import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def train_model():
    df = pd.read_excel("student_data.csv.xlsx")
    X = df.drop('result', axis=1)
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k_scores = {}
    for k_val in range(2, 21, 2):
        m = knn(n_neighbors=k_val)
        m.fit(X_train, y_train)
        k_scores[k_val] = round(m.score(X_test, y_test) * 100, 2)

    model = knn(n_neighbors=4)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), target_names=["Fail", "Pass"])

    return model, scaler, train_acc, test_acc, k_scores, cm, report