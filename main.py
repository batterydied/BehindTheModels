import numpy as np
import pandas as pd
from preprocessing import Preprocessor
from decision_tree_regressor import DecisionTreeRegressor
from linear_regression import LinearRegression
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from sklearn.linear_model import LinearRegression as SKLinearRegression

def main():
    df = pd.read_csv('./data/student_habits_performance.csv')
    X = df.iloc[:, 1: -1]
    y = df.iloc[:, -1]

    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(X, y)

    dtr = DecisionTreeRegressor()
    lr = LinearRegression(batch_size = 100)

    skdtr = SKDecisionTreeRegressor()
    sklr = SKLinearRegression()

    dtr.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    skdtr.fit(X_train, y_train)
    sklr.fit(X_train, y_train)

    print(f'Decision Tree r2: {dtr.score(X_test, y_test)}')
    print(f'SK Decision Tree r2: {skdtr.score(X_test, y_test)}')
    print('-----------------')
    print(f'Linear Regressor r2: {lr.score(X_test, y_test)}')
    print(f'SK Linear Regressor r2: {skdtr.score(X_test, y_test)}')

if __name__ == '__main__':
    main()



