import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocessing(X, y):
    numerical_cols = X.select_dtypes(include= ['number']).columns
    categorical_cols = X.select_dtypes(include= ['object', 'category']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

    scaler = StandardScaler()
    encoder = OneHotEncoder(drop= 'first', sparse_output= False)

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    X_train = X_train.drop(columns= categorical_cols)
    X_test = X_test.drop(columns= categorical_cols)

    X_train_encoded_cols = pd.DataFrame(data= X_train_encoded, columns= encoded_cols, index= X_train.index)
    X_test_encoded_cols = pd.DataFrame(data = X_test_encoded, columns= encoded_cols, index= X_test.index)

    X_train = pd.concat([X_train, X_train_encoded_cols], axis= 1).values
    X_test = pd.concat([X_test, X_test_encoded_cols], axis= 1).values

    y_train = y_train.values
    y_test = y_test.values

    return [X_train, X_test, y_train, y_test]

