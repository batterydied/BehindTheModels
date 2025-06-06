import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor:
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        self.scaler = None
        self.encoder = None
        self.encoded_cols = None
        self.original_col_order = None

    def preprocess(self, X, y):
        self.original_col_order = list(X.columns)

        numerical_cols = X.select_dtypes(include= ['number']).columns
        categorical_cols = X.select_dtypes(include= ['object', 'category']).columns

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

        scaler = StandardScaler()
        encoder = OneHotEncoder(drop= 'first', sparse_output= False, handle_unknown='ignore')

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        encoded_cols = encoder.get_feature_names_out(categorical_cols)

        X_train = X_train.drop(columns= categorical_cols)
        X_test = X_test.drop(columns= categorical_cols)

        X_train_encoded_cols = pd.DataFrame(data = X_train_encoded, columns= encoded_cols, index= X_train.index)
        X_test_encoded_cols = pd.DataFrame(data = X_test_encoded, columns= encoded_cols, index= X_test.index)

        X_train = pd.concat([X_train, X_train_encoded_cols], axis= 1).values
        X_test = pd.concat([X_test, X_test_encoded_cols], axis= 1).values

        y_train = y_train.values
        y_test = y_test.values

        self.scaler = scaler
        self.encoder = encoder
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.encoded_cols = encoded_cols

        return [X_train, X_test, y_train, y_test]

    def transform_input(self, input):
        input_df = pd.DataFrame(input, columns = self.original_col_order)

        input_df[self.numerical_cols] = self.scaler.transform(input_df[self.numerical_cols])

        input_encoded = self.encoder.transform(input_df[self.categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns = self.encoded_cols, index = input_df.index)

        input_df = input_df.drop(columns=self.categorical_cols)
        input_df = pd.concat([input_df, input_encoded_df], axis=1)

        return input_df.values


