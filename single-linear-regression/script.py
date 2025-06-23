import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, batch_size=65, n_iter=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.weights = None
        self.bias = None
    

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        num_batches = num_samples // self.batch_size

        for _ in range(self.n_iter):

            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, num_samples)
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                y_pred = self.predict(X_batch)
                err = y_pred - y_batch
                
                gradient_weights = np.dot(X_batch.T, err) / self.batch_size
                gradient_bias = np.mean(err)
                
                self.weights = self.weights - self.learning_rate * gradient_weights
                self.bias = self.bias - self.learning_rate * gradient_bias
            
           
            
            

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def remove_outliers(dataframe, column_name1):
    z = 1.5
    mean = dataframe[column_name1].mean()
    std = dataframe[column_name1].std() * z
    lower_bound = mean - std
    upper_bound = mean + std

    filtered_df = dataframe.loc[(dataframe[column_name1] < upper_bound) & (dataframe[column_name1] > lower_bound)]
    # print(len(dataframe), len(filtered_df))

    return filtered_df

def remove_outliers2(dataframe, column_name1):
    q1 = dataframe[column_name1].quantile(0.25)
    q3 = dataframe[column_name1].quantile(0.75)
    iqr = q3-q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)

    filtered_df = dataframe.loc[(dataframe[column_name1] < upper_bound) & (dataframe[column_name1] > lower_bound)]
    # print(len(dataframe), len(filtered_df))

    return filtered_df

def main(path_train, path_test):
    dfdraw = pd.read_csv('main.csv')
    xdraww = np.array(dfdraw['X'])
    ydrawa = np.array(dfdraw['Y'])
    plt.scatter(xdraww,ydrawa)
    plt.show()
    print(xdraww.shape)
    dfdraw = remove_outliers(dfdraw, 'Y')
    xdraww = np.array(dfdraw['X'])
    ydrawa = np.array(dfdraw['Y'])
    plt.scatter(xdraww,ydrawa)
    plt.show()
    print(xdraww.shape)

    df = pd.read_csv(path_train)
    df = remove_outliers(df,'Y')
    X_train = np.array(df['X']).reshape(-1,1)
    y_train = np.array(df['Y'])
    # df.info()

    df_test = pd.read_csv(path_test)
    X_test = np.array(df_test['X']).reshape(-1,1)
    y_test = np.array(df_test['Y'])
    # df_test.info()s

    # X_train_mean = np.mean(X_train, axis=0)
    # X_train_std = np.std(X_train, axis=0)
    # X_train_standardized = (X_train - X_train_mean) / X_train_std
    # X_test_standardized = (X_test - X_train_mean) / X_train_std

    # y_train_mean = np.mean(y_train, axis=0)
    # y_train_std = np.std(y_train, axis=0)
    # y_train_standardized = (y_train - y_train_mean) / y_train_std
    # y_test_standardized = (y_test - y_train_mean) / y_train_std

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)



    # yrmse_test = y_test_standardized * y_train_std + y_train_mean
    # yrmse_pred = y_pred * y_train_std + y_train_mean
    rmse = calculate_rmse(y_test, y_pred)
 
    print(rmse)

if __name__ == "__main__":
    path_train = sys.argv[1]
    path_test = sys.argv[2]
    main(path_train, path_test)