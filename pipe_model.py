import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import spearmanr
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import joblib
import numpy as np


def make_model(argv):
    csv_path = argv[0]
    dataset_name = str(argv[1])
    model_name = str(argv[2])

    # Load the .csv file into a DataFrame
    data = pd.read_csv(csv_path)

    ### ÄNDRA DESSA
    max_depth_values = [9,12]
    min_count_values = [25,100]
    threshold_values = [0.0, 3.9075]
    hidden_layer_sizes = (2500, 2000, 1500, 1000, 1000)

    features = ['VLMC dist', 'threshold', 'min_count', 'max_depth', 'Tree1 VLMC size', 'Tree2 VLMC size', 'Tree1 GC ratio', 'Tree2 GC ratio']

    # Filter the rows
    data = data[data['max_depth'].isin(max_depth_values)]
    data = data[data['min_count'].isin(min_count_values)]
    data = data[data['threshold'].isin(threshold_values)]

    X = data[features]
    X['VLMC dist'] = X['VLMC dist'].fillna(0)
    y = data["evo dist"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', learning_rate='adaptive', tol=1e-6, early_stopping=True, max_iter=1000)

    regressor = TransformedTargetRegressor(regressor=mlp, transformer=MinMaxScaler())

    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', regressor)
    ])

    pipe.fit(X_train, y_train,)

    y_pred_mlp = pipe.predict(X_test)

    print("MLPRegressor")
    print("Score    :", round(pipe.score(X_test, y_test),4))
    print("Spearman :", round(spearmanr(y_pred_mlp, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_mlp, squared=False),4))

    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()

    regr = LinearRegression().fit(X_train_scaled, y_train_scaled)
    y_pred_lr = regr.predict(np.array(X_test_scaled).reshape(-1, len(X.columns)))
    y_pred_lr_d = scaler_y.inverse_transform(y_pred_lr.reshape(-1,1)).flatten()

    print("\n")
    print("Linear regressor")
    print(f"Coefficient of determination: {regr.score(np.array(X_test_scaled).reshape(-1, len(X.columns)), y_test_scaled)}")
    print(f"Spearman R: {spearmanr(y_pred_lr_d, y_test)}")
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred_lr_d, squared=False)}")
    

    # Save the model
    if not os.path.exists('./models'):
        os.makedirs('./models')

    joblib.dump(pipe, './models/'+model_name)

    X_train.to_csv('./models/X_train_'+dataset_name)
    y_train.to_csv('./models/y_train_'+dataset_name)
    X_test.to_csv('./models/X_test_'+dataset_name)
    y_test.to_csv('./models/y_test_'+dataset_name)


if __name__ == "__main__":
    if len(sys.argv[1:]) != 3:
        print("The script requires 3 arguments as <file.csv> <dataset_name.csv> <model_name.pkl>")
    else:
        make_model(sys.argv[1:])
