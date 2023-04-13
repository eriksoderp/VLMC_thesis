import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import pandas as pd
import math
import sys


def make_model(argv):
    model_filename = str(argv[0])
    csv_filename = str(argv[1])
    run_linear = int(argv[2])
    X_args = argv[3].split(',')

    # Load the saved model
    with open(model_filename, 'rb') as f:
        loaded_model = pickle.load(f)


    # Load the .csv file into a DataFrame
    data = pd.read_csv(csv_filename)

    # Filter the rows
    data = data[data['max_depth'].isin([9,12])]
    data = data[data['min_count'].isin([25,100])]
    data = data[data['threshold'].isin([3.9075])]

    # Prepare the data
    #X = data[["VLMC dist","Original VLMC size","Mod VLMC size","min_count","max_depth"]].fillna(0)
    #y = data["Evolutionary dist"]
    X = data[X_args].fillna(0)
    y = data["evo dist"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize the features using a Scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale the target variable
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    y_pred_scaled = loaded_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    print("MLPRegressor")
    print("Score    :", round(loaded_model.score(X_test_scaled, y_test_scaled),4))
    print("Spearman :", round(spearmanr(y_pred, y_test).correlation,4))
    print("RMSE      :", round(math.sqrt(mean_squared_error(y_test, y_pred)),0))

    if(run_linear):
        # Train and print Linear Regression
        linear_regression = LinearRegression()
        linear_regression.fit(X_train_scaled, y_train_scaled)
        y_pred_lr_scaled = linear_regression.predict(X_test_scaled) 
        y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1)).flatten()
        
        print("\n")
        print("Linear regressor")
        print("Score    :", round(linear_regression.score(X_test_scaled, y_test_scaled),4))
        print("Spearman :", round(spearmanr(y_pred_lr, y_test).correlation,4))
        print("RMSE      :", round(math.sqrt(mean_squared_error(y_test, y_pred_lr)),0), "\n")

if __name__ == "__main__":
    if len(sys.argv[1:]) != 4:
        print("The script requires 4 arguments as <model.sav> <file.csv> <include linear fit?>  <list_of_included_features>")
    else:
        make_model(sys.argv[1:])