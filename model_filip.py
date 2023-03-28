import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV
import pickle
import sys
import matplotlib.pyplot as plt

def make_model(argv):
    csv_path = argv[0]
    number_of_cores = int(argv[1])
    number_of_kfolds = int(argv[2])
    model_filename = argv[3]
    X_args = argv[4].split(',')

    print(X_args)

    # Load the .csv file into a DataFrame
    data = pd.read_csv(csv_path)

    # Filter the rows
    #data = data[data['max_depth'] == 9]
    #data = data[data['min_count'] == 100]
    #data = data[data['threshold'] == 3.9075]

    # Prepare the data
    # ["VLMC dist", "threshold", "min_count", "max_depth"]
    X = data[X_args].fillna(0)
    y = data["Evolutionary dist"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """parameter_space = {
        'hidden_layer_sizes': [(200,200), (500,500), (1000,500), (2000,500), (500,500,500)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant','adaptive'],
    }"""

    parameter_space = {
        'hidden_layer_sizes': [(400,200,100)],
    }

    # Train and print MLP Regression
    mlp_regressor = MLPRegressor()
    grid_search = GridSearchCV(
        mlp_regressor, 
        parameter_space, 
        n_jobs=number_of_cores, 
        cv=number_of_cores,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)

    best_mlp = grid_search.best_estimator_
    y_pred_mlp = best_mlp.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)

    print("\n", "MLPRegressor")
    print("Score    :", round(best_mlp.score(X_test, y_test),4))
    print("Spearman :", round(spearmanr(y_pred_mlp, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_mlp, squared=False),4))

    # Train and print Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred_lr = linear_regression.predict(X_test)

    print("\n", "Linear regressor")
    print("Score    :", round(linear_regression.score(X_test, y_test),4))
    print("Spearman :", round(spearmanr(y_pred_lr, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_lr, squared=False),4), "\n")

    # Save the model
    pickle.dump(best_mlp, open(model_filename, 'wb'))

    if(1):
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot actual vs predicted values for MLPRegressor
        ax1.scatter(y_pred_mlp, y_test, alpha=0.5, color='b')
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Actual Values")
        ax1.set_title("Actual vs Predicted Values (MLPRegressor)")

        # Plot actual vs predicted values for LinearRegressor
        ax2.scatter(y_pred_lr, y_test, alpha=0.5, color='r')
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Actual Values")
        ax2.set_title("Actual vs Predicted Values (LinearRegressor)")

        # Show the plots
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv[1:]) != 5:
        print("The script requires 5 arguments as <file.csv> <number_of_cores> <number_of_kfolds> <model_filename> <list_of_included_features>")
    else:
        make_model(sys.argv[1:])