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
from matplotlib import rc, rcParams


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
    data = data[data['max_depth'] == 12]
    data = data[data['min_count'].isin([100])]
    data = data[data['threshold'].isin([3.9075])]

    # Prepare the data
    # ["VLMC dist", "threshold", "min_count", "max_depth"]
    X = data[X_args].fillna(0)
    y = data["evo dist"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features using a Scaler
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    parameter_space = {
        'hidden_layer_sizes': [(500),(500,500),(500,500,500),(500,500,500,500),(500,500,500,500,500)],
        'activation': ['relu'],
        'solver': ['adam'],
        #'alpha': [0.0001, 0.01],
        #'learning_rate': ['constant','adaptive'],
    }

    """parameter_space = {
        'hidden_layer_sizes': [(200,200,200), (500,500,500), (1000,1000,1000)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.01],
        'learning_rate': ['constant','adaptive'],
    }
    
    parameter_space = {
        'hidden_layer_sizes': [(500,500,500)],
        'activation': ['relu'],
        'early_stopping': [True],
        'solver': ['adam']
    }"""

    # Train and print MLP Regression
    mlp_regressor = MLPRegressor()
    grid_search = GridSearchCV(
        mlp_regressor, 
        parameter_space, 
        n_jobs=number_of_cores, 
        cv=number_of_kfolds
    )
    
    grid_search.fit(X_train_scaled, y_train)

    best_mlp = MLPRegressor(**grid_search.best_params_)
    best_mlp.fit(X_train_scaled, y_train)

    y_pred_mlp = best_mlp.predict(X_test_scaled)
    print("Best Parameters:", grid_search.best_params_)
    print("\n")

    print("MLPRegressor")
    print("Score    :", round(best_mlp.score(X_test_scaled, y_test),4))
    print("Spearman :", round(spearmanr(y_pred_mlp, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_mlp),4))

    # Train and print Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train)
    y_pred_lr = linear_regression.predict(X_test_scaled)
    
    print("\n")
    print("Linear regressor")
    print("Score    :", round(linear_regression.score(X_test_scaled, y_test),4))
    print("Spearman :", round(spearmanr(y_pred_lr, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_lr),4), "\n")

    # Save the model
    pickle.dump(best_mlp, open(model_filename, 'wb'))

    if(1):

        # Colors
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Set font to "LATEX"
        rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
        plt.rcParams.update()
        
        # Plot actual vs predicted values for MLPRegressor
        ax1.scatter(y_pred_mlp, 
                    y_test, 
                    alpha=1, 
                    color='royalblue', 
                    marker=".",
                    s=40, 
                    #linewidths=1, 
                    #edgecolors='w'
        )        
        ax1.set_xlabel("Predicted values")
        ax1.set_ylabel("Actual values")
        ax1.set_title("MLPRegressor - Actual vs Predicted values ")
        # 1:1 line
        ax1.axline((0,0), slope=1, color='firebrick', linestyle='dashed', linewidth=2)
                # Axis limits
        #ax1.set(xlim=(0,0.008), ylim=(0,0.008))
        # Grid line
        ax1.set_axisbelow(True)
        ax1.grid(color='gray', linestyle='dashed', linewidth=0.5)

        # Plot actual vs predicted values for LinearRegressor
        ax2.scatter(y_pred_lr, 
                    y_test, alpha=1, 
                    color='darkgrey', 
                    marker=".",
                    s=40,
                    #linewidths=1, 
                    #edgecolors='w'
        )        
        ax2.set_xlabel("Predicted values")
        ax2.set_ylabel("Actual values")
        ax2.set_title("LinearRegressor - Actual vs Predicted values ")
        # 1:1 line
        ax2.axline((0,0), slope=1, color='firebrick', linestyle='dashed', linewidth=2)
                # Axis limits
        #ax2.set(xlim=ax1.get_xlim(), ylim=ax1.get_ylim())
        # Grid line
        ax2.set_axisbelow(True)
        ax2.grid(color='gray', linestyle='dashed', linewidth=0.5)

        plt.savefig("fig.png"))
        #plt.show()


if __name__ == "__main__":
    if len(sys.argv[1:]) != 5:
        print("The script requires 5 arguments as <file.csv> <number_of_cores> <number_of_kfolds> <model_filename> <list_of_included_features>")
    else:
        make_model(sys.argv[1:])