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


def make_model(argv):
    csv_path = argv[0]
    number_of_cores = int(argv[1])
    number_of_kfolds = int(argv[2])
    dataset_name = str(argv[3])
    X_args = argv[4].split(',')

    print(X_args)

    # Load the .csv file into a DataFrame
    data = pd.read_csv(csv_path)

    ### ÄNDRA DESSA
    max_depth_values = [9,12]
    min_count_values = [25,100]
    threshold_values = [3.9075]
    hidden_layer_sizes_values = [(2,2,2),(4,4)]
    
    # Filter the rows
    data = data[data['max_depth'].isin(max_depth_values)]
    data = data[data['min_count'].isin(min_count_values)]
    data = data[data['threshold'].isin(threshold_values)]

    X = data[X_args].fillna(0)
    y = data["evo dist"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features using a Scaler
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Scale the target variable
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    parameter_space = {
        'hidden_layer_sizes': hidden_layer_sizes_values,
        'activation': ['relu'],
        'solver': ['adam'],
        #'alpha': [0.0001, 0.01],
        #'learning_rate': ['constant','adaptive'],
    }

    # Train and print MLP Regression
    mlp_regressor = MLPRegressor()
    grid_search = GridSearchCV(
        mlp_regressor, 
        parameter_space, 
        n_jobs=number_of_cores, 
        cv=number_of_kfolds,
        verbose=3
    )
    
    grid_search.fit(X_train_scaled, y_train_scaled)

    best_mlp = MLPRegressor(**grid_search.best_params_)
    best_mlp.fit(X_train_scaled, y_train_scaled)

    y_pred_scaled = best_mlp.predict(X_test_scaled)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


    print("Best Parameters:", grid_search.best_params_)
    print("\n")

    print("MLPRegressor")
    print("Score    :", round(best_mlp.score(X_test_scaled, y_test_scaled),4))
    print("Spearman :", round(spearmanr(y_pred_mlp, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_mlp),4))

    # Train and print Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_scaled, y_train_scaled)
    y_pred_lr_scaled = linear_regression.predict(X_test_scaled) 
    y_pred_lr = scaler_y.inverse_transform(y_pred_lr_scaled.reshape(-1, 1)).flatten()
    
    print("\n")
    print("Linear regressor")
    print("Score    :", round(linear_regression.score(X_test_scaled, y_test_scaled),4))
    print("Spearman :", round(spearmanr(y_pred_lr, y_test).correlation,4))
    print("RMSE      :", round(mean_squared_error(y_test, y_pred_lr),4), "\n")

    # Save the model
    if not os.path.exists('./models'):
        os.makedirs('./models')

    best_hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes']
    hl_sizes_str = '-'.join(str(s) for s in best_hidden_layer_sizes)
    max_depth_str = '-'.join(str(md) for md in max_depth_values)
    min_count_str = '-'.join(str(mc) for mc in min_count_values)
    threshold_str = '-'.join(f"{t:.4f}" for t in threshold_values)

    data_file_name = f"./models/model_{dataset_name}_max_depth_{max_depth_str}_min_count_{min_count_str}_threshold_{threshold_str}_hidden_layer_sizes_{hl_sizes_str}_cv_{number_of_kfolds}.sav"
    pickle.dump(best_mlp, open(data_file_name, 'wb'))

    # Save scalers
    if not os.path.exists('./scalers'):
        os.makedirs('./scalers')

    scaler_x_file_name = f"./scalers/scaler_x_{dataset_name}_max_depth_{max_depth_str}_min_count_{min_count_str}_threshold_{threshold_str}_hidden_layer_sizes_{hl_sizes_str}_cv_{number_of_kfolds}.sav"
    scaler_y_file_name = f"./scalers/scaler_y_{dataset_name}_max_depth_{max_depth_str}_min_count_{min_count_str}_threshold_{threshold_str}_hidden_layer_sizes_{hl_sizes_str}_cv_{number_of_kfolds}.sav"

    pickle.dump(scaler_x, open(scaler_x_file_name, 'wb'))
    pickle.dump(scaler_y, open(scaler_y_file_name, 'wb'))

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

        if not os.path.exists('./plots'):
            os.makedirs('./plots')

        #plot_file_name = f"./plots/plot_{str(dataset_name)}_layers_{str(best_hidden_layer_sizes)}_max-depth_{str(max_depth_values)}_min-count_{str(min_count_values)}_threshold_{str(threshold_values)}_cv_{str(number_of_kfolds)}.png"
        plot_file_name = f"./plots/plot_{dataset_name}_max_depth_{max_depth_str}_min_count_{min_count_str}_threshold_{threshold_str}_hidden_layer_sizes_{hl_sizes_str}_cv_{number_of_kfolds}.png"
        plt.savefig(plot_file_name)
        #plt.show()


if __name__ == "__main__":
    if len(sys.argv[1:]) != 5:
        print("The script requires 5 arguments as <file.csv> <number_of_cores> <number_of_kfolds> <dataset_name> <list_of_included_features>")
    else:
        make_model(sys.argv[1:])