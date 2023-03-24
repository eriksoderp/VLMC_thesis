import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import sys
import pickle
import argparse


def make_model(argv):
    csv_path = argv[0]
    number_of_cores = argv[1]
    number_of_kfolds = argv[2]
    model_filename = argv[3]
    # Read csv from input
    data = pd.read_csv(csv_path)

    # Create feature set and drop NaNs
    X = data.drop(['Evolutionary dist', 'Unnamed: 0'], axis=1)
    X['VLMC dist'] = X['VLMC dist'].fillna(0)
    
    # Create ground truth set
    y = data['Evolutionary dist']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use GridSearchCV to find optimal parameters
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'max_iter': [500, 1000, 1500]
    }

    grid_search = GridSearchCV(
        estimator=MLPRegressor(),
        param_grid=param_grid,
        cv=number_of_kfolds,
        n_jobs=number_of_cores,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print('Best Hyperparameters:', grid_search.best_params_)
    print('Best Score:', grid_search.best_score_)

    # Make and fit model with best parameters
    best_mlp = MLPRegressor(**grid_search.best_params_)
    best_mlp.fit(X_train, y_train)

    # Make predictions with best model
    y_pred = best_mlp.predict(X_test)

    print(f"Score: {best_mlp.score(X_test, y_test)}")
    print(f"Spearman R: {spearmanr(y_pred, y_test)}")
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")

    # Save the model
    pickle.dump(best_mlp, open(model_filename, 'wb'))

    """
    To retrieve model and use it, make a script with the following code

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print(result)
    """



if __name__ == "__main__":
    if len(sys.argv[1:]) != 4:
        print("The script requires 4 arguments as <file.csv> <number_of_cores> <number_of_kfolds> <model_filename>")
    else:
        make_model(sys.argv[1:])
