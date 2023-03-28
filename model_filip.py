import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV

# Load the .csv file into a DataFrame
data = pd.read_csv("ecoli_65534_distances.csv")

# Filter the rows
data = data[data['max_depth'] == 9]
data = data[data['min_count'] == 100]
data = data[data['threshold'] == 3.9075]
print(data)

# Prepare the data
X = data[["VLMC dist", "threshold", "min_count", "max_depth"]].fillna(0)
y = data["Evolutionary dist"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the MLPRegressor model
# mlp_regressor = MLPRegressor(hidden_layer_sizes=2000)
# mlp_regressor.fit(X_train, y_train)

# Use GridSearchCV to find optimal parameters
param_grid = {
    'hidden_layer_sizes': [(2000,500)],
    #'activation': ['relu', 'tanh'],
    #'solver': ['adam', 'lbfgs'],
    #'max_iter': [500, 1000, 1500]
}

grid_search = GridSearchCV(
    estimator=MLPRegressor(),
    param_grid=param_grid,
    #cv=3,
    #n_jobs=number_of_cores,
    verbose=True
)

#grid_search.fit(X_train, y_train)

print('Best Hyperparameters:', grid_search.best_params_)
print('Best Score:', grid_search.best_score_)

# Make and fit model with best parameters
best_mlp = MLPRegressor(**grid_search.best_params_)
best_mlp.fit(X_train, y_train)

# Predict the test set results
y_pred_mlp = grid_search.predict(X_test)

# Performance regression
print("\n", "MLPRegressor")
print("Score    :", round(grid_search.score(X_test, y_test),4))
print("Spearman :", round(spearmanr(y_pred_mlp, y_test).correlation,4))
print("MSE      :", round(mean_squared_error(y_test, y_pred_mlp),4))

# Train Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predict Linear Regression
y_pred_lr = linear_regression.predict(X_test)

# Performance linear
print("\n", "Linear regressor")
print("Score    :", round(linear_regression.score(X_test, y_test),4))
print("Spearman :", round(spearmanr(y_pred_lr, y_test).correlation,4))
print("MSE      :", round(mean_squared_error(y_test, y_pred_lr),4), "\n")
