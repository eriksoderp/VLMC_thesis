## Predicting evolutionary distances from variable length Markov chains with deep regression
Repository for Master's Thesis: "Predicting evolutionary distances from variable length Markov chains with deep regression" at Chalmers University of Technology. 

Throughout the project, we used the MLPRegressor architecture from Scikitlearn. We also used Scikitlearn's linear regressor for benchmarking our model. In pipe_model.py, one can see how we structured our models with a pipeline architecture for normalization and to prevent data leakage in training. 

The project was divided into two parts; Part 1, where we investigated how a model could predict synthetic mutations made on the two pathogens SARS-CoV-2 and Escherichia coli, and Part 2, where we studied if we could make models to predict evolutionary distances based on the divergence times retrieved from TimeTree.org. 

The code specifically for Part 1 and the datasets used can be found in the "Part 1"-folder. Similarly, the datasets for Part 2 can be found in the "Part 2"-folder. 
