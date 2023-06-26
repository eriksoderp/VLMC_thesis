## Predicting evolutionary distances from variable length Markov chains with deep regression
Repository for Master's Thesis: "Predicting evolutionary distances from variable length Markov chains with deep regression" at Chalmers University of Technology. 

The project is built on the code at https://github.com/Schlieplab/dvstar, however, this repository omits that code. Thus, in order to run the scripts dvstar_build.py and build_vlmcs.py, you have to have a build of the dvstar-repo. The details on how to do this are clearly explained in dvstar's README. However, if you already have data similar to what can be found in this repository, you don't need to have a build of dvstar to run the model scripts.

Throughout the project, we used the MLPRegressor architecture from Scikitlearn. We also used Scikitlearn's linear regressor for benchmarking our model. In pipe_model.py, one can see how we structured our models with a pipeline architecture for normalization and to prevent data leakage in training. 

The project was divided into two parts; Part 1, where we investigated how a model could predict synthetic mutations made on the two pathogens SARS-CoV-2 and Escherichia coli, and Part 2, where we studied if we could make models to predict evolutionary distances based on the divergence times retrieved from [TimeTree.org](https://timetree.org/). 

The code specifically for Part 1 and the datasets used can be found in the "Part1"-folder. Similarly, the code and datasets for Part 2 can be found in the "Part2"-folder. 
