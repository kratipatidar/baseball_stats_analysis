Baseball is one of the most popular games in the US. So naturally, it has a plethora of statistics associated with it. There are various batting, pitching and fielding statistics, that can help determine the result of a game to a high degree of certainty. 

The aim of this project is to utilize the baseball dataset, and come up with new features that would help in predicting whether the home team wins a particular match or not. 

For achieving this, first a good amount of feature engineering was carried out on the data, to come up with predictors. Next, different kinds of plots were obtained to gain an in-depth understanding of how the predictor interacts with the response. Added to this, the interaction between different predictors was also observed through these plots to help segregate predictors that add value to our prediction analysis from predictors that hamper our predictions.

After this, several classification models were fit on the data, and accuracy scores were obtained on these fits. A train-test split was done to test the model performance on unseen data.

Lastly, all these steps were wrapped into a docker container, which made up the production phase of this project.