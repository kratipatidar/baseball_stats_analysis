## Description of Project
This project is rooted in Machine Learning. The project aims to predict whether the home team wins (in baseball) using various classification algorithms. Once the desired accuracy for the said prediction is obtained, the code files are wrapped in a docker container to automate predictions.

## Workflow
1. Firstly, a python script was created to analyze the categorical and continuous features of the input dataset. This analysis helped determine which features would contribute to accurate predictions, thus leading to elimination of features with a high potential of skewing analysis results.
2. Next, different machine learning classification models like Random Forest, XG Boost, Support Vector Machine, etc were fit to the training dataset. Predictions were obtained on the test dataset (not used for model fiiting), and the prediction accuracies of different models were compared.
3. Finally, a docker script was created to automate visulaizations and predictions. In this way, this project aimed to create a running prediction service.
