# Applied-Machine-Learning
This repository contains machine learning pipelines from a course in my graduate program.

## Environment and Requirements

- Install [Anaconda](https://www.anaconda.com/download)
- Run `conda env create -f environment.yml` to create an environment with the necessary dependencies.
- NOTE: if using an Apple Silicon GPU, update the pytorch installation to the pytorch-nightly version after activating the new environment with the following commands.
    - `conda activate applied_machine_learning`
    - `conda install pytorch torchvision torchaudio -c pytorch-nightly`

## Notebook Summaries

- `Suicide-Rate-Classification`: this notebook evaluates a decision tree, multilayer perceptron, and random forest for the problem of predicting high vs. low suicide rates as a classification problem.
- `Suicide-Rate-Regression`: this notebook applies linear regression to the problem of predicting an individual's suicide rate. The regression model worked better for this problem than the classification approach. 
- `Fake-News-Classification`: this notebook evaluates a decision tree, multilayer perceptron, and random forest for the problem of fake news classification. The pipeline evaluates the models using ROC curves.
- `Species-Clustering`: this notebook applies K-Means clustering and DBSCAN to remove anomalies in a synthetic species dataset.
- `Heart-Failure-Classification`: this notebook tests various ensemble methods for the problem of heart failure classification. The notebook compares single models (Naive Bayes, Decision Tree, Multilayer perceptron) to ensembles of weak classifiers for each model. The notebook also compares these ensembles to a random forest classifier.
- `Cybersecurity-Intrusion-Detection`: this notebook contains a cybersecurity intrusion detection pipeline to classify various cyber attacks.
- `Titanic-Survivor-Classification`: this notebook applies various machine learning classifiers to the problem of survivor prediction for the Titanic. The notebook tested a support vector machine, multilayer perceptron, and random forest. Grid search is applied to hyperparameter tuning.
- `Apriori-Analysis-and-Custom-ANN`: this notebook demonstrates how to transform data to use with the Weka framework for Apriori analysis. This notebook also extends a custom artificial neural network to support any number of hidden layers.
- `Credit_Fraud_Detection`: this notebook applies various models (decision tree, multilayer perceptron, support vector machine, random forest, and pytorch neural network) with and without regularization to the problem of credit fraud detection.
- `Image-classification-NN` and `Image-Clasification-CNN`: these notebooks compare the performance of a fully connected neural network to a convolutional neural network for the problem of image classification. The `Image-Classification-NN` notebook outlines how to optimize the neural network by adding convolutional layers. The `Image-Classification-CNN` notebook implements the model outlined. 
