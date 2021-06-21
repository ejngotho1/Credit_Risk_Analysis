# Credit_Risk_Analysis

## Overview of Project

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we'll need to employ different techniques to train and evaluate models with unbalanced classes. We are tasked with the use of `imbalanced-learn` and `scikit-learn` libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we’ll oversample the data using the `RandomOverSampler` and `SMOTE` algorithms, and undersample the data using the `ClusterCentroids` algorithm. Then, we’ll use a combinatorial approach of over and undersampling using the `SMOTEENN` algorithm. Next, compare two new machine learning models that reduce bias, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk. Once we’re done, we’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Deliverables:
This new assignment consists of three technical analysis deliverables and a written report.

1. ***Deliverable 1:*** Use Resampling Models to Predict Credit Risk
2. ***Deliverable 2:*** Use the SMOTEENN Algorithm to Predict Credit Risk
3. ***Deliverable 3:*** Use Ensemble Classifiers to Predict Credit Risk
4. ***Deliverable 4:*** A Written Report on the Credit Risk Analysis [README.md]


## Deliverables:
Data Sources

* Data Source: ` Module-17-Challenge-Resources.zip` and `LoanStats_2019Q1.csv`
* Data Tools:  `credit_risk_resampling_starter_code.ipynb` and `credit_risk_ensemble_starter_code.ipynb`.
* Software: `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas`

### Supervised Machine Learning and Credit Risk
#### Predicting Credit Risk

##### Create a Machine Learning Environment

Your new virtual environment will use Python 3.7 and accompanying Anaconda packages. After creating the new virtual environment, you'll install the imbalanced-learn library in that environment.

**NOTE**
Consult the [imbalanced-learn documentation](https://imbalanced-learn.readthedocs.io/en/stable/) for additional information about the imbalanced-learn library.

Check out the Windows OS instructions below:

##### Windows Setup
Before we create a new environment in Windows, we'll need to update the global conda environment:

1. Launch the Anaconda Prompt, or open your PythonData Anaconda Prompt and deactivate this environment.

2. Update the global conda environment by typing conda update conda and press Enter

3. After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

4. In the command line, type conda create -n mlenv python=3.7 anaconda.

5. After all the packages are collected, you'll see the prompt Proceed ([y]/n)?. Press the "Y" key (for "yes") and press Enter.

6. Activate your mlenv environment by typing conda activate mlenv and press Enter, or open your Anaconda Prompt (mlenv).

##### Check Dependencies for the imbalanced-learn Package
Before we install the imbalanced-learn package, we need to confirm that all of the package dependencies are satisfied in our mlenv environment:

* NumPy, version 1.11 or later
* SciPy, version 0.17 or later
* Scikit-learn, version 0.21 or later

# Deliverable 1:  
## Use Resampling Models to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling `RandomOverSampler` and `SMOTE` algorithms, and then you’ll use the undersampling `ClusterCentroids` algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.


**Follow the instructions below:**

Follow the instructions below and use the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 1.

Open the `credit_risk_resampling_starter_code.ipynb` file, rename it `credit_risk_resampling.ipynb`, and save it to your **Credit_Risk_Analysis** folder.

Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:

* Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
* Create the target variables.
* Check the balance of the target variables.

Next, begin resampling the training data. First, use the oversampling `RandomOverSampler` and `SMOTE` algorithms to resample the data, then use the undersampling `ClusterCentroids` algorithm to resample the data. For each resampling algorithm, do the following:

* Use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
* Calculate the accuracy score of the model.
* Generate a confusion matrix.
* Print out the imbalanced classification report.


#### Deliverable 1 Requirements

For all three algorithms, the following have been completed:
- An accuracy score for the model is calculated
- A confusion matrix has been generated
- An imbalanced classification report has been generated 

# Deliverable 2:  
## Use the SMOTEENN algorithm to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imbalanced-learn` and `scikit-learn` libraries, you’ll use a combinatorial approach of over and undersampling with the `SMOTEENN` algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the `SMOTEENN` algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.


**Follow the instructions below:**

Follow the instructions below and use the information in the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 2.

1. Continue using your `credit_risk_resampling.ipynb` file where you have already created your training and target variables.
2. Using the information we have provided in the starter code, resample the training data using the `SMOTEENN` algorithm.
3. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
4. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.


#### Deliverable 2 Requirements

The combinatorial SMOTEENN algorithm does the following:
- An accuracy score for the model is calculated
- A confusion matrix has been generated
- An imbalanced classification report has been generated 

# Deliverable 3:  
## Use Ensemble Classifiers to Predict Credit Risk 
### Deliverable Requirements:

Using your knowledge of the `imblearn.ensemble` library, you’ll train and compare two different ensemble classifiers, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier`, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.


**Follow the instructions below:**

Follow the instructions below and use the information in the `credit_risk_resampling_starter_code.ipynb` file to complete Deliverable 3.

1. Open the `credit_risk_ensemble_starter_code.ipynb` file, rename it `credit_risk_ensemble.ipynb`, and save it to your **Credit_Risk_Analysis** folder.
2. Using the information we have provided in the starter code, create your training and target variables by completing the following:
    - Create the training variables by converting the string values into numerical ones using the `get_dummies()` method.
    - Create the target variables.
    - Check the balance of the target variables.
3. Resample the training data using the `BalancedRandomForestClassifier` algorithm with 100 estimators.
    - Consult the following [Random Forest documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) for an example.

4. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
5. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
6. Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
7. Next, resample the training data using the `EasyEnsembleClassifier` algorithm with 100 estimators.
    - Consult the following [Easy Ensemble documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html) for an example.
8. After the data is resampled, use the `LogisticRegression` classifier to make predictions and evaluate the model’s performance.
9. Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

Save your `credit_risk_ensemble.ipynb` file to your **Credit_Risk_Analysis** folder.


#### Deliverable 3 Requirements

The `BalancedRandomForestClassifier` algorithm does the following:
- An accuracy score for the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated 
- The features are sorted in descending order by feature importance

The `EasyEnsembleClassifier` algorithm does the following:
- An accuracy score of the model is calculated 
- A confusion matrix has been generated 
- An imbalanced classification report has been generated  

### DELIVERABLE RESULTS:
Below are the results from the various techniques used to predictive model for High-Risk loans.

## SMOTEEN

![image](https://user-images.githubusercontent.com/57301554/122705162-c883de00-d21a-11eb-9692-6623594d3b5c.png)
![image](https://user-images.githubusercontent.com/57301554/122705200-da658100-d21a-11eb-9766-b80fe80083e6.png)

## SMOTE

![image](https://user-images.githubusercontent.com/57301554/122705296-0a148900-d21b-11eb-8160-df5a9eacc531.png)
![image](https://user-images.githubusercontent.com/57301554/122705338-1ef11c80-d21b-11eb-8287-992ccbe29fb4.png)

## RandomOverSampler

![image](https://user-images.githubusercontent.com/57301554/122705576-a76fbd00-d21b-11eb-85a8-7d47890a10f0.png)
![image](https://user-images.githubusercontent.com/57301554/122705613-b5bdd900-d21b-11eb-9df4-284a43521f08.png)

## ClusterCentroids

![image](https://user-images.githubusercontent.com/57301554/122705731-f0277600-d21b-11eb-8665-7c23fba11fbe.png)
![image](https://user-images.githubusercontent.com/57301554/122705763-fc133800-d21b-11eb-9643-6f7aec64b726.png)

## EasyEnsembleClassifier

![image](https://user-images.githubusercontent.com/57301554/122705996-7774e980-d21c-11eb-91c7-4888d9d9ac91.png)

## BalancedRandomForestClassifier

![image](https://user-images.githubusercontent.com/57301554/122706055-a9864b80-d21c-11eb-9454-36f7f079a351.png)

# Summary

EasyEnsembleClassifier is the most effective because it provides highest Score for all Risk loans. The precision is low or none for all the models. Using EasyEnsembleClassifier will perform a High-Risk loan precision as a great value for the overall analysis.








