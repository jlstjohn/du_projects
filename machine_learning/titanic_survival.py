# Python file implementaion of Titanic Dataset Exploration
# Complete EDA was done in a separate Jupyter Notebook file found here:
# https://github.com/jlstjohn/du_projects/blob/main/machine_learning/titanicSurvival.ipynb
# This will focus more on data cleaning/training/testing

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# load in full dataset
print('Getting dataset')
titanic = sns.load_dataset('titanic')

# features we want to keep, including the target ('survived')
features = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embark_town', 'alone']

# create dataframe of the target and features we are interested in
data = pd.DataFrame(data= titanic[features])
#print(data.head())

print('Cleaning/Preprocessing Data')
# impute age using the median of each passenger class for NaN values
data = data.fillna({'age': data.groupby('pclass').age.transform('median')})
#print(data.info())

# drop remaining records with null values (2 records)
data = data.dropna(axis= 0)

# convert to categorical datatypes
categories = ['pclass', 'sex', 'sibsp', 'parch', 'embark_town', 'alone']
data_dummies = pd.get_dummies(data, columns= categories, drop_first= True)

print('Creating Training and Test Sets')
# create feature and target set
X = pd.DataFrame(data_dummies.drop(labels= ['survived'], axis= 1))
y = pd.DataFrame(data= data_dummies, columns= ['survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 42, stratify= y.survived)

# instantiate models
print('Begin model training')

# implement a logistic regression model
logreg_model = LogisticRegression(max_iter= 1000)
logreg_model_preds = cross_val_predict(estimator= logreg_model, X= X_train, y= y_train.survived, cv= 5, method= 'predict_proba')

# implement an sgd classifier
sgd_clf = SGDClassifier()
sgd_clf_preds = cross_val_predict(estimator= sgd_clf, X= X_train, y= y_train.survived, cv= 5, method= 'decision_function')

# implement a suport vector classifier w/scaled data
svm_clf_scaled = \
Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', svm.SVC(probability= True))
])
svm_clf_preds= cross_val_predict(estimator= svm_clf_scaled, X= X_train, y= y_train.survived, cv= 5, method= 'predict_proba')

print('\n')
print('*** Logistic Regression ***')
print('\nConfusion Matrix:')
print(confusion_matrix(y_true= y_train, y_pred= logreg_model_preds[:, 1] >= 0.5))
print('\nClassification Report:')
print(classification_report(y_true= y_train, y_pred= logreg_model_preds[:, 1] >= 0.5))
print('\nROC-AUC Score:')
print(roc_auc_score(y_true= y_train, y_score= logreg_model_preds[:, 1]))
print('\n')
fpr, tpr, thresh = roc_curve(y_true= y_train, y_score= logreg_model_preds[:, 1])
plt.figure(figsize= (8, 8), num= 1)
plt.plot(fpr, tpr, linewidth= 2)
plt.plot([(0, 0), (1, 1)], 'k--')
plt.title('Logistic Regression ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('\n')

print('*** SGD Classifier ***')
print('\nConfusion Matrix:')
print(confusion_matrix(y_true= y_train, y_pred= sgd_clf_preds > 0))
print('\nClassification Report:')
print(classification_report(y_true= y_train, y_pred= sgd_clf_preds > 0))
print('\nROC-AUC Score:')
print(roc_auc_score(y_true= y_train, y_score= sgd_clf_preds))
fpr, tpr, thresh = roc_curve(y_true= y_train, y_score= sgd_clf_preds)
print('\n')
plt.figure(figsize= (8, 8), num= 2)
plt.plot(fpr, tpr, linewidth= 2)
plt.plot([(0, 0), (1, 1)], 'k--')
plt.title('SGD Classifier ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('\n')

print('*** Support Vector Classifier ***')
print('\nConfusion Matrix:')
print(confusion_matrix(y_true= y_train, y_pred= svm_clf_preds[:, 1] >= 0.5))
print('\nClassification Report:')
print(classification_report(y_true= y_train, y_pred= svm_clf_preds[:, 1] >= 0.5))
print('\nROC-AUC Score:')
print(roc_auc_score(y_true= y_train, y_score= svm_clf_preds[:, 1]))
print('\n')
fpr, tpr, thresh = roc_curve(y_true= y_train, y_score= svm_clf_preds[:, 1])
plt.figure(figsize= (8, 8), num= 3)
plt.plot(fpr, tpr, linewidth= 2)
plt.plot([(0, 0), (1, 1)], 'k--')
plt.title('Support Vector ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('\n')

plt.close('all')

# tune SVM classifier to see if we can get closer to log reg model
print('Performing a grid search on SVM classifier')

# set up the grid
grid = {'svm_clf__kernel': ['rbf'],
        'svm_clf__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'svm_clf__C': [1, 10, 50, 100, 200, 300]
       }

# set up the search
grid_search = \
GridSearchCV(estimator= svm_clf_scaled,
             param_grid= grid,
             cv= 5,
             verbose= 5,
             scoring= 'roc_auc')

# perform the grid search
grid_search.fit(X_train, y_train.survived)

# print best estimator, its parameters and the resulting score
print(f'Best estimator: {grid_search.best_estimator_}')
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC-AUC score: {grid_search.best_score_}')

# apply this estimator to your test set
best_svm_clf_scaled_preds = grid_search.predict_proba(X_test)[:, 1]

print('*** Best Support Vector Classifier w/Test Data ***')
print('\nConfusion Matrix:')
print(confusion_matrix(y_true= y_test, y_pred= best_svm_clf_scaled_preds >= 0.5))
print('\nClassification Report:')
print(classification_report(y_true= y_test, y_pred= best_svm_clf_scaled_preds >= 0.5))
print('\nROC-AUC Score:')
print(roc_auc_score(y_true= y_test, y_score= best_svm_clf_scaled_preds))
print('\n')
fpr, tpr, thresh = roc_curve(y_true= y_test, y_score= best_svm_clf_scaled_preds)
plt.figure(figsize= (8, 8), num= 4)
plt.plot(fpr, tpr, linewidth= 2)
plt.plot([(0, 0), (1, 1)], 'k--')
plt.title('Best Support Vector w/Test Data ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('\n')

plt.close()

# see github for output