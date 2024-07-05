'''
#FIX THIS
This project looks at bikeshare data from
https://raw.githubusercontent.com/arjayit/cs4432_data/master/bike_share_hour.csv
'''

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

def additional_EDA(df):
    '''
    This function takes the bike share dataframe as an argument and plots 7 additional exploratory charts, 6 bar charts and one line plot.
    '''
    seasons = ['winter', 'spring', 'summer', 'fall']
    weather_conds = ['clear/mostly clear', 'misty/overcast', 'light rain/snow', 'heavy rain/snow/fog']
    workdays = ['non-workingday', 'workingday']
    # bar plot of bike share count vs season
    # notice summer has the highest count, while winter has the lowest
    plt.figure(figsize= (8, 6), num= 1)
    labels = seasons
    sns.barplot(data= bikes, x= 'season', y= 'cnt', hue= 'season', errorbar= None)
    plt.xticks([0, 1, 2, 3], labels)
    plt.title('Bikes per Season')
    plt.xlabel('Season')
    plt.ylabel('Bikes (avg count)')
    plt.show()
    
    # bar plot for working day vs. bike share count
    # notice the increase in people renting bikes from 2011 to 2012
    plt.figure(figsize= (8, 6), num= 2)
    ax= sns.barplot(data= bikes, x= 'workingday', y= 'cnt', hue= 'yr', errorbar= None)
    plt.xticks([0, 1], workdays)
    plt.title('Bikes on Workdays vs Non-Workdays')
    plt.xlabel('Workingday')
    plt.ylabel('Bikes (avg count)')
    
    for i in ax.containers:
        ax.bar_label(i,)
    plt.show()
    
    # bar chart for month vs count
    # notice which months have the highest bike share count (June-Sept, slight dip in July)
    plt.figure(figsize= (10, 8), num= 3)
    ax = sns.barplot(data= bikes, x= 'mnth', y= 'cnt', errorbar= None)
    plt.title('Bikes per Month')
    plt.xlabel('Month')
    plt.ylabel('Bikes (avg count)')
    
    for i in ax.containers:
        ax.bar_label(i,)
    plt.show()
    
    # bar chart of weathersit vs count
    # notice that as weather get worse, bike ride share count decreases
    # plt.figure(figure= (8, 6), num= 4)
    labels = weather_conds
    g1 = sns.barplot(data= bikes, x= 'weathersit', y= 'cnt', hue= 'weathersit', errorbar= None)
    plt.xticks([0, 1, 2, 3], labels)
    plt.title('Bikes in Various Weather Conditions')
    plt.xlabel('Weather Conditions')
    plt.ylabel('Bikes (avg count)')
    for t, l in zip(g1.legend_.texts, labels):
        t.set_text(l)
    plt.show()
    
    # line plot of weathersit vs count by season
    # notice how seasons with a usual higher content of "bad weather" days have a lower overall count of bike share rentals
    plt.figure(figsize= (8,6), num= 5)
    xlabels= weather_conds
    legend_labels = seasons
    g2 = sns.pointplot(data= bikes, x= 'weathersit', y= 'cnt', hue= 'season', errorbar= None)
    plt.title('Bike Rental Weather Conditions per Season')
    plt.xticks([0, 1, 2, 3], xlabels)
    plt.xlabel('Weather Conditions')
    plt.ylabel('Bikes (avg count)')
    for t, l in zip(g2.legend_.texts, legend_labels):
        t.set_text(l)
    plt.show()
    
    # bar plot of hour vs count
    # notice that the hours of 8:00-9:00 AM and 5:00-7:00 PM are the busiest times of days
    plt.figure(figsize= (8,6), num= 6)
    sns.barplot(data= bikes, x= 'hr', y= 'cnt', errorbar= None)
    plt.title('Bike Rental per Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Bikes (avg count)')
    plt.show()
    
    # bar plot of hour vs count on weekends and workingdays
    # notice the hourly trend does change on weekends, on weekends the busies hours are between 12:00-4:00 PM
    plt.figure(figsize= (10, 6), num= 7)
    legend_labels = workdays
    g3 = sns.barplot(data= bikes, x= 'hr', y= 'cnt', hue= 'workingday', errorbar= None)
    plt.title('Bike Rental per Hour Depending on Workingday')
    plt.xlabel('Hour of Day')
    plt.ylabel('Bikes (avg count)')
    for t, l in zip(g3.legend_.texts, workdays):
        t.set_text(l)
    plt.show()
    
    plt.close('all')

print('Loading Bike Share Data')
# read in data as a pandas dataframe
url = 'https://raw.githubusercontent.com/arjayit/cs4432_data/master/bike_share_hour.csv'
bikes = pd.read_csv(url)

# convert the following columns to categorical datatype, since they are numerical values used to indicate categories
cols = ['season', 'yr', 'holiday', 'weekday', 'workingday', 'weathersit']
for col in cols:
    bikes[col] = bikes[col].astype('category')
    
print('EDA')
#check for null values
bikes.info()

#optional heatmap to show nulls/lack of nulls visually
#sns.heatmap(bikes.isnull(), cmap= 'viridis')
#plt.show()
#plt.close('all')

print('Data contains no null values, proceed with EDA')

print(bikes.describe())

# Take a closer look at months/seasons to interpret their relationship
mon_in_season = bikes.groupby('season')['mnth'].unique()
print(mon_in_season)

# according to the README file, the seasons are:
# 1) spring, 2) summer, 3) fall, 4) winter
# however, based on what we observe above, the actual seasons should be:
# 1) winter, 2) spring, 3) summer, 4) fall

# optional additional visualizations to get a feel for the data have been placed into the following function call for ease of commenting/uncommenting during runs
#additional_EDA(bikes)

print('\nData Prep')

# # optional correlation matrix: to visualize registered, casual, and count columns correlation. Temp and atemp also appear to be almost identical
# plt.figure(figsize= (8, 6), num= 8)
# sns.heatmap(bikes.corr(numeric_only= True).round(2), annot= True, vmin= -1.0, vmax= 1.0, cmap= 'bwr')
# plt.show()

# scale numerical features
columns = ['casual', 'registered']    # other numerical features (such as temp/atemp were not included because they were already normalized)
scaler = StandardScaler()
bikes_scaled = pd.DataFrame(scaler.fit_transform(bikes[columns]), index= bikes.index, columns= columns)
bikes_scaled.describe()

# replace the original columns in the dataframe
bikes['casual'] = bikes_scaled['casual']
bikes['registered'] = bikes_scaled['registered']
bikes.head()

# Drop the following columns from your dataset: casual, registered, dteday, instant.
bikes.drop(columns= ['casual', 'registered', 'dteday', 'instant'], inplace= True)
bikes.head()


# # Implement a histogram of the count column: shows distribution is right-skewed
# plt.figure(figsize= (8, 6), num= 9)

# sns.histplot(data= bikes, x= 'cnt')
# plt.title('Bike Rental Count Distribution')
# plt.xlabel('Scaled Count')
# plt.ylabel('Count')

plt.show()

print('Preparing train/test data with a test size of 33%')

# Implement a train/test split with a test size of 33%.
features_train, features_test, target_train, target_test = train_test_split(bikes.drop(columns= ['cnt']), bikes['cnt'], 
                                                                            test_size= 0.33, random_state= 42)

print('Fit Baseline Linear Regression Model with Cross Validation')
# Implement a baseline linear regression algorithm
lin_model = LinearRegression()
lin_model.fit(features_train, target_train)

# implement cross validation to get r2 and mse
lin_cv_score_r2 = cross_val_score(estimator= lin_model, X= features_train, y= target_train, scoring= 'r2')
lin_cv_score_mse = cross_val_score(estimator= lin_model, X= features_train, y= target_train, scoring= 'neg_mean_squared_error')
print(f'Avg R2 Score: {np.mean(lin_cv_score_r2)}')
print(f'MSE Score: {np.mean(-lin_cv_score_mse)}')

# calculate RMSE based on MSE
print(f'RMSE Score: {np.sqrt(np.mean(-lin_cv_score_mse))}')

print('Model Training with One-Hot-Encoded Variables')

# Create one-hot-encoded values for your categorical columns using get_dummies and add them to your source dataset.
# Drop the original categorical columns from your source datasource
dummy_bikes = pd.get_dummies(bikes, columns= ['season', 'yr', 'holiday', 'weekday', 'workingday', 'weathersit'], drop_first= True, dtype= int)
dummy_bikes.head()

print('Preparing train/test on new data with a test size of 33%')

# Do a test/train split based on your new source dataset.
X_train, X_test, y_train, y_test = train_test_split(dummy_bikes.drop(columns= ['cnt']), dummy_bikes['cnt'],
                                                                            test_size= 0.33, random_state= 42)

print('Fit Linear Regression Model with Cross Validation on New Data Set')
# Implement and fit a new linear model on the new training set
lin_model_dummy = LinearRegression()
lin_model_dummy.fit(X_train, y_train)

# implement cross validation to get r2 and mse
lin_cvs_r2 = cross_val_score(estimator= lin_model_dummy, X= X_train, y= y_train, scoring= 'r2')
lin_cvs_mse = cross_val_score(estimator= lin_model_dummy, X= X_train, y= y_train, scoring= 'neg_mean_squared_error')
r2 = np.mean(lin_cvs_r2)
mse = np.mean(-lin_cvs_mse)

# calculate RMSE based on MSE
rmse = np.sqrt(np.mean(-lin_cvs_mse))
model_info = ['Linear Regression', r2, mse, rmse]
arr = np.reshape(model_info, (1, 4))

print('Linear Regression Model:')
print(f'R2 Score: {r2}')
print(f'MSE Score: {mse}')
print(f'RMSE Score: {rmse}')

# create dataframe to store model scores
cv_scores = pd.DataFrame(arr, columns= ['model type', 'r2', 'mse', 'rmse'])

print('We will continue evaluation of other models with the new Data Set')

print('Instantiating other models')

# Implement a decision tree regressor with random_state = 0.
dec_tree = DecisionTreeRegressor(random_state= 0)
# Implement a RandomForestRegressor with random_state = 0 and n_estimators = 30.
rf_regr = RandomForestRegressor(n_estimators= 30, random_state= 0)
# Implement an SGDRegressor with max_iter = 1000 and tol = 1e-3.
sgd_reg = SGDRegressor(max_iter= 1000, tol= 1e-3, random_state= 42)
# Implement a Lasso Regressor with alpha = 0.1
lasso = Lasso(alpha= 0.1, random_state= 42)
# Implement an ElasticNetRegressor with random_state = 0.
elastic = ElasticNet(random_state= 0)
# Implement a RidgeRegressor with alpha = 0.5.
ridge = Ridge(alpha= 0.5, random_state= 42)
# Implement a BaggingRegressor.
bag_reg = BaggingRegressor(random_state= 42)

print('Fitting models')

# fit and score all the instantiated models
models = [dec_tree, rf_regr, sgd_reg, lasso, elastic, ridge, bag_reg]

for model in models:
    model.fit(X_train, y_train)

    model_cv_r2 = cross_val_score(estimator= model, X= X_train, y= y_train, scoring= 'r2')
    model_cv_mse = cross_val_score(estimator= model, X= X_train, y= y_train, scoring= 'neg_mean_squared_error')

    r2 = np.mean(model_cv_r2)
    mse = np.mean(-model_cv_mse)
    rmse = np.sqrt(np.mean(-model_cv_mse))
    model_info = [model, r2, mse, rmse]

    # print(f'{model}:')
    # print(f'R2 Score: {r2}')
    # print(f'MSE Score: {mse}')
    # print(f'RMSE Score: {rmse}')
    # print('\n')

    cv_scores.loc[len(cv_scores.index)] = model_info

# change the names of models at index 2 and 7 to more clearly reflect the model type
cv_scores.iloc[2, 0] = 'Random Forest'
cv_scores.iloc[7, 0] = 'Bagging Regression'

# convert all scores to same datatype then sort by lowest rmse
cv_scores['rmse'] = cv_scores['rmse'].apply(pd.to_numeric)
cv_scores.sort_values(by= 'rmse', inplace= True)
cv_scores

print('The top three performing models are: ', '\n', cv_scores[0:3])

print('Model Tuning with Top Three Models')

# set up the parameter grid
cv_param_grid = {'random_state': [42],
                 'max_features': [0.2, 0.4, 0.6, 0.8, 1.0],
                 }

print('Executing Grid Search for Decision Tree')
# grid search for decision tree
dt_grid_search = GridSearchCV(estimator= dec_tree,
                              param_grid= cv_param_grid,
                              cv= 5,
                              verbose = 1,
                              scoring= 'neg_mean_squared_error')

dt_grid_search.fit(X_train, y_train)

print('Executing Grid Search for Random Forest')
# grid search for random forest
rf_grid_search = GridSearchCV(estimator= rf_regr,
                              param_grid= cv_param_grid,
                              cv= 5,
                              verbose = 1,
                              scoring= 'neg_mean_squared_error')

rf_grid_search.fit(X_train, y_train)

print('Executing Grid Search for Bagging Regressor')
# grid search for bagging regressor
br_grid_search = GridSearchCV(estimator= bag_reg,
                              param_grid= cv_param_grid,
                              cv= 5,
                              verbose = 1,
                              scoring= 'neg_mean_squared_error')

br_grid_search.fit(X_train, y_train)

# print the estimator, its best resulting parameters, and the best resulting score
print(f'Best Decision Tree estimator parameters: {dt_grid_search.best_params_}')
print(f'Best RMSE Score: {np.sqrt(-dt_grid_search.best_score_)}')
print('\n')
print(f'Best Random Forrest estimator parameters: {rf_grid_search.best_params_}')
print(f'Best RMSE Score: {np.sqrt(-rf_grid_search.best_score_)}')
print('\n')
print(f'Best Bagging estimator parameters: {br_grid_search.best_params_}')
print(f'Best RMSE Score: {np.sqrt(-br_grid_search.best_score_)}')
print('\n')

print('Further Tuning with Top Performing Model: Random Forest')

# set up the parameter grid
rand_param_grid = {'bootstrap': [True, False],
                   'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                   'max_features': [1.0, 'sqrt'],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'n_estimators': [int(x) for x in np.linspace(200, 2000, num = 10)],
                  }

# set up randomized search
rand_grid_search = RandomizedSearchCV(estimator= rf_regr,
                                      param_distributions= rand_param_grid,
                                      n_iter= 20,
                                      n_jobs= 1,
                                      cv= 3,
                                      verbose = 1,
                                      scoring= 'neg_mean_squared_error',
                                      random_state= 42,
                                     )

# evaluate randomized grid search
rand_grid_search.fit(X_train, y_train)

# Take your best_estimator_ and see how it compares by doing cross_vals for r2, mse, and calculating rmse.
best_rf_model = rand_grid_search.best_estimator_

brf_cv_r2 = cross_val_score(estimator= best_rf_model, X= X_train, y= y_train, scoring= 'r2')
brf_cv_mse = cross_val_score(estimator= best_rf_model, X= X_train, y= y_train, scoring= 'neg_mean_squared_error')

r2 = np.mean(brf_cv_r2)
mse = np.mean(-brf_cv_mse)
rmse = np.sqrt(np.mean(-brf_cv_mse))

print('Best Random Forest Regressor Model:')
print('\n')
print(f'R2 Score: {r2}')
print(f'MSE Score: {mse}')
print(f'RMSE Score: {rmse}')
print('\n')

print('Best Random Forest Regressor Model Parameters:')
print(f'{rand_grid_search.best_params_}')

# make predictions
best_rf_preds = best_rf_model.predict(X_test)
print('Bike count model predictions: ', best_rf_preds)

# calculate RMSE on test set
np.sqrt(mean_squared_error(y_test, best_rf_preds))