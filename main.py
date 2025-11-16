import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings

#File path for the dataset
data_file = 'steel.csv'

#Column names in the dataset
column_names = [
    'normalising_temperature', 'tempering_temperature', 'percent_silicon',
    'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur',
    'percent_carbon', 'percent_manganese', 'tensile_strength'
]

#Target variable to predict
target_variable = 'tensile_strength'

#Setting up 10-fold cross-validation
CV_FOLDS = KFold(n_splits=10, shuffle=True, random_state=42)

#Defining scoring metrics since the desired metric is RMSE and R^2
#For RMSE, 'neg_root_mean_squared_error' is used as scikit-learn maximises scores
#the value is then squared and we take the root and flip the sign
scoring = {
    'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
    'r2': make_scorer(r2_score, greater_is_better=True)
}

#helper functions
