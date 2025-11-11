import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings

data_file = 'steel.csv'

column_names = [
    'normalising_temperature', 'tempering_temperature', 'percent_silicon',
    'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur',
    'percent_carbon', 'percent_manganese', 'tensile_strength'
]

target_variable = 'tensile_strength'