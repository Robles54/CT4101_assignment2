import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import warnings

#Filepath for the dataset
data_file = 'steel.csv'

#Column names in the dataset
column_names = [
    'normalising_temperature', 'tempering_temperature', 'percent_silicon',
    'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur',
    'percent_carbon', 'percent_manganese', 'tensile_strength'
]

# Target variable to predict
target_variable = 'tensile_strength'

# Setup for 10-fold cross-validation
CV_FOLDS = KFold(n_splits=10, shuffle=True, random_state=42)

#Defining scoring metrics since the desired metric is RMSE and R^2
#For RMSE, 'neg_root_mean_squared_error' is used as scikit-learn maximises scores
#the value is then squared and we take the root and flip the sign
scoring = {
    
    'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
    'r2': make_scorer(r2_score, greater_is_better=True)
}

#helper functions
def print_cv_results(model_name, cv_results):
    """Helper function to print formatted cross-validation results."""
    print(f"\n--- {model_name} Results ---")
    
    #Check for train scores
    if 'train_rmse' in cv_results:
        #cross_validate returns arrays of scores for each fold + calculate the mean and std.
        mean_train_rmse = -np.mean(cv_results['train_rmse'])
        std_train_rmse = np.std(cv_results['train_rmse'])
        mean_train_r2 = np.mean(cv_results['train_r2'])
        std_train_r2 = np.std(cv_results['train_r2'])
        
        print(f"  Average Train RMSE: {mean_train_rmse:.4f} +/- {std_train_rmse:.4f}")
        print(f"  Average Train R^2:  {mean_train_r2:.4f} +/- {std_train_r2:.4f}")

    #Test scores present
    mean_test_rmse = -np.mean(cv_results['test_rmse'])
    std_test_rmse = np.std(cv_results['test_rmse'])
    mean_test_r2 = np.mean(cv_results['test_r2'])
    std_test_r2 = np.std(cv_results['test_r2'])

    print(f"  Average Test RMSE: {mean_test_rmse:.4f} +/- {std_test_rmse:.4f}")
    print(f"  Average Test R^2:  {mean_test_r2:.4f} +/- {std_test_r2:.4f}")
    print("-" * (len(model_name) + 20))

#main

def main():
    #Main function to run the ML pipeline.

    #Loading data
    try:
        data = pd.read_csv(data_file, header=0)
    except FileNotFoundError:
        print(f"Error: The data file '{data_file}' was not found.")
        print("Please make sure 'steel.csv' is in the same directory as this script.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Successfully loaded data from '{data_file}'.")
    print(f"Data shape: {data.shape}")

    #Separating features X and target y
    try:
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]
    except KeyError:
        print(f"Error: Target variable '{target_variable}' not found in columns.")
        return

    #Define Models and Pipelines
    
    #Model 1: Random Forest Regressor
    #tuning: n_estimators: the number of trees in the forest
    #and max_depth: the maximum depth of each tree
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    #Model 2: Support Vector Regression (SVR)
    #tuning: C: Regularization parameter and gamma: Kernel coefficient for 'rbf' kernel
    pipe_svr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf'))
    ])

    print("\nStarting model evaluation...")

    #Evaluate Models with Default Hyperparameters
    print("\nRunning 10-fold CV with default hyperparameters...")

    #Random Forest with default
    rf_default_results = cross_validate(pipe_rf, X, y, cv=CV_FOLDS, scoring=scoring, return_train_score=True)
    print_cv_results("Random Forest (Default)", rf_default_results)

    #SVR with Default
    svr_default_results = cross_validate(pipe_svr, X, y, cv=CV_FOLDS, scoring=scoring, return_train_score=True)
    print_cv_results("SVR (Default)", svr_default_results)

    #Hyperparameter Tuning with GridSearchCV
    print("\nStarting GridSearchCV for hyperparameter tuning...")

    #Random Forest Tuning
    print("\nTuning Random Forest (n_estimators, max_depth)...")
    param_grid_rf = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20]
    }
    
    #refit on the 'rmse' score using the negative version
    grid_rf = GridSearchCV(
        pipe_rf, 
        param_grid_rf, 
        cv=CV_FOLDS, 
        scoring=scoring, 
        refit='rmse', 
        return_train_score=True, 
        n_jobs=-1,
        verbose=1
    )
    grid_rf.fit(X, y)

    print("\n--- Random Forest Tuning Results ---")
    print(f"Best Parameters: {grid_rf.best_params_}")
    print(f"Best Test RMSE: {-grid_rf.best_score_:.4f}")
    
    #display full grid search results as a DataFrame
    rf_grid_df = pd.DataFrame(grid_rf.cv_results_)
    print("\nFull RF Grid Search Results (for report):")
    rf_results_cols = [
        'param_model__n_estimators', 'param_model__max_depth', 
        'mean_test_rmse', 'std_test_rmse', 'mean_test_r2',
        'mean_train_rmse', 'std_train_rmse', 'mean_train_r2'
    ]
    #flip sign for RMSE scores
    rf_grid_df['mean_test_rmse'] = -rf_grid_df['mean_test_rmse']
    rf_grid_df['mean_train_rmse'] = -rf_grid_df['mean_train_rmse']
    print(rf_grid_df[rf_results_cols].sort_values(by='mean_test_rmse'))

    #SVR Tuning
    print("\nTuning SVR (C, gamma)...")
    param_grid_svr = {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 0.1, 1, 10]
    }

    grid_svr = GridSearchCV(
        pipe_svr,
        param_grid_svr,
        cv=CV_FOLDS,
        scoring=scoring,
        refit='rmse',
        return_train_score=True,
        n_jobs=-1,
        verbose=1
    )
    grid_svr.fit(X, y)

    print("\n--- SVR Tuning Results ---")
    print(f"Best Parameters: {grid_svr.best_params_}")
    print(f"Best Test RMSE: {-grid_svr.best_score_:.4f}")

    #display full grid search results
    svr_grid_df = pd.DataFrame(grid_svr.cv_results_)
    print("\nFull SVR Grid Search Results (for report):")
    svr_results_cols = [
        'param_model__C', 'param_model__gamma',
        'mean_test_rmse', 'std_test_rmse', 'mean_test_r2',
        'mean_train_rmse', 'std_train_rmse', 'mean_train_r2'
    ]
    #flip sign for RMSE scores
    svr_grid_df['mean_test_rmse'] = -svr_grid_df['mean_test_rmse']
    svr_grid_df['mean_train_rmse'] = -svr_grid_df['mean_train_rmse']
    print(svr_grid_df[svr_results_cols].sort_values(by='mean_test_rmse'))

    print("\n\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()