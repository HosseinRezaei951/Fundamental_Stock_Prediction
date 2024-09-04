import os, time
import pandas as pd

from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVMClassifier
from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier 
from sklearn.ensemble import GradientBoostingClassifier

from Assets.Log import Log
from Assets.Data_Labeling import Data_Labeling
from Assets.Data_Preparation import Data_Preparation
from Assets.ML_Algoritms import ML_Algoritms

activations = {
    "Data_Labeling": False,
    "Data_Preparation": False,
    "ML_Algoritms": False
}
activations["Data_Labeling"] = True
activations["Data_Preparation"] = True
activations["ML_Algoritms"] = True

######### Main Parameters #########
dataFolder_path = "Data"
resultFolder_path = "Results"
dataFiles_name = "Main_Data"
number_of_splits = 5
random_state_numbers = [7, 42, 75, 101, 216]
index_columns = ["CompanyId", "PersianYear"]
target_columns = ["r_dicho", "b_dicho"]
drop_columns = []
######### Main Parameters #########

######### Data_Labeling Parameters #########
dl_input_path = dataFolder_path
dl_output_path = f'{resultFolder_path}/1_Lebeled_Data'
dl_columns = ["Return", "Beta"]
dl_rules = "R2B2"
######### Data_Labeling Parameters #########

######### Data_Preparation Parameters #########
dp_input_path = dl_output_path
dp_output_path = f'{resultFolder_path}/2_Prepared_Data'
dp_methods = ["MinMax", "Standard"]
######### Data_Preparation Parameters #########

######### ML_Algorithms Parameters #########
ml_input_path = dp_output_path
ml_output_path = f'{resultFolder_path}/3_Algorithms_Results'
ml_save_models_flag = False
ml_algorithms = {
    "DT": {
        "estimator": DecisionTreeClassifier(),
        "param_grid": {
            'max_features': ['sqrt', "log2"],
            'max_depth': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 3, 5, 7],
        }
    },
    "LR": {
        "estimator": LogisticRegressionClassifier(),
        "param_grid": {
            'penalty' : ['l2'],
            'C' :  [10, 1.0, 0.1, 0.01],
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        }
    },
    "RF": {
        "estimator": RandomForestClassifier(),
        "param_grid": {
            'n_estimators': [5, 10, 20, 50],
            'max_features': ['sqrt', "log2"],
        }
    },    
    "SVM":{
        "estimator": SVMClassifier(),
        "param_grid": {
            'kernel': ['rbf', 'sigmoid'],
            'C': [10, 1.0, 0.1, 0.01],
            'gamma': ['scale'],
            'probability': [True],
        }
    },
    "GB": {
        "estimator": GradientBoostingClassifier(),
        "param_grid": {
            'n_estimators' : [5, 10, 20, 50],
            'learning_rate' :  [0.001, 0.01, 0.1],
        }
    }
}
ml_score_names = [
    'cross_validation_accuracy_mean',
    'test_accuracy_mean','specificity_mean', 
    'sensitivity_mean','mcc_mean',
    'f1_mean', 'auc_roc_mean', 'aupr_mean'
]
ml_nJobs = -1
ml_scorer = 'accuracy'
######### Algorithms Parameters #########

if not os.path.isdir(resultFolder_path):
    os.makedirs(resultFolder_path)

# get the start time
start_time = time.time()
main_log = Log(resultFolder_path, "main_log")


######### Log Parameters #########
# Main
main_log(F'\n..:: Log Parameters ::..')
main_log(F'==> dataFolder_path: {dataFolder_path}')
main_log(F'==> resultFolder_path: {resultFolder_path}')
main_log(F'==> dataFiles_name: {dataFiles_name}')
main_log(F'==> number_of_splits: {number_of_splits}')
main_log(F'==> random_state_numbers: {random_state_numbers}')
main_log(F'==> index_columns: {index_columns}')

# Data_Labeling
main_log(F'==> dl_input_path: {dl_input_path}')
main_log(F'==> dl_output_path: {dl_output_path}')
main_log(F'==> dl_columns: {dl_columns}')
main_log(F'==> dl_rules: {dl_rules}')

# Preparing
main_log(F'==> dp_input_path: {dp_input_path}')
main_log(F'==> dp_output_path: {dp_output_path}')
main_log(F'==> dp_methods: {dp_methods}')

# Algorithms
main_log(F'==> ml_input_path: {ml_input_path}')
main_log(F'==> ml_output_path: {ml_output_path}')
main_log(F'==> ml_algorithms: {ml_algorithms}')
main_log(F'==> ml_score_names: {ml_score_names}')
main_log(F'==> ml_scorer: {ml_scorer}')
main_log(F'==> ml_nJobs: {ml_nJobs}')
######### Log Parameters #########
    
if activations["Data_Labeling"]:
    DL = Data_Labeling(dl_input_path, dl_output_path, main_log)

if activations["Data_Preparation"]:
    DP = Data_Preparation(dp_input_path, dp_output_path, main_log)

if activations["ML_Algoritms"]:
    ML = ML_Algoritms(
        ml_input_path, ml_output_path, ml_save_models_flag,
        ml_algorithms, ml_scorer, ml_score_names, ml_nJobs, main_log
    )

######### Main Loop #########

df = pd.read_csv(f'{dataFolder_path}/{dataFiles_name}.csv')
main_log(f'\n==> \"{dataFiles_name}\" (Orginal) shape: {df.shape}')
new_df = df.dropna()
new_df = new_df.drop(columns=drop_columns)

groupBase_column_set = set(new_df["CompanyId"])
for x in groupBase_column_set:
    if len(new_df[new_df["CompanyId"]==x]) < 2:
        new_df = new_df.drop(new_df[new_df["CompanyId"] == x].index)

new_df = new_df.astype({'Return_ratio_without_risk': 'float64'})
main_log(f'==> \"{dataFiles_name}_new\" (After droping drop_columns and Nan rows) shape: {new_df.shape}')
new_df.to_csv(f'{dataFolder_path}/{dataFiles_name}_new.csv', index=False)

file_name = dataFiles_name+'_new'
nRows, nColumns = new_df.shape
number_of_features = (nColumns - len(index_columns+dl_columns))
main_log(f'==> \"{file_name}_new\" number_of_features: {number_of_features}')

# Data_Labeling
if activations["Data_Labeling"]:
    DL(file_name, dl_rules)

for random_state in random_state_numbers:
    for dp_method in dp_methods: 
        shuffle_flag = True
        if random_state == None:
            shuffle_flag = False

        # Data_Preparation
        if activations["Data_Preparation"]:
            DP(
                f'{file_name}', number_of_features, target_columns,
                index_columns, dl_columns, dp_method,
                number_of_splits, shuffle_flag, random_state
            )  

        # ML_Algoritms
        if activations["ML_Algoritms"]:
            ML(
                f'{file_name}', number_of_features,
                target_columns, dp_method, number_of_splits,
                shuffle_flag, random_state
            )

######### Main Loop #########

# get the end time
end_time = time.time()

# get the execution time
elapsed_time = end_time - start_time
main_log(f'\n==> Execution time: {elapsed_time} seconds')

main_log.write_log()
print("end")