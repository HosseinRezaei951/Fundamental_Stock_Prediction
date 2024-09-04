import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


class Data_Preparation:
    def __init__(self, input_path, output_path, Log):
        self.iPath = input_path
        self.oPath = output_path
        self.Log = Log
        
    def __call__(
            self, file_name, n_features, target_columns,
            index_columns, labeling_columns, preprocessing_method,
            number_of_splits, shuffle_flag, random_state
    ):
        skf = StratifiedKFold(
            n_splits=number_of_splits,
            shuffle=shuffle_flag,
            random_state=random_state
        )
        
        for target in target_columns:
            self.Log(f'\n..:: Start Data_Preparation \"{file_name}\" for {target} with '+
                        f'(random_state: {random_state}, preprocessing_method: {preprocessing_method}) ::..')
            
            df = pd.read_csv(f'{self.iPath}/{file_name}.csv')
            self.Log(f'==> \"{file_name}\" shape: {df.shape}')
            
            tmp_path = f'{self.oPath}/{target}/{random_state}_{preprocessing_method}'
            train_folds_path = f'{tmp_path}/train_folds'
            test_folds_path = f'{tmp_path}/test_folds'
            ranked_features_path = f'{tmp_path}/ranked_features/'

            self.create_outPutFolder(train_folds_path)
            self.create_outPutFolder(test_folds_path)
            self.create_outPutFolder(ranked_features_path)
            
            unWanted_columns = index_columns+labeling_columns+target_columns
            X = df.drop(unWanted_columns, axis=1)
            y = df[target]

            for fold_number, (train_index, test_index) in enumerate(skf.split(X, y)):

                # Splitting
                df_train = df.iloc[train_index]
                df_test = df.iloc[test_index]
                self.Log(
                    f'==> fold_{fold_number+1} >> ' +
                    f'\"train\" shape: {df_train.shape}, '+
                    f'\"test\" shape: {df_test.shape}'
                )

                train_values = df_train.drop(unWanted_columns, axis=1)
                test_values = df_test.drop(unWanted_columns, axis=1)

                # Scaling
                copy_train_values = train_values.copy()
                copy_test_values = test_values.copy()

                if  preprocessing_method == "MinMax":
                    MinMax_scaler = MinMaxScaler().fit(copy_train_values)
                    copy_train_values = MinMax_scaler.transform(copy_train_values)
                    copy_test_values = MinMax_scaler.transform(copy_test_values)

                if preprocessing_method == "Standard":
                    Standard_scaler = StandardScaler().fit(copy_train_values)
                    copy_train_values = Standard_scaler.transform(copy_train_values)
                    copy_test_values = Standard_scaler.transform(copy_test_values)

                if preprocessing_method == "SN":
                    Standard_scaler = StandardScaler().fit(copy_train_values)
                    copy_train_values = Standard_scaler.transform(copy_train_values)
                    copy_test_values = Standard_scaler.transform(copy_test_values)
                    copy_train_values = preprocessing.normalize(copy_train_values)
                    copy_test_values = preprocessing.normalize(copy_test_values)

                if preprocessing_method == "MN":
                    MinMax_scaler = MinMaxScaler().fit(copy_train_values)
                    copy_train_values = MinMax_scaler.transform(copy_train_values)
                    copy_test_values = MinMax_scaler.transform(copy_test_values)
                    copy_train_values = preprocessing.normalize(copy_train_values)
                    copy_test_values = preprocessing.normalize(copy_test_values)
                
                tmp_df_train = pd.DataFrame(copy_train_values, columns=train_values.columns)
                df_train.loc[:, tmp_df_train.columns] = tmp_df_train.loc[:,tmp_df_train.columns].values
                
                tmp_df_test = pd.DataFrame(copy_test_values, columns=test_values.columns)
                df_test.loc[:, tmp_df_test.columns] = tmp_df_test.loc[:,tmp_df_test.columns].values

                df_train.to_csv(f'{train_folds_path}/fold_{fold_number+1}.csv', index=False)
                df_test.to_csv(f'{test_folds_path}/fold_{fold_number+1}.csv', index=False)  
                    
                # Feature Ranking KBest
                self.Log(f'..:: Start KBest Feature Ranking on Fold {fold_number+1} ::..')
                KBest_finalScores = self.KBest_ranking(
                    df_train, n_features, target, unWanted_columns,
                    number_of_splits, shuffle_flag, random_state,
                    fold_number+1, ranked_features_path
                )
                self.Log(f'==> Final KBest Ranked Features for Fold {fold_number}:')
                self.Log(f'{KBest_finalScores}')
                self.Log(f'..:: End KBest Feature Ranking on Fold {fold_number+1} ::..')

            self.Log(f'..:: End Data_Preparation \"{file_name}\" for {target} with '+
                    f'(random_state: {random_state}, Scaling_method: {preprocessing_method}) ::..')
            
      
    def get_finalScores(self, features_lists, features):
        final_scores = pd.DataFrame(data = {'Features':features})

        for i in range(len(features_lists)):
            tmp_scores = []
            for feature in final_scores["Features"]:
               tmp_scores.append(features_lists[i].index(feature)+1)
            final_scores[i+1] = tmp_scores

        modes = []
        means = []
        stds = []
        for i in range(len(final_scores)):
            data = final_scores.iloc[i, 1:]
            modes.append(data.mode()[0])
            means.append(data.mean())
            stds.append(data.std())
        final_scores['Mode'] = modes
        final_scores['Mean'] = means
        final_scores['Std'] = stds
        
        final_scores = final_scores.sort_values(by=['Mode', 'Mean', "Std"])
        final_scores.reset_index(drop=True, inplace=True)
        return final_scores

    

    def KBest_ranking(
            self, df, n_features, target_column, unWanted_columns,
            number_of_splits, shuffle_flag, random_state,
            fold_number, outPut_path
    ):
        features = [x for x in df.columns if x not in unWanted_columns]
        final_scores = pd.DataFrame(data={'Features': features})

        KBest_ouputs = []
        skf = StratifiedKFold(
            n_splits=number_of_splits,
            shuffle=shuffle_flag,
            random_state=random_state
        )

        X = df[features]
        y = df[target_column]
        for sample_index, _ in skf.split(X, y):
            sample_X = X.iloc[sample_index]
            sample_y = y.iloc[sample_index]

            # Create the SelectKBest object with the f_regression scoring function
            selector = SelectKBest(score_func=f_regression, k=n_features)

            # Apply feature selection to the dataset
            X_new = selector.fit_transform(sample_X, sample_y)

            # Get the scores of the selected features
            feature_scores = selector.scores_

            # Get the indices of the selected features sorted by score (descending order)
            selected_feature_indices = np.argsort(feature_scores)[::-1][:n_features]

            # Get the names of the selected features
            selected_features = final_scores.Features[selected_feature_indices]

            # Print the ranked feature names and scores
            KBest_ouputs.append(selected_features.tolist())          

        final_scores = self.get_finalScores(KBest_ouputs, features)
        final_scores.to_csv(f"{outPut_path}/KBest_fold_{fold_number}.csv", index=False)
                
        return final_scores


    def create_outPutFolder(self, tmp_path):
        if not os.path.isdir(tmp_path):
            os.makedirs(tmp_path)