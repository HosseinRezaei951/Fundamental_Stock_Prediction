import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score,
    f1_score, auc, matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from joblib import dump

class  ML_Algoritms:
    def __init__(
            self, input_path, output_path, save_models_flag,
            algorithms, scorer, score_names, n_jobs, Log
    ):
        self.iPath = input_path
        self.oPath = output_path
        self.saveModels = save_models_flag
        self.algorithms = algorithms
        self.scorer = scorer
        self.scNames = score_names
        self.n_jobs = n_jobs
        self.alNames = list(self.algorithms.keys())
        self.Log = Log
           
    def __call__(
            self, file_name, number_of_features, target_columns,
            preprocessing_method, number_of_splits, shuffle_flag,
            random_state
    ):
        skf = StratifiedKFold(
            n_splits=number_of_splits,
            shuffle=shuffle_flag,
            random_state=random_state
        )

        self.n_features = number_of_features
        
        self.algorithms["DT"]["param_grid"]['random_state'] = [random_state]
        self.algorithms["RF"]["param_grid"]['random_state'] = [random_state]
        self.algorithms["SVM"]["param_grid"]['random_state'] = [random_state]
        self.algorithms["LR"]["param_grid"]['random_state'] = [random_state]
        self.algorithms["GB"]["param_grid"]['random_state'] = [random_state]

        for target_column in target_columns:
            self.results = {alName: {} for alName in self.alNames}
            self.models = {alName: {} for alName in self.alNames}

            self.max_auc_roc = {alName: 0 for alName in self.alNames}
            self.max_aupr = {alName: 0 for alName in self.alNames}
            
            result_path = f'{self.oPath}/{target_column}/{random_state}_{preprocessing_method}'
            self.create_outPutFolder(result_path)

            self.i_feature = 1
            while (self.i_feature <= self.n_features):

                self.Log(
                    f'\n..:: Start Running ML_Algoritms on \"{file_name}\" for {target_column} '+
                    f'with (random_state: {random_state}, '+
                    f'preprocessing_method: {preprocessing_method}, '+
                    f'number_of_features: {self.i_feature}) ::..'
                )
                
                for alName in self.alNames:
                    
                    all_X_train = []
                    all_y_train = []
                    all_X_test = []
                    all_y_test = []
                    all_y_pred = []
                    all_cross_validation_accuracy = []
                    all_bestModels = []
                    all_bestParams = []

                    self.Log( f'\n..:: Start Running {alName} Classifier With {self.i_feature} Features ::..')
                    for fold_number in range(1, number_of_splits+1):  
                                
                        tmp_iPath = f'{self.iPath}/{target_column}/{random_state}_{preprocessing_method}'
                        df_train = pd.read_csv(f'{tmp_iPath}/train_folds/fold_{fold_number}.csv')
                        df_test = pd.read_csv(f'{tmp_iPath}/test_folds/fold_{fold_number}.csv')

                        df_features = pd.read_csv(f'{tmp_iPath}/ranked_features/KBest_fold_{fold_number}.csv')
                        
                        features = df_features["Features"].values.tolist()
                        selected_features = features[0:self.i_feature]

                        X_train, X_test = df_train[selected_features], df_test[selected_features]
                        y_train, y_test = df_train[target_column], df_test[target_column]
                        
                        all_X_train.append(X_train)
                        all_y_train.append(y_train)
                        all_X_test.append(X_test)
                        all_y_test.append(y_test)

                        # HyperParameter Tuning
                        grid_obj = GridSearchCV(
                            estimator=self.algorithms[alName]["estimator"],
                            param_grid=self.algorithms[alName]["param_grid"],
                            scoring=self.scorer, cv=skf, n_jobs=self.n_jobs
                        )
                        
                        # Training and finding best Classifer
                        grid_obj.fit(X_train, y_train)
                        best_est = grid_obj.best_estimator_
                        all_bestModels.append(best_est)

                        y_pred = self.predict_byGmeansThreshold(best_est, X_test, y_test)
                        all_y_pred.append(y_pred)
                        
                        if self.saveModels:
                            dump(
                                best_est,
                                f'{result_path}/by_Algorithms/{alName}/{self.i_feature}_fold{fold_number}.joblib'
                            )
                            
                        self.Log(
                            f'..:: HyperParameter Tuned for Fold {fold_number} ::..\n'+
                            f'==> best_params_: {grid_obj.best_params_}\n' +
                            f'==> Mean cross-validated \"{self.scorer}\" of the best_estimator: '+
                            f'{self.round_percentage(grid_obj.best_score_)}%'
                        )

                        all_bestParams.append(grid_obj.best_params_)
                        all_cross_validation_accuracy.append(grid_obj.best_score_)


                    self.calc_results(
                        alName, all_y_test, all_y_pred,
                        all_cross_validation_accuracy, all_bestParams
                    )
                    self.Log(self.get_algorithmLog(alName))

                    
                    if (
                        self.results[alName][self.i_feature]['aupr_mean'] > self.max_aupr[alName] or
                        self.results[alName][self.i_feature]['auc_roc_mean'] > self.max_auc_roc[alName]
                    ):
                        self.plot_algorithmResults(alName, all_y_test, all_y_pred, result_path)
                        self.max_aupr[alName] = self.results[alName][self.i_feature]['aupr_mean']
                        self.max_auc_roc[alName] = self.results[alName][self.i_feature]['auc_roc_mean']
                                            
                    
                    self.Log( f'..:: End Running {alName} Classifier With {self.i_feature} Features ::..')

                self.Log(
                    f'..:: End Running ML_Algoritms on \"{file_name}\" for {target_column} '+
                    f'with (random_state: {random_state}, preprocessing_method: {preprocessing_method}, '+
                    f'number_of_features: {self.i_feature}) ::..'
                )

                self.i_feature += 1

            self.save_grouped_results(result_path)
            

    def predict_byGmeansThreshold(self, model, X_test, y_test):

        y_prob = model.predict_proba(X_test)[:, 1]
                        
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))

        # locate the index of the largest g-mean
        best_threshold_index = np.argmax(gmeans)
        best_threshold = thresholds[best_threshold_index]
        y_pred_new = np.where(y_prob >= best_threshold, 1, -1)

        return y_pred_new

    def round_percentage(self, number):
        return np.round(number*100, 2)


    def save_grouped_results(self, outPut_path):
        grouped_results = {
            (alName, n_features) : {
                key:value for key, value in self.results[alName][n_features].items()
            } 
            for alName in self.results.keys()
            for n_features in self.results[alName].keys()
        }
        self.df_results = pd.DataFrame.from_dict(grouped_results, orient="index")
        
        self.df_results.index.names = ["Algorithm", "Number_of_Features"]
        self.df_results.to_csv(f'{outPut_path}/results_df.csv')
        
        for s_name in self.scNames:
            self.plot_groupScores(s_name, outPut_path)


    def plot_groupScores(self, score_name, outPut_path):
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(1, 1)

        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'font.weight': 'normal',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })

        ax_scores = fig.add_subplot(gs[0, 0])
        
        # Define the line styles for each line
        line_styles = [
            (0, (1, 0)),      # solid
            (0, (1, 1)),      # dotted
            (0, (5, 1)),      # dashed
            (0, (3, 1, 1, 1)), # dashdot
            (0, (3, 1, 1, 1, 1, 1))  # densely dashdotdotted
        ]
        marker_styles = ["o", "^", "p", "s", "D"]

        # Plot each series with the corresponding line style
        for i, col in enumerate(self.df_results[score_name].unstack(level=0).columns):
            ax_scores.plot(
                self.df_results[score_name].unstack(level=0).index,
                self.df_results[score_name].unstack(level=0)[col],
                linestyle=line_styles[i % len(line_styles)],  # Cycle through the line styles
                marker=marker_styles[i % len(marker_styles)],  # Cycle through the marker styles
                label=col
            )
        
        # show grid lines
        tick_positions = np.arange(0, self.n_features + 1, 10)
        tick_positions = np.insert(tick_positions, 1, 1)
        if tick_positions[-1] != self.n_features:
            tick_positions = np.append(tick_positions, self.n_features)

        ax_scores.set_xticks([i for i in range(1, self.n_features+1)])
        ax_scores.set_xticklabels(['' if i not in tick_positions else str(i) for i in range(1, self.n_features+1)])

        title = score_name.replace("_", " ").title()
        if "Aupr" in title:
            title = title.replace("Aupr","AUPR")
        if "Auc Roc" in title:
            title = title.replace("Auc Roc","AUC-ROC")

        ax_scores.set_title(title)
        ax_scores.set_xlabel('Number of Features')
        ax_scores.set_ylabel('Score (%)')
        ax_scores.grid(True)
        
        # show the legend
        ax_scores.legend(loc='best')

        # show/save the plot
        plt.tight_layout()
        plt.savefig(f"{outPut_path}/{score_name}.png", dpi=120)
        # plt.show()
        plt.close()


    def plot_algorithmResults(self, alName, all_y_test, all_y_pred, outPut_path):

        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'font.weight': 'normal',
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })
        
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')

        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision') 

        # Plot the no-skill line
        ax_roc.plot([0, 1], [0, 1], linestyle='--', label='No Skill', color='black')

        y = np.concatenate(all_y_test)
        no_skill = len(y[y==1]) / len(y)

        ax_pr.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', color='black')

        tprs = []
        roc_aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        precisions = []
        pr_aucs = []
        mean_recall = np.linspace(0, 1, 100)

        for i in range(len(all_y_test)):
            y_test = all_y_test[i]
            y_pred = all_y_pred[i]

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(
                fpr, tpr, alpha=0.3, linewidth=1, 
                label=f'Fold {i+1} (AUC={self.round_percentage(roc_auc)}%)'
            )

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            roc_aucs.append(roc_auc)
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            pr_auc = auc(recall, precision)
            ax_pr.plot(
                recall, precision, alpha=0.3, linewidth=1, 
                label=f'Fold {i+1} (AUC={self.round_percentage(pr_auc)}%)'
            )
            r_recall = np.fliplr([recall])[0]
            r_precision = np.fliplr([precision])[0]
            interp_precisions = np.interp(mean_recall, r_recall, r_precision)
            interp_precisions[-1] = no_skill
            precisions.append(interp_precisions)
            pr_aucs.append(pr_auc)
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_roc_auc = np.mean(roc_aucs, axis=0)
        std_roc_auc = np.std(roc_aucs, axis=0)

        ax_roc.plot(
            mean_fpr, mean_tpr, color="b", lw=2,
            label=f'Mean (AUC={self.round_percentage(mean_roc_auc)}% \u00B1 {self.round_percentage(std_roc_auc)})'
        )
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        
        ax_roc.fill_between(
            mean_fpr, tprs_lower, tprs_upper,
            color="grey", alpha=0.1,
            label=f'\u00B1 1 std. dev.',
        )
        
        mean_precision = np.mean(precisions, axis=0)
        mean_precision[0] = 1.0
        mean_pr_auc = np.mean(pr_aucs, axis=0)
        std_pr_auc = np.std(pr_aucs, axis=0)
                
        ax_pr.plot(
            mean_recall, mean_precision, color="b", lw=2,
            label=f'Mean (AUC={self.round_percentage(mean_pr_auc)}%  \u00B1 {self.round_percentage(std_pr_auc)})'
        )
        
        # Compute the standard deviation of tprs
        std_precision = np.std(precisions, axis=0)
        precisions_upper = np.minimum(mean_precision + std_precision, 1)
        precisions_lower = np.maximum(mean_precision - std_precision, 0)

        ax_pr.fill_between(
            mean_recall, precisions_lower, precisions_upper,
            color="grey", alpha=0.1,
            label=f'\u00B1 1 std. dev.',
        )
        
        # show grid lines
        ax_roc.grid(True)
        ax_pr.grid(True)

        # show the legend
        ax_roc.legend(loc='best')
        ax_pr.legend(loc='best')

        # show/save the plot
        plt.tight_layout()
        plt.savefig(f"{outPut_path}/by_Algorithms/{alName}/{self.i_feature}.png", dpi=120)
        plt.close()
                

    def get_algorithmLog(self, alName):
        results = self.results[alName][self.i_feature]
        log_str = ""
        log_str += f'\n..:: {alName} Classifier Mean Results with {self.i_feature} features ::.. \n'

        log_str += f'==> Cross Validation Accuracy: ' +\
        f'{results["cross_validation_accuracy_mean"]}% \u00B1 {results["cross_validation_accuracy_std"]} \n'

        log_str += f'==> Test Accuracy: ' +\
        f'{results["test_accuracy_mean"]}% \u00B1 {results["test_accuracy_std"]} \n'

        log_str += f'==> Specificity: ' +\
        f'{results["specificity_mean"]}% \u00B1 {results["specificity_std"]} \n'

        log_str += f'==> Sensitivity: ' +\
        f'{results["sensitivity_mean"]}% \u00B1 {results["sensitivity_std"]} \n'

        log_str += f'==> MCC: ' +\
        f'{results["mcc_mean"]}% \u00B1 {results["mcc_std"]} \n'

        log_str += f'==> F1: ' +\
        f'{results["f1_mean"]}% \u00B1 {results["f1_std"]} \n'

        log_str += f'==> ROC-AUC: ' +\
        f'{results["auc_roc_mean"]}% \u00B1 {results["auc_roc_std"]} \n'

        log_str += f'==> AUPR: ' +\
        f'{results["aupr_mean"]}% \u00B1 {results["aupr_std"]} \n'
        
        return log_str


    def calc_results(
            self, alName, all_y_test, all_y_pred,
            all_cross_validation_accuracy, all_bestParams
    ):
        all_test_accuracy = []
        all_f1 = []
        all_auc_roc = []
        all_aupr = []
        all_specificity = []
        all_sensitivity = []
        all_mcc = []
        for i in range(len(all_y_test)):
            y_test = all_y_test[i]
            y_pred = all_y_pred[i]

            # Accuracy
            test_accuracy = accuracy_score(y_test, y_pred)
            all_test_accuracy.append(test_accuracy)

            # F1 Score
            f1 = f1_score(y_test, y_pred)
            all_f1.append(f1)
    
            # ROC-AUC Score
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            auc_roc = auc(fpr, tpr)
            all_auc_roc.append(auc_roc)

            # PR-AUC Score
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            aupr = auc(recall, precision)
            all_aupr.append(aupr)


            # Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Specificity
            specificity = np.round(tn/(fp+tn), 4)
            all_specificity.append(specificity)

            # Sensitivity
            sensitivity = np.round(tp/(tp+fn), 4)
            all_sensitivity.append(sensitivity)         

            # MCC
            mcc = matthews_corrcoef(y_test, y_pred)
            all_mcc.append(mcc)


        results = {           
            "cross_validation_accuracy_mean": self.round_percentage(np.mean(all_cross_validation_accuracy)), 
            "cross_validation_accuracy_std": self.round_percentage(np.std(all_cross_validation_accuracy)), 

            "test_accuracy_mean": self.round_percentage(np.mean(all_test_accuracy)),
            "test_accuracy_std": self.round_percentage(np.std(all_test_accuracy)),

            "specificity_mean": self.round_percentage(np.mean(all_specificity)),
            "specificity_std": self.round_percentage(np.std(all_specificity)),

            "sensitivity_mean": self.round_percentage(np.mean(all_sensitivity)),
            "sensitivity_std": self.round_percentage(np.std(all_sensitivity)),

            "mcc_mean": self.round_percentage(np.mean(all_mcc)),
            "mcc_std": self.round_percentage(np.std(all_mcc)),
            
            "f1_mean": self.round_percentage(np.mean(all_f1)),  
            "f1_std": self.round_percentage(np.std(all_f1)),

            "auc_roc_mean": self.round_percentage(np.mean(all_auc_roc)),
            "auc_roc_std": self.round_percentage(np.std(all_auc_roc)),
            
            "aupr_mean": self.round_percentage(np.mean(all_aupr)),
            "aupr_std": self.round_percentage(np.std(all_aupr)),
            
            "best_param_fold_1": all_bestParams[0],
            "best_param_fold_2": all_bestParams[1],
            "best_param_fold_3": all_bestParams[2],
            "best_param_fold_4": all_bestParams[3],
            "best_param_fold_5": all_bestParams[4],
        }
        self.results[alName][self.i_feature] = results

    def create_outPutFolder(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
            os.makedirs(f'{path}/by_Algorithms')
            for alName in self.alNames:
                os.makedirs(f'{path}/by_Algorithms/{alName}/')
