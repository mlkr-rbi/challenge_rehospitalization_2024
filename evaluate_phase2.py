import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    matthews_corrcoef, f1_score, roc_auc_score, accuracy_score )
from webdav3.client import Client
from hmeasure import h_score
from datetime import datetime
from pathlib import Path


spe_teams = {'Tim_03': 'ATMC',
 'Tim_19': 'ReAdmit',
 'Tim_25': 'ZMF',
 'Tim_14': 'MediBoost',
 'Tim_02': 'Artificial Innovators',
 'Tim_10': 'Kronos',
 'Tim_11': 'LIMMB',
 'Tim_12': 'LMHS 1',
 'Tim_21': 'SiraćHC',
 'Tim_08': 'IMPERIA'}

team_dict = {'Tim_02': 'aispe1-rdp',
             'Tim_03': 'aispe2-rdp',
             'Tim_08': 'aispe3-rdp',
             'Tim_10': 'aispe4-rdp',
             'Tim_11': 'aispe5-rdp',
             'Tim_12': 'aispe6-rdp',
             'Tim_14': 'aispe7-rdp',
             'Tim_19': 'aispe8-rdp',
             'Tim_21': 'aispe9-rdp',
             'Tim_25': 'aispe10-rdp'}

def calculate_final_metrics(tims_dir, tim_dirs):
    # The code `tims_dir` appears to be a variable or object being referenced or used in Python.
    # However, the code snippet provided is incomplete and does not provide enough context to
    # determine its exact purpose or functionality. More information or surrounding code would be
    # needed to provide a more accurate explanation.
    # It looks like the code `tims_dir` is just a variable name being assigned, but no value or
    # operation is being performed on it.

    for tim in tim_dirs:
        path_final_private = tims_dir / f"{tim}" / 'Final' / "metrics_private.csv"
        print('*'*30, tim, '*'*30)
        print(path_final_private)
        items = os.listdir(tims_dir / tim)
        max_number_file = None
        for item_name in items:
            print(item_name)
            if item_name != 'metrics_private.csv' and item_name:
                if item_name.split('_')[1] == 0:
                    pass
                else:
                    try:
                        number_str = item_name.split('_')[1]
                        if item_name.split('_')[0] == 'Tim':
                            number_str = item_name.split('_')[2]
                        if '.' in number_str:
                            number_str = number_str.split('.')[0]
                        max_number_file = item_name
                    except ValueError:
                        pass
        tim_path = tims_dir/ f"{tim}" / 'Final'
        print(f"For the folder: {tim_path}, file is: {max_number_file}")
        if max_number_file is not None:
            eval = Eval(root = tims_dir, team_name = tim)
            eval.read_data() 
            df_private = pd.DataFrame(columns=["Team", "Date", "Order",
                            "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy", "fname"
                        ])
            eval.eval_final(df_private, fname = tim_path / f"{max_number_file}")
        else:
            pass

def final_leaderboard():
    df_leaderboard = pd.DataFrame(columns=["Team", "Date", "Order", 
                "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"])
    print(df_leaderboard)
    folder_teams_list
    tims = ["Tim_{:02d}/final".format(i) for i in range(1, 29)]
    for tim in tims:
        tim_path=Path("./Timovi/"+ tim )
        df = pd.read_csv(tim_path / 'metrics_private.csv', index_col=0)
        df = df[df['MCC'] == df['MCC'].max()]
        df_leaderboard = pd.concat([df_leaderboard,df], axis=0, join='outer', ignore_index=False)
        # df_leaderboard = df_leaderboard.append(df[df['MCC'] == df['MCC'].max()])

    df_leaderboard_sorted = df_leaderboard.sort_values(by='MCC', ascending=False)
    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'
                                ].apply(lambda x: x.split('/')[0])

    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'].map(teams)
    df_selected = df_leaderboard_sorted[['Team', 'MCC', 
                    'H_measure', "F1", "ROC_AUC"]].reset_index(drop=True)

    df_selected['H_measure'] = df_selected['H_measure'].apply(lambda x: round(x, 4))
    df_selected['Redni broj'] = df_selected.index + 1
    # make the 'Redni broj' column the first column
    cols = df_selected.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_selected = df_selected[cols]
    df_selected.to_html('leaderboard_table_final.html', index=False)
    df_selected

def read_df_order(path):
    df = pd.read_csv(path, index_col=0)
    return df['Order'].max()

def load_from_r():
    pred_proba_test_iid = pd.read_csv('Challenge Phase2/pred_proba_test_iid.csv', index_col=0)
    pred_proba_test_ood = pd.read_csv('Challenge Phase2/pred_proba_test_ood.csv', index_col=0)
    pred_test_iid = pd.read_csv('Challenge Phase2/pred_test_iid.csv', index_col=0)
    pred_test_ood = pd.read_csv('Challenge Phase2/pred_test_ood.csv', index_col=0)
    pred_test_iid.rename(columns={'x': 'Label'}, inplace=True)
    pred_test_ood.rename(columns={'x': 'Label'}, inplace=True)
    pred_test_iid.reset_index(drop=True, inplace=True)
    pred_test_ood.reset_index(drop=True, inplace=True)

    pred_proba_test_ood.rename(columns={'No_Readmission': 'Probability_0', 'Readmission':'Probability_1'}, inplace=True)
    pred_proba_test_iid.rename(columns={'No_Readmission': 'Probability_0', 'Readmission':'Probability_1'}, inplace=True)
    pred_proba_test_iid.reset_index(drop=True, inplace=True)
    pred_proba_test_ood.reset_index(drop=True, inplace=True)

    yhat_iid = pd.concat([pred_test_iid, pred_proba_test_iid], axis=1)
    yhat_ood = pd.concat([pred_test_ood, pred_proba_test_ood], axis=1)
    yhat = pd.concat([yhat_iid, yhat_ood], axis=0)
    yhat['Label'] = yhat['Label'].apply(lambda x: 1 if x == 'Readmission' else 0)
    yhat.reset_index(drop=True, inplace=True)
    yhat.to_csv('Challenge Phase2/Baseline_1_06032024.csv')
    yhat.to_csv('Timovi/Baseline/Baseline_1_06032024.csv')

def clean(folder_teams_list):
    for i in folder_teams_list:
        for j in ['/Submission', '/Upload', '/Evaluation']:
            submissions_folder_path="./Timovi/" + i + j
            os.makedirs(submissions_folder_path, exist_ok=True)
            df = pd.DataFrame(columns=["Team", "Date", "Order", 
                        "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"
                ])
            if j == '/Evaluation':
                df.to_csv(submissions_folder_path + '/metrics_public.csv')
                df.to_csv(submissions_folder_path + '/metrics_private.csv')


class Eval():
    def __init__(self, root = '', team_name = 'Tim_00', 
                seed= 44, no_val_samples = 2000) -> None:

        self.yhat = None
        self.team_name = team_name
        self.date = None
        self.Label = []
        self.trainy = []
        self.testy = None
        self.test_hat = None
        self.val_ood_index = None
        self.test_index = None
        self.proba_bool = False
        self.order = None
        self.no_val_samples = no_val_samples
        self.probabilities_1 = None
        self.probabilities_1_ood = None
        self.root = root
        self.h_measure = None
        self.MCC = None
        self.brier_score = None
        self.f1 = None
        self.roc_auc = None
        self.accuracy = None
        self.metrics_private = None
        self.metrics_public = None
        self.seed = seed
        
    def read_data(self):
        test = pd.read_csv(Path('Challenge Phase2/test_iid_labels.csv'))
        self.testy = test['Label']
        ood_test = pd.read_csv(Path('Challenge Phase2/test_ood_labels.csv'))
        self.ood_testy = ood_test['Label']
        train = pd.read_csv(Path('Challenge Phase2/train_prep_labels.csv'))
        self.trainy = train['Label']
        self.sr = self.trainy.value_counts(normalize=True)[1]

    def load_predictions(self, file):
            self.test_hat = pd.read_csv(file)
            # self.ood_hat = pd.read_csv(file2)
            try:
                self.yhat = self.test_hat['Label'].iloc[-self.testy.shape[0]:] # Ovdje je promijenjeno
                self.yhat_val = self.yhat.sample(self.no_val_samples, random_state=self.seed)
                self.yhat_ood = self.test_hat['Label'].iloc[:-self.testy.shape[0]] # Ovdje je promijenjeno
                self.yhat_ood_val = self.yhat_ood.sample(self.no_val_samples, random_state=self.seed)
                if 'Probability_0' in self.test_hat.columns and 'Probability_1' in self.test_hat.columns:
                    self.proba_bool = True
                    self.probabilities_1 = self.test_hat['Probability_1'
                                                ].iloc[-self.testy.shape[0]:].values # Ovdje je promijenjeno
                    
                    self.probabilities_1_ood = self.test_hat['Probability_1'
                                                ].iloc[:-self.testy.shape[0]].values # Ovdje je promijenjeno
                    
                    self.probabilities_1_val = self.test_hat['Probability_1'
                            ].iloc[-self.testy.shape[0]:].sample(
                                self.no_val_samples, random_state=self.seed) # Ovdje je promijenjeno
                            
                    self.probabilities_1_val_ood = self.test_hat['Probability_1'
                            ].iloc[:-self.testy.shape[0]].sample(
                                self.no_val_samples, random_state=self.seed) # Ovdje je promijenjeno
                            
                if type(file) == str:
                    self.order = file.split('/')[-1].split('_')[1]
                else:
                    self.order = file.name.split('_')[1]
            except:
                if type(file) == str:
                    self.order = file.split('/')[-1].split('_')[1]
                else:
                    self.order = file.name.split('_')[1]
                print("No 'Label' column in the file")
                return
            
    def calc_public_metrics(self, metrics_public, fname, order):
        self.load_predictions(fname)
        self.order = order
        self.calc_metrics_leaderboard(metrics_public, save=True)
        self.metrics_public = self.save_metrics(
                self.root / f"{self.team_name}/Evaluation/metrics_public.csv", metrics_public, fname)
        return self.metrics_public
        
    def calc_private_metrics(self, metrics_private, fname, order):
        self.order = order
        self.calc_metrics_private(metrics_private, save=True)
        self.metrics_private = self.save_metrics(
            self.root / f"{self.team_name}/Evaluation/metrics_private.csv", metrics_private, fname)
        return self.metrics_private
    
    def calc_metrics_leaderboard(self, metrics_public, save=False, 
                                path=None, verbose=False):
        if self.yhat is None:
            print("yhat and probabilities are not defined")
            return
        else: 
            y_val = self.testy.sample(self.no_val_samples, random_state=self.seed)
            y_val_ood = self.ood_testy.sample(self.no_val_samples, random_state=self.seed)
            print(len(self.yhat), len(self.testy), len(y_val))
            print('*'*50, self.proba_bool, '*'*50)
            if len(self.yhat) != len(self.testy)   :
                print("yhat and testy have different lengths!!!!!!!")
                return
            if self.proba_bool:
                self.roc_auc = roc_auc_score(y_val, 
                                self.probabilities_1_val)
                self.roc_auc_ood = roc_auc_score(y_val_ood, 
                                self.probabilities_1_val_ood)
                
            self.h_measure = h_score(y_val.values,
                self.yhat_val.values, severity_ratio=self.sr)*0.75 + h_score(
                y_val_ood.values, self.yhat_ood_val.values, severity_ratio=self.sr)*0.25
                    
            self.MCC = matthews_corrcoef(y_val, self.yhat_val
                )*0.75 + matthews_corrcoef(y_val_ood, self.yhat_ood_val)*0.25 
            print(matthews_corrcoef(y_val, self.yhat_val), matthews_corrcoef(y_val_ood, self.yhat_ood_val))
            
            self.f1 = f1_score(y_val, self.yhat_val
                )*0.75 + f1_score(y_val_ood, self.yhat_ood_val)*0.25
            print(f1_score(y_val, self.yhat_val), f1_score(y_val_ood, self.yhat_ood_val))
            self.accuracy = accuracy_score(y_val, self.yhat_val
                )*0.75 + accuracy_score(y_val_ood, self.yhat_ood_val)*0.25
    
        if verbose:
            print("H_measure: {:.2f}, MCC: {:.4f}".format(
                self.h_measure, self.log_loss, self.MCC))
            print("f1: {:.4f}, roc_auc: {:.4f}, accuracy: {:.4f}".format(
                self.f1,self.roc_auc, self.accuracy))

    def calc_metrics_private(self, metrics_private, save=False, 
                            path=None, verbose=False):
        if self.yhat is None:
            print("yhat and probabilities are not defined")
            return
        else: 
            if len(self.yhat) != len(self.testy)   :
                print("yhat and testy have different lengths!!!!!!!")
                return
            testy = self.testy.values
            yhat = self.yhat.values
            print('*'*50, self.proba_bool, '*'*50)
            if self.proba_bool:
                self.roc_auc = roc_auc_score(self.testy, self.probabilities_1
                    )*0.75 + roc_auc_score(self.ood_testy, self.probabilities_1_ood)*0.25
                
            self.h_measure = h_score(testy, yhat, 
                        severity_ratio=self.sr)*0.75 + h_score(
                        self.ood_testy.values, self.yhat_ood.values, severity_ratio=self.sr)*0.25
                        
            self.MCC = matthews_corrcoef(testy, yhat)*0.75 + matthews_corrcoef(
                self.ood_testy, self.yhat_ood.values)*0.25
            
            self.f1 = f1_score(testy, yhat)*0.75 + f1_score(
                self.ood_testy, self.yhat_ood.values)*0.25
            
            self.accuracy = accuracy_score(testy, yhat)*0.75 + accuracy_score(
                self.ood_testy, self.yhat_ood.values)*0.25
            
        if verbose:
            print("H_measure: {:.2f}, MCC: {:.4f}".format(
                self.h_measure, self.MCC))
            print("f1: {:.4f}, roc_auc: {:.4f}, accuracy: {:.4f}".format(
                self.f1, self.roc_auc, self.accuracy))
        self.metrics_private = metrics_private

    def save_metrics(self, path, df, fname):
        self.date = datetime.now().strftime("%Y-%m-%d")
        print(self.date, self.order)
        new_row = pd.DataFrame({
                "Team": [self.team_name],
                "Date": [self.date],
                "Order": [self.order],
                "H_measure": [np.round(self.h_measure, 6)],
                "MCC": [np.round(self.MCC,6)],
                "F1": [np.round(self.f1,6)],
                "ROC_AUC": [np.round(self.roc_auc,6)],
                "Accuracy": [np.round(self.accuracy,6)],
                "fname": [fname.parts[-1]]  
            })
        df = pd.concat([df, new_row], join='outer', ignore_index=True)
        df.to_csv(path)
        return df
    
    def eval_final(self, new_metrics_private, fname):
        self.load_predictions(fname)
        self.calc_metrics_private(new_metrics_private, save=True)
        self.metrics_private = self.save_metrics(
            self.root / f"{self.team_name}/metrics_private.csv", new_metrics_private, fname)
        return self.metrics_private
    

def create_leaderboard():
    df_leaderboard = pd.DataFrame(columns=["Team", "Date", "Order", 
                "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"])
    print(df_leaderboard)
    tims = folder_teams_list + ['Baseline']
    for tim in tims:
        tim_path=Path("./Timovi/"+ tim )
        df = pd.read_csv(tim_path / 'Evaluation' / 'metrics_public.csv', index_col=0)
        df_t = df[df['MCC'] == df['MCC'].max()]
        if len(df_t) > 1:
            df_t = pd.DataFrame(df_t.iloc[0,:]).T
        df_leaderboard = pd.concat([df_leaderboard,df_t], axis=0, join='outer', ignore_index=False)

    #drop rows with NaN values
    df_leaderboard_sorted = df_leaderboard.sort_values(by='MCC', ascending=False)
    # add to dictionary spe_teams 'Baseline': 'Baseline'
    teams = spe_teams.copy()
    teams['Baseline'] = 'Baseline'
    
    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'].map(teams)
    df_selected_val = df_leaderboard_sorted[['Team', 'MCC', 'Order']].reset_index(drop=True)
    df_selected_val['Redni broj'] = df_selected_val.index + 1
    # make the 'Redni broj' column the first column
    cols = df_selected_val.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_selected_val = df_selected_val[cols]
    df_selected_val.to_html('public_leaderboard_table.html', index=False)
    df_selected_val.head(22)

def calculate_metrics_private_public(team_dir, tim_dirs):
    for tim in tim_dirs:
        path_public = tims_dir / f"{tim}/Evaluation/metrics_public.csv"
        max_number = read_df_order(path_public)
        if max_number.is_integer():
            pass
        else:
            max_number = -1
        max_number_file = []
        items = os.listdir(tims_dir / tim / "Submission")
        for item_name in items:
            print(item_name)
            if item_name.endswith('.csv') or item_name.endswith('.txt'):
                try:
                    number_str = item_name.split('_')[1]
                    if item_name.split('_')[0] == 'Tim':
                        number_str = item_name.split('_')[2]
                    if '.' in number_str:
                        number_str = number_str.split('.')[0]
                    number = int(number_str)
                    if number > max_number:
                        max_number = number
                        max_number_file.append(item_name)
                except ValueError:
                    continue
                
        tim_path = tims_dir/ f"{tim}"
        print(f"For the folder: {tim_path}, file is number: {max_number_file}")
        eval = Eval(root = tims_dir, team_name = tim, seed= seed, no_val_samples = sample)
        eval.read_data() 
        df_public = pd.read_csv(tim_path / 'Evaluation' /  "metrics_public.csv", index_col=0)
        df_private = pd.read_csv(tim_path / 'Evaluation' / "metrics_private.csv", index_col=0)
        if max_number_file is not None:
            for file_name in max_number_file:
                df_public = eval.calc_public_metrics(df_public,
                                    fname = tim_path / 'Submission' / f"{file_name}",
                                    order = file_name.split('_')[1])
                df_private = eval.calc_private_metrics(df_private, 
                                    fname = tim_path / 'Submission' / f"{file_name}",
                                    order = file_name.split('_')[1])
        else:
            # print(f"File for {tim} is not found, they didint make the submission yet.")
            continue  

def plot_submissions(metric = 'MCC', df_selected = df_selected):
    fig, axs = plt.subplots(1, 1, figsize=(16, 9))
    tims = ["Tim_{:02d}".format(i) for i in range(1, 29)] s
    markers = ['o', 'x', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', ]
    names = df_selected['Team'].values[:10]
    teams_rev = {v: k for k, v in teams.items()}
    for num, team in enumerate(names):
        tim_path = Path("./Timovi/" + teams_rev[team])
        df = pd.read_csv(tim_path / 'metrics_public.csv', index_col=0)
        if len(df)>0: 
            axs.plot(df['Order'], df[metric], label=team, 
                    marker = markers[num])
            axs.set_ylim(0, 0.35)
            axs.set_xlabel("Order", fontsize=24)
            axs.set_ylabel(metric, fontsize=24)
            
    # plot the baseline line on the subplot
    df = pd.read_csv(Path("./Timovi/Baseline/metrics_public.csv"), index_col=0)
    baseline = df[metric].max()
    axs.axhline(y=baseline, color='r', linestyle='--')
    axs.text(6, baseline, 'baseline', color = 'r', fontsize=16, ha='right')
    axs.set_title("Submissions for best score teams", fontsize=24)
    axs.legend(fontsize=18, loc='upper left')
    plt.tight_layout()
    plt.show()
    
def clear_finals():
    tims_dir = Path("./Timovi/")
    tim_dirs = ['Tim_{:02d}/final'.format(i) for i in range(1, 29)] + ['Baseline']

    for tim in tim_dirs:
        print('*'*30, tim, '*'*30)
        print('Deleting files !!!!!! ')
        items = os.listdir(tims_dir / tim)
        # Delete files in the directory
        for item in items:
            file_path = tims_dir / tim / item
            os.remove(file_path)

if __name__ == "__main__":
    folder_teams_list = list(spe_teams.keys())
    test = pd.read_csv('Challenge Phase2/test.csv')
    train = pd.read_csv('Challenge Phase2/train.csv')
    options = {
    'webdav_hostname': "",
    'webdav_login':    "",
    'webdav_password': "",
    'webdav_verbose': True,
    }

    client = Client(options)
    seed = 6
    sample = 1433
    tims_dir = Path("./Timovi/")
    tim_dirs = folder_teams_list + ['Tim_00'] # the baseline team
    calculate_metrics_private_public(tims_dir, tim_dirs)
    tim_dirs = folder_teams_list
    calculate_final_metrics(tims_dir, tim_dirs)
    create_leaderboard()
    plot_submissions('MCC')


