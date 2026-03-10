import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from hmeasure import h_score
from datetime import datetime
from pathlib import Path
from utils import read_df_order
      
TEAMS = {
    "Baseline": "Baseline",
    "Tim_01": "accelerate healthcare",
    "Tim_02": "Artificial Innovators",
    "Tim_03": "ATMC",
    "Tim_04": "Blizanci",
    "Tim_05": "Care4You",
    "Tim_06": "Foo",
    "Tim_07": "Hessijanci",
    "Tim_08": "IMPERIA",
    "Tim_09": "Kotao",
    "Tim_10": "Kronos",
    "Tim_11": "LIMMB",
    "Tim_12": "LMHS 1",
    "Tim_13": "LMHS 2",
    "Tim_14": "MediBoost",
    "Tim_15": "MedIQ",
    "Tim_16": "Miracool",
    "Tim_17": "Orion",
    "Tim_18": "Panacea",
    "Tim_19": "ReAdmit",
    "Tim_20": "RehopPrevent",
    "Tim_21": "SiraćHC",
    "Tim_22": "tim ReAdmitNet",
    "Tim_23": "Tim Tim",
    "Tim_24": "zesoi",
    "Tim_25": "ZMF",
    "Tim_26": "K-nearest winners",
    "Tim_27": "",
    "Tim_28": "C tim",
}

def clean():
    tim_dirs = ['Tim_{:02d}'.format(i) for i in range(1, 29)] + ['Baseline']
    for i in tim_dirs:
        submissions_folder_path="./EDIH_AI4Health_Challenge/" + i 
        os.makedirs(submissions_folder_path, exist_ok=True)
        df = pd.DataFrame(columns=["Team", "Date", "Order", 
                    "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"
                ])
        df.to_csv(submissions_folder_path + '/metrics_public.csv')
        df.to_csv(submissions_folder_path + '/metrics_private.csv')


def reset_metrics_public(root_dir, tim_dir):

    path_public = root_dir / tim_dir / f"/metrics_public.csv"
    print(path_public)
    df_public = pd.DataFrame(columns=["Team", "Date", "Order", 
                    "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy", "fname"
                ])
    df_private = pd.DataFrame(columns=["Team", "Date", "Order",
                    "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy", "fname"
                ])
    submissions_folder_path = root_dir / tim_dir / 'Submission'
    if tim_dir == 'Baseline':
        submissions_folder_path = root_dir / tim_dir
    number_list = []
    items = os.listdir(submissions_folder_path)

    for item in items:
        if item.endswith('.csv') or item.endswith('.txt'):
            if item not in [ 'metrics_private.csv',
                        'metrics_public.csv',
                        'submission_template.csv', 'Upute-28-02-2024.txt']:
                try:
                    number_str = item.split('_')[1]
                    if item.split('_')[0] == 'Tim':
                        number_str = item.split('_')[2]
                    if '.' in number_str:
                        number_str = number_str.split('.')[0]
                    number = int(number_str)
                    eval = Eval(root = root_dir, team_name = tim_dir)
                    eval.read_data()
                    print(item, number)
                    df_public = eval.calc_public_metrics(df_public,
                                    fname = submissions_folder_path / f"{item}",
                                    order = number)
                    df_private = eval.calc_private_metrics(df_private, 
                                    fname = submissions_folder_path / f"{item}",
                                    order = number)
                    print('DF PRIVATE SUCCESSFULLY UPDATED')
                except ValueError:
                    pass
    return df_public, df_private

def clean_mp2(tim):
    path = Path(f"./Phase 1/EDIH_AI4Health_Challenge/{tim}/metrics_private2.csv")
    df = pd.read_csv(path)
    df_original = pd.read_csv(Path(f"./Phase 1/EDIH_AI4Health_Challenge/{tim}/metrics_public.csv"))
    #keep rows that are in df_original names are in fname column 
    try:
        df = df[df['fname'].isin(df_original['fname'])]
    except:
        pass
    df.to_csv(path)
    
for tim in os.listdir("./Phase 1/EDIH_AI4Health_Challenge/"):
    df_public, df_private = reset_metrics_public(Path("./Phase 1/EDIH_AI4Health_Challenge/"), tim)
    # clean_mp2(tim)

def validation_leaderboard():
    df_leaderboard = pd.DataFrame(columns=["Team", "Date", "Order", 
                "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"])
    tims = ["Tim_{:02d}".format(i) for i in range(1, 29)] + ["Baseline"]
    for tim in tims:
        tim_path=Path("./EDIH_AI4Health_Challenge/"+ tim )
        df = pd.read_csv(tim_path / 'metrics_public.csv', index_col=0)
        df_t = df[df['MCC'] == df['MCC'].max()]
        if len(df_t) > 1:
            df_t = pd.DataFrame(df_t.iloc[0,:]).T
        df_leaderboard = pd.concat([df_leaderboard,df_t], axis=0, join='outer', ignore_index=False)

    #drop rows with NaN values
    df_leaderboard_sorted = df_leaderboard.sort_values(by='MCC', ascending=False)
    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'].map(TEAMS)
    df_selected_val = df_leaderboard_sorted[['Team', 'MCC', 'Order']].reset_index(drop=True)
    df_selected_val['Redni broj'] = df_selected_val.index + 1
    # make the 'Redni broj' column the first column
    cols = df_selected_val.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_selected_val = df_selected_val[cols]
    df_selected_val.to_html('leaderboard_table.html', index=False)
    df_selected_val.head(22)
    return df_selected_val, df_leaderboard_sorted

def final_evaluation():
    tims_dir = Path("./EDIH_AI4Health_Challenge/")
    tim_dirs = ['Tim_{:02d}/final'.format(i) for i in range(1, 29)] + ['Baseline']

    for tim in tim_dirs:
        path_final_private = tims_dir / f"{tim}/metrics_private.csv"
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
        tim_path = tims_dir/ f"{tim}"
        print(f"For the folder: {tim_path}, file is: {max_number_file}")
        if max_number_file is not None:
            eval = Eval(root = tims_dir, team_name = tim)
            eval.read_data() 
            df_private = pd.DataFrame(columns=["Team", "Date", "Order",
                            "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy", "fname"])
            eval.eval_final(df_private, fname = tim_path / f"{max_number_file}")
        else:
            pass
        
        return df_private
    
def plot_submissions(metric = 'MCC'):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    tims = ["Tim_{:02d}".format(i) for i in range(1, 29)] + ["Baseline"]
    # Iterate over the subplots and plot the TEAMS
    markers = ['o', 'x', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', ]
    for i, ax in enumerate(axs.flat):
        # Get the TEAMS to plot in the current subplot
        start_index = i * 7
        end_index = start_index + 7
        consecutive_tims = tims[start_index:end_index]
        # Plot the TEAMS in the current subplot
        for num, team in enumerate(consecutive_tims):
            tim_path = Path("./EDIH_AI4Health_Challenge/" + team)
            df = pd.read_csv(tim_path / 'metrics_public.csv', index_col=0)
            if len(df)>0: 
                ax.plot(df['Order'], df[metric], label=TEAMS[team], 
                        marker = markers[num])
                # set axis y limit to 0.35
                ax.set_ylim(0, 0.35)
                ax.set_xlabel("Order")
                ax.set_ylabel(metric)
                
        # plot the baseline line on the subplot
        df = pd.read_csv(Path("./EDIH_AI4Health_Challenge/Baseline/metrics_public.csv"), index_col=0)
        baseline = df[metric].max()
        ax.axhline(y=baseline, color='r', linestyle='--')
        # add text to the baseline line on the subplot 'baseline'
        ax.text(6, baseline, 'baseline', color = 'r', fontsize=16, ha='right')
        # Set the subplot title and labels
        ax.set_title("Submissions for teams: {} - {}".format(i*7+1, i*7+7))
        ax.legend()
        
    plt.tight_layout()
    plt.show()
    
def final_leaderboard()
    df_leaderboard = pd.DataFrame(columns=["Team", "Date", "Order", 
                "H_measure", "MCC", "F1", "ROC_AUC", "Accuracy"])
    tims = ["Tim_{:02d}/final".format(i) for i in range(1, 29)]
    for tim in tims:
        tim_path=Path("./EDIH_AI4Health_Challenge/"+ tim )
        df = pd.read_csv(tim_path / 'metrics_private.csv', index_col=0)
        df = df[df['MCC'] == df['MCC'].max()]
        df_leaderboard = pd.concat([df_leaderboard,df], axis=0, join='outer', ignore_index=False)
        # df_leaderboard = df_leaderboard.append(df[df['MCC'] == df['MCC'].max()])

    df_leaderboard_sorted = df_leaderboard.sort_values(by='MCC', ascending=False)
    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'
                                ].apply(lambda x: x.split('/')[0])

    df_leaderboard_sorted['Team'] = df_leaderboard_sorted['Team'].map(TEAMS)
    df_selected = df_leaderboard_sorted[['Team', 'MCC', 
                    'H_measure', "F1", "ROC_AUC"]].reset_index(drop=True)

    df_selected['H_measure'] = df_selected['H_measure'].apply(lambda x: round(x, 4))
    df_selected['Redni broj'] = df_selected.index + 1
    # make the 'Redni broj' column the first column
    cols = df_selected.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_selected = df_selected[cols]
    df_selected.to_html('leaderboard_table_final.html', index=False)

    names_sel = df_selected['Team'].values[:11]
    TEAMS_REV = {v: k for k, v in TEAMS.items()}
    keep_tim = [TEAMS_REV[i] for i in names_sel]
    keep_tim
    # new dict from lists names_sel and keep_tim
    selected_teams_dic = {keep_tim[i]: names_sel[i] for i in range(len(names_sel))}
    selected_teams_dic
    return df_selected

def plot_submissions(metric = 'MCC', df_selected = None):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(1, 1, figsize=(16, 9))
    tims = ["Tim_{:02d}".format(i) for i in range(1, 29)] 
    # Iterate over the subplots and plot the teams
    markers = ['o', 'x', 's', 'D', '^', 'v', 'p', 'P', '*', 'X', 'd', ]
    names = df_selected['Team'].values[:10]
    # reverse teams dictionary
    TEAMS_REV = {v: k for k, v in TEAMS.items()}
        # Get the TEAMS to plot in the current subplot
    for num, team in enumerate(names):
        tim_path = Path("./EDIH_AI4Health_Challenge/" + TEAMS_REV[team])
        df = pd.read_csv(tim_path / 'metrics_public.csv', index_col=0)

        if len(df)>0: 
            axs.plot(df['Order'], df[metric], label=team, 
                    marker = markers[num])
            # set axis y limit to 0.35
            axs.set_ylim(0, 0.35)
            # font size of the x and y axis to 24
            
            axs.set_xlabel("Order", fontsize=24)
            axs.set_ylabel(metric, fontsize=24)
    # plot the baseline line on the subplot
    df = pd.read_csv(Path("./EDIH_AI4Health_Challenge/Baseline/metrics_public.csv"), index_col=0)
    baseline = df[metric].max()
    axs.axhline(y=baseline, color='r', linestyle='--')
    # add text to the baseline line on the subplot 'baseline'
    axs.text(6, baseline, 'baseline', color = 'r', fontsize=16, ha='right')
    # Set the subplot title and labels
    axs.set_title("Submissions for best score TEAMS", fontsize=24)
    axs.legend(fontsize=18, loc='upper left')
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    

class Eval():
    def __init__(self, root = '', team_name = 'Tim_00') -> None:

        self.yhat = None
        self.team_name = team_name
        self.date = None
        self.Label = []
        self.trainy = []
        self.testy = None
        self.test_hat = None
        self.val_index = None
        self.val_ood_index = None
        self.test_index = None
        self.proba_bool = False
        self.order = None
        self.probabilities_0 = None
        self.probabilities_1 = None
        self.root = root
        self.h_measure = None
        self.MCC = None
        self.brier_score = None
        self.f1 = None
        self.roc_auc = None
        self.accuracy = None
        self.metrics_private = None
        self.metrics_public = None
        
    def read_data(self):
        test = pd.read_csv(Path('Challenge synthetic/test-synthetic_over_shuffled.csv'))
        self.testy = test['Label']
        self.val_index = test.sample(3154, random_state=32).index
        train = pd.read_csv(Path('Challenge synthetic/train.csv'))
        self.trainy = train['Label']
        self.severity_ratio = self.trainy.value_counts(normalize=True)[1]
        self.train = train
        self.test = test

    def load_predictions(self, file):
            print(file)
            self.test_hat = pd.read_csv(file)
            try:
                self.yhat = self.test_hat['Label']
                self.yhat_val = self.test_hat['Label'].sample(3154, random_state=32)
                # check if probabilities are in the file
                if 'Probability_0' in self.test_hat.columns and 'Probability_1' in self.test_hat.columns:
                    self.proba_bool = True
                    self.probabilities_0 = self.test_hat['Probability_0'].values
                    self.probabilities_1 = self.test_hat['Probability_1'].values
                    self.probabilities_1_val = self.test_hat['Probability_1'].sample(3154, random_state=32)
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
            
    def calc_public_metrics(self, metrics_public, fname, order=0):
        self.load_predictions(fname)
        self.order = order
        self.calc_metrics_leaderboard(metrics_public, save=True)
        self.metrics_public = self.save_metrics(
                self.root / f"{self.team_name}/metrics_public2.csv", metrics_public, fname)
        return self.metrics_public
        
    def calc_private_metrics(self, metrics_private, fname, order=0, save = True):
        self.order = order
        self.calc_metrics_private(metrics_private, save=save)
        self.metrics_private = self.save_metrics(
            self.root / f"{self.team_name}/metrics_private2.csv", metrics_private, fname)
        return self.metrics_private
    
    def calc_metrics_leaderboard(self, metrics_public, save=False, path=None, verbose=False):
        if self.yhat is None:
            print("yhat and probabilities are not defined")
            return
        else: 
            y_val = self.testy.sample(3154, random_state=32)
            print('*'*50, self.proba_bool, '*'*50)
            if len(self.yhat) != len(self.testy)   :
                print("yhat and testy have different lengths!!!!!!!")
                return
            if self.proba_bool:
                self.roc_auc = roc_auc_score(y_val, 
                                self.probabilities_1_val)
                
            self.h_measure = h_score(y_val.values,
                    self.yhat_val.values, severity_ratio=self.severity_ratio)
            self.MCC = matthews_corrcoef(y_val, self.yhat_val)
            self.f1 = f1_score(y_val, self.yhat_val)
            self.accuracy = accuracy_score(y_val, self.yhat_val)

        if verbose:
            print("H_measure: {:.2f}, MCC: {:.4f}".format(
                self.h_measure, self.log_loss, self.MCC))
            print("f1: {:.4f}, roc_auc: {:.4f}, accuracy: {:.4f}".format(
                self.f1,self.roc_auc, self.accuracy))

    def calc_metrics_private(self, metrics_private, save=False, path=None, verbose=False):
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
                self.roc_auc = roc_auc_score(
                self.testy, 
                self.probabilities_1)
                
            self.h_measure = h_score(testy, yhat, 
                        severity_ratio=self.severity_ratio)
            self.MCC = matthews_corrcoef(testy, yhat)
            self.f1 = f1_score(testy, yhat)
            self.accuracy = accuracy_score(testy, yhat)
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
                "H_measure": [self.h_measure],
                "MCC": [self.MCC],
                "F1": [self.f1],
                "ROC_AUC": [self.roc_auc],
                "Accuracy": [self.accuracy],
                "fname": [fname.parts[-1]]
            })
        
        df = df.append(new_row, ignore_index=True)
        df.to_csv(path)
        return df
    
    def clean_self(self):
        self.date = None
        self.h_measure = None
        self.MCC = None
        self.brier_score = None
        self.roc_auc = None
        self.accuracy = None
        self.f1 = None
        self.metrics_private = None
        self.metrics_public = None
    
    def eval_final(self, new_metrics_private, fname):
        self.load_predictions(fname)
        self.calc_metrics_private(new_metrics_private, save=True)
        self.metrics_private = self.save_metrics(
            self.root / f"{self.team_name}/metrics_private.csv", new_metrics_private, fname)
        return self.metrics_private
    
if __name__ == "__main__":

    df_public, df_private = reset_metrics_public(Path("./Phase 1/EDIH_AI4Health_Challenge/"), 'Tim_01')

    tims_dir = Path("./EDIH_AI4Health_Challenge/")
    tim_dirs = ['Tim_{:02d}'.format(i) for i in range(1, 29)] + ['Baseline']
    tim_dirs = ['Tim_{:02d}'.format(i) for i in range(1, 29)]


    for tim in tim_dirs:
        path_public = tims_dir / f"{tim}/metrics_public.csv"
        print(path_public)
        max_number = read_df_order(path_public)
        if max_number.is_integer():
            pass
        else:
            max_number = -1
        max_number_file = None
        items = os.listdir(tims_dir / tim)
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
                        max_number_file = item_name
                except ValueError:
                    continue
        tim_path = tims_dir/ f"{tim}"
        print(f"For the folder: {tim_path}, file is number: {max_number_file}")
        eval = Eval(root = tims_dir, team_name = tim)
        eval.read_data() 
        df_public = pd.read_csv(tim_path / "metrics_public.csv", index_col=0)
        df_private = pd.read_csv(tim_path / "metrics_private.csv", index_col=0)
        if max_number_file is not None:
            eval.calc_public_metrics(df_public,
                                    fname = tim_path / f"{max_number_file}",
                                    order = number)
            eval.calc_private_metrics(df_private, 
                                    fname = tim_path / f"{max_number_file}",
                                    order = number)
        else:
            # print(f"File for {tim} is not found, they didint make the submission yet.")
            continue
    validation_leaderboard()
    plot_submissions('MCC')
    plot_submissions('H_measure')
    df_private = final_evaluation()
    df_selected = final_leaderboard()
    plot_submissions('MCC', df_selected)