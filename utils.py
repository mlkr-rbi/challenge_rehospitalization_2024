import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import matthews_corrcoef
from hmeasure import h_score
from sklearn.metrics import (
    precision_score, precision_recall_curve,
    recall_score, f1_score, roc_auc_score, 
    accuracy_score, classification_report, confusion_matrix,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve

def read_df_order(path):
    df = pd.read_csv(path, index_col=0)
    return df['Order'].max()


class Eval():
    def __init__(self, D, 
                yhat = None, 
                probabilities = None,
                testy = None,
                team_name = None,
                date = None) -> None:

        self.yhat = yhat
        self.probabilities = probabilities
        self.team_name = team_name
        self.date = date
        self.Label = []
        self.trainy = []
        self.testy = testy
        self.val_index = [1,3,4]
        self.val_ood_index = [7,8,9]
        self.test_index = None
        self.D = D
        
        self.h_measure = 0
        self.MCC = 0
        self.brier_score = 0
        self.calibrated_probabilities = []
        self.f1 = 0
        self.precision = 0
        self.recall = 0
        self.roc_auc = 0
        self.accuracy = 0
        self.classification_report = ""
        self.confusion_matrix = []
        self.log_loss = 0
        self.ref_log_loss = 0
        self.ref_brier_score = 0
        self.brier_skill = 0
        
    def load_y(self, ood=False):
        if ood:
            self.trainy = self.D.train['Label']
            self.testy = self.D.ood_test['Label'].values
        else:
            self.trainy = self.D.train['Label']
            self.testy = self.D.test['Label'].values
    
    def calc_ref_metrics(self):
        prob = self.trainy.value_counts(normalize=True)
        arr = [[prob[0], prob[1]] for _ in range(len(self.trainy))]
        self.ref_log_loss = log_loss(self.trainy, arr)
        print('Random train cls: Log Loss=%.3f' % (self.ref_log_loss))
        proba_one = [prob[1] for _ in range(len(self.trainy))]
        self.ref_brier_score = brier_score_loss(self.trainy, proba_one)
        print('Random train cls: Brier Loss=%.3f' % (self.ref_brier_score))
    
    def brier_score_skill(self):
        model_brier = brier_score_loss(
            self.testy, self.probabilities[::,1])
        # calculate skill score
        skill = 1 - (model_brier / self.ref_brier_score)
        print('Brier Skill Score: %.3f' % skill)
        return skill
    # Methods that require probabilities and yhat
    
    def calc_metrics(self, save=False, path=None, verbose=False):
        if self.yhat is None or self.probabilities is None :
            print("yhat and probabilities are not defined")
            return
        else: 
            self.h_measure = h_score(self.testy, self.yhat)
            self.brier_skill = self.brier_score_skill()
            self.log_loss = log_loss(self.testy, self.probabilities)
            self.MCC = matthews_corrcoef(self.testy, self.yhat)
            self.f1 = f1_score(self.testy, self.yhat)
            self.precision = precision_score(self.testy, self.yhat)
            self.recall = recall_score(self.testy, self.yhat)
            self.roc_auc = roc_auc_score(
                self.testy, self.probabilities[::,1])
            self.accuracy = accuracy_score(self.testy, self.yhat)
            self.classification_report = classification_report(
                self.testy, self.yhat)
            self.confusion_matrix = confusion_matrix(
                self.testy, self.yhat)
        if verbose:
            print("H_measure: {:.2f}, brier_skill: {:.4f}, log_loss: {:.4f}, MCC: {:.4f}".format(
                self.h_measure, self.brier_skill, self.log_loss, self.MCC))
            print("f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, roc_auc: {:.4f}, accuracy: {:.4f}".format(
                self.f1, self.precision, self.recall, self.roc_auc, self.accuracy))
            print("classification_report: \n", self.classification_report)
            print("confusion_matrix: \n", self.confusion_matrix)
            # load metrics df from Data/metrics_hidden.csv
            # append new metrics
            # save to Data/metrics_hidden.csv
        if save:
            self.save_metrics("Data/metrics_hidden.csv")
    
    def calc_metrics_leaderboard(self, save=False, path=None, verbose=False):
        if self.yhat is None or self.probabilities is None :
            print("yhat and probabilities are not defined")
            return
        else: 
            y_val = self.testy[self.val_index]
            y_prob = self.probabilities[self.val_index][::,1]
            yhat = self.yhat[self.val_index]
            self.h_measure = h_score(y_val, yhat)
            self.brier_skill = self.brier_score_skill()
            self.log_loss = log_loss(y_val, 
                        self.probabilities[self.val_index])
            
            self.MCC = matthews_corrcoef(y_val, yhat)
            self.f1 = f1_score(y_val, yhat)
            self.precision = precision_score(y_val, yhat)
            self.recall = recall_score(y_val, yhat)
            self.roc_auc = roc_auc_score(self.testy, 
                                self.probabilities)
            self.accuracy = accuracy_score(y_val, yhat)
            self.classification_report = classification_report(y_val, yhat)
            self.confusion_matrix = confusion_matrix(y_val, yhat)
        if verbose:
            print("H_measure: {:.2f}, brier_skill: {:.4f}, log_loss: {:.4f}, MCC: {:.4f}".format(
                self.h_measure, self.brier_skill, self.log_loss, self.MCC))
            print("f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, roc_auc: {:.4f}, accuracy: {:.4f}".format(
                self.f1, self.precision, self.recall, self.roc_auc, self.accuracy))
            print("classification_report: \n", self.classification_report)
            # prety plot confusion matrix
            print("confusion_matrix test: \n", self.confusion_matrix)
        if save:
            self.save_metrics("Data/metrics_public.csv")
            
    def plot_roc_auc(self):
            # calculate roc curves
        fpr, tpr, thresholds = roc_curve(self.testy, self.yhat)
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label=self.model.__class__.__name__)
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()
    
    def plot_pr_curve(self):
        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(self.testy, self.yhat)
        # plot the roc curve for the model
        no_skill = len(self.testy[self.testy==1]) / len(self.testy)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label=self.model.__class__.__name__)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        # show the plot
        plt.show()
            
    def save_metrics(self, path):
        df = pd.read_csv(path, index_col=0)
        new_row = pd.DataFrame({
                "Team": [self.team_name],
                "Date": [self.date],
                "H_measure": [self.h_measure],
                "MCC": [self.MCC],
                "F1": [self.f1],
                "Precision": [self.precision],
                "Recall": [self.recall],
                "ROC_AUC": [self.roc_auc],
                "Accuracy": [self.accuracy],
                "Log_Loss": [self.log_loss],
                "Brier_Skill": [self.brier_skill]
            })
        df = df.append(new_row, ignore_index=True)
        df.to_csv(path)
    
    def plot_gmeans(self):   
        fpr, tpr, thresholds = roc_curve(self.testy, self.yhat)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='DT')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()
    
    def plot_calibration_curve(self):
        # reliability diagram
        fop, mpv = calibration_curve(self.testy, self.probabilities[::,1], n_bins=10, normalize=True)
        # plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot model reliability
        plt.plot(mpv, fop, marker='.')
        plt.show()
        
    def plot_f1_pr(self):
        precision, recall, thresholds = precision_recall_curve(self.testy, self.yhat)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
        # plot the roc curve for the model
        no_skill = len(self.testy[self.testy==1]) / len(self.testy)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        # show the plot
        plt.show()
        
        
def evaluate_m(m, D,  ood=False, team_name="Team1", date=datetime.now()):
    if ood:
        eval = Eval(D, 
                yhat = m.predict(D.ood_test.drop('Label', axis=1)), 
                probabilities = m.predict_proba(D.ood_test.drop('Label', axis=1)),
                team_name = team_name,
                date = date)
        eval.load_y(ood=True)
    else:
        eval = Eval(D, 
                yhat = m.predict(D.test.drop('Label', axis=1)), 
                probabilities = m.predict_proba(D.test.drop('Label', axis=1)),
                team_name = team_name,
                date = date)
        eval.load_y()
    eval.calc_metrics(verbose=True)
    return eval

def read_pwd():
    with open('Data/pwd.txt') as f:
        return f.readline().strip()