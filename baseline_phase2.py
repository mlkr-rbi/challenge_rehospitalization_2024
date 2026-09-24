import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from hmeasure import h_score
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import ADASYN, SMOTENC

import warnings
warnings.filterwarnings("ignore")

from utils import Eval, evaluate_m

c_map = {'MediBoost': '#01b4dd',
'SiraćHC': '#f29042',
'ZMF': '#d83187',
'IMPERIA': '#24195c',
 'Kronos': '#212529',
 'Baseline': '#bab9bb',
 'ATMC': '#e31a1c',
 'LHMS 1': '#fffc00',
 'Artificial Innovators': '#228b22',
 'irb-red': '#d92725',
 'irb-blue': '#22458b',
 'irb-gray': '#bab9bb',
 'Challengers': '#988fca',}
cmap = c_map
FONTSIZE = 36
FONTNAME = 'Sans'
MARKERSCALE = 4

# %%
import matplotlib as mpl  
mpl.rc('font',family=FONTNAME)

# %%
class Dataset:
    def __init__(self):
        
        self.load_data()
        self.numerical = ["LOS", "LOS_ICU",   
            "Weight_Discharge", "Height_Discharge", "AdmissionYear", 
            ]
        self.categorical = ["AdmissionDx","AdmissionType",
            "Age_Group", "Gender","Surgery_Count", "Discharge_Specialty",
            "Dx_Discharge", "Discharge_Status","Education","Current_Work_Status", 
            'PreviousAdmissionDays',  # 'Total_patient_visits',
            ]
        self.multi_cat = ["Dx_Secondary","Administered_Drugs"]
        self.drop_col = ['VISIT_COUNT','PATIENT_CARDINALITY',
            'NextAdmissionDays', 'Glukose',"ERAdmissionCount",
            'Potassium', 'Sodium', 'Creatinine', 'Urea', 'CRP',
            'Patient'
            ]
        self.rare_drugs_col = None
        self.h_measure = 0
        self.MCC = 0
        self.predictions = []
        self.Label = []    
        self.train = None
        self.test = None
        self.ood_test = None
        self.fixed_validation_samples = []
        self.fixed_leaderboard_samples = []
        self.cat_encoder_maps = {}
        self.multi_cat_enc_maps = {}
        self.scale_params_maps = {}

    def mcc_calc(self):
        # Calculation of MCC
        self.MCC = matthews_corrcoef(self.test['Label'], self.predictions)
        
    def load_data(self):
        self.df = pd.read_csv("C:\\Experiments\\chpkg\\Data\\dataset.csv", index_col=0)
        self.train = pd.read_csv("C:\\Experiments\\chpkg\\Challenge Phase2\\old\\train.csv")
        self.train.Dx_Secondary = self.train.Dx_Secondary.apply(
                lambda x: np.array(['missing'])  if type(x
                ) is float else np.array(x[2:-2].split("' '")))
        
        self.test = pd.read_csv("C:\\Experiments\\chpkg\\Challenge Phase2\\test_iid_labels.csv")
        self.test.Dx_Secondary = self.test.Dx_Secondary.apply(
                lambda x: np.array(['missing']) if type(x
                ) is float else np.array(x[2:-2].split("' '")))
        
        self.ood_test = pd.read_csv("C:\\Experiments\\chpkg\\Challenge Phase2\\test_ood_labels.csv")
        self.ood_test.Dx_Secondary = self.ood_test.Dx_Secondary.apply(
                lambda x: np.array(['missing']) if type(x
                ) is float else np.array(x[2:-2].split("' '")))

        self.Label = self.train['Label']
    
    def encode_cat_var(self):
        # Initialize dictionaries to store encoder mappings
        self.cat_encoder_maps = {}
        self.multi_cat_enc_maps = {}
        # Encode ordinal variables
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(pd.concat(
            [self.train[self.categorical],self.test[self.categorical],
                self.ood_test[self.categorical]]))
        
        self.cat_encoder_maps.update({col: {
            cat: idx for idx, cat in enumerate(encoder)
                } for col, encoder in zip(self.categorical, ordinal_encoder.categories_)})
        
        ordinal_encoded = ordinal_encoder.transform(
                            self.train[self.categorical])
        self.train.loc[:,self.categorical] = ordinal_encoded
        
        ordinal_encoded_test = ordinal_encoder.transform(
                            self.test[self.categorical])
        self.test.loc[:,self.categorical] = ordinal_encoded_test
        
        ordinal_encoded_ood_test = ordinal_encoder.transform(
                            self.ood_test[self.categorical])
        
        self.ood_test.loc[:,self.categorical] = ordinal_encoded_ood_test

    def encode_numerical_var(self, scaling_func=MinMaxScaler):
        #impute missing values with median
        imputer = SimpleImputer(strategy='median')
        self.train[self.numerical] = imputer.fit_transform(self.train[self.numerical])
        # Initialize dictionary to store scaling parameters and scale values
        self.scale_params_maps = {}
        scaler = scaling_func()
        scaled_numerical = scaler.fit_transform(self.train[self.numerical])
        # Save scaling parameters
        self.scale_params_maps.update({col: {
            'min': scaler.data_min_[idx], 'max': scaler.data_max_[idx]
            } for idx, col in enumerate(self.numerical)})
    
        # Replace original numerical columns with scaled values
        self.train[self.numerical] = scaled_numerical
        scaled_numerical_test = scaler.transform(self.test[self.numerical])
        self.test[self.numerical] = scaled_numerical_test
    
    def ordinal_encode(self):
        def ordinal_encode_secondary(train):
            train.Dx_Discharge = train.Dx_Discharge.apply(
                lambda x: np.nan if type(x) == float else x[:1])
            train.AdmissionDx = train.AdmissionDx.apply(
                lambda x: np.nan if type(x) == float else x[:1])

            init_counts = {}
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                init_counts[letter] = 0
            l = []
            for value in train.Dx_Secondary.values:
                letter_counts = init_counts.copy()
                for v in value:
                    if v == 'missing':
                        pass
                    else:
                        letter = v[:1]
                        letter_counts[letter] = letter_counts.get(letter, 0) + 1
                l.append(letter_counts)
            # Create a new dataframe from the list of dictionaries
            new_df = pd.DataFrame()
            for k in l:
                new_df = new_df.append(k, ignore_index=True)
            new_df.index = train.index
            # Add the new columns to the existing dataframe
            train = pd.concat([train, new_df], axis=1)
            return train
        self.train = ordinal_encode_secondary(self.train)
        self.test = ordinal_encode_secondary(self.test) 
        self.ood_test = ordinal_encode_secondary(self.ood_test)

    def clean_data(self):
        # self.train = self.train.drop(
        #     columns=self.drop_col+self.multi_cat)
        # self.test = self.test.drop(
        #     columns=self.drop_col+self.multi_cat)
        # self.ood_test = self.ood_test.drop(
        #     columns=self.drop_col+self.multi_cat)
        self.train = self.train.drop(
            columns=self.multi_cat)
        self.test = self.test.drop(
            columns=self.multi_cat)
        self.ood_test = self.ood_test.drop(
            columns=self.multi_cat)
        self.encode_cat_var()
        self.encode_numerical_var()
        def remove_cols():
            self.train = self.train.drop(
                columns=self.train.iloc[:, 17:1825].columns)
            self.test = self.test.drop(
                columns=self.test.iloc[:, 17:1825].columns)
            self.ood_test = self.ood_test.drop(
                columns=self.ood_test.iloc[:, 17:1825].columns)
        # remove_cols()
        
    def impute_missing_values(self):
        imputer = SimpleImputer(strategy='median')
        nan_columns = self.train.columns[self.train.isna().any()].tolist()
        print(nan_columns)
        self.train[nan_columns] = imputer.fit_transform(self.train[nan_columns])
        self.test[nan_columns] = imputer.transform(self.test[nan_columns])
        self.ood_test[nan_columns] = imputer.transform(self.ood_test[nan_columns])
        nan_columns = self.test.columns[self.test.isna().any()].tolist()
        imputer = SimpleImputer(strategy='median')
        self.test[nan_columns] = imputer.fit_transform(self.test[nan_columns])
        nan_columns = self.ood_test.columns[self.ood_test.isna().any()].tolist()
        self.ood_test[nan_columns] = imputer.fit_transform(
            self.ood_test[nan_columns])
    
    def preprocess(self):
        self.load_data()
        self.clean_data()
        self.impute_missing_values()

from hyperopt.pyll.base import scope
from sklearn.metrics import roc_curve, precision_recall_curve

class MLModel:
    def __init__(self, model, D) -> None:
        self.model = model
        self.train = D.train.drop('Label', axis=1)
        self.test = D.test.drop('Label', axis=1)
        self.ood_test = D.ood_test.drop('Label', axis=1)
        self.y = D.train['Label']
        self.ytest = D.test['Label']
        self.yood_test = D.ood_test['Label']
        self.predictions = None
        self.h_measure = None
        self.mcc = None
        self.yhat = None
        self.ood_yhat = None
        self.val = None
        self.yval = None
        self.best_params = None
        self.probabilities = None
        self.ood_probabilities = None
        self.zero_columns = None
        self.score = 0  
        self.categorical = D.train.columns ^ D.numerical
        self.numerical = D.numerical
        self.cat_features = None
        self.multi_cat_enc_maps = {}
        self.cat_encoder_maps = {}
        self.scale_params_maps = {}
    
    def clean_data(self):
        zero_columns = self.train.columns[(self.train == 0).all(axis=0)]
        self.categorical = self.categorical ^ zero_columns.tolist()
        print(zero_columns, len(zero_columns))
        #drop columns with all zeros in train and test
        self.train = self.train.drop(columns=zero_columns)
        self.test = self.test.drop(columns=zero_columns)
        self.ood_test = self.ood_test.drop(columns=zero_columns)
        self.categorical = self.categorical ^ ['Label']
        self.train[self.categorical] = self.train[self.categorical].astype(int)
        self.cat_features = [int(
                self.train.columns.get_loc(c)) for c in self.categorical]
        
    def oversample_smotenc(self, ratio = 0.2, val=False, test_size = 0.25, 
                        funny_val=False, seed = 4):
        smote = SMOTENC(categorical_features=self.cat_features, 
                        sampling_strategy=ratio, 
                        random_state=seed)
        if val:
            self.train, self.val, self.y, self.yval = train_test_split(
                self.train, self.y, test_size=test_size, random_state=seed)
        if funny_val:
            _, self.val, _, self.yval = train_test_split(
                self.train, self.y, test_size=test_size, random_state=seed)
        self.train, self.y = smote.fit_resample(self.train, self.y)
    
    def oversample_adasyn(self, val=False, test_size = 0.25, seed = 4, 
                        best_model=False, verbose=0):
        adasyn = ADASYN(sampling_strategy='minority', random_state=seed)
        if val:
            self.train, self.val, self.y, self.yval = train_test_split(
                self.train, self.y, test_size=test_size, random_state=seed)
        if best_model:
            best_t, best_y = adasyn.fit_resample(self.val, self.yval)
            return best_t, best_y
        else:
            self.train, self.y = adasyn.fit_resample(self.train, self.y)

    def parameters_decision_tree(self):
            # Decision Tree parameters for hyperopt
        params = {
            'max_depth': hp.choice('max_depth', [None, hp.quniform(
                    'max_depth_int', 1, 20, 1)]),
            'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
            'max_features': hp.choice(
                    'max_features', ['auto', 'sqrt', 'log2', None]),
            'criterion': hp.choice('criterion', ["gini", "entropy"])
        }
        return params

    def parameters_rf_classifier(self):
        # Random Forest Classifier parameters for hyperopt
        params = {
            'n_estimators': scope.int(hp.quniform(
                'n_estimators', 10, 1000, 1)),
            'max_depth':  scope.int(hp.quniform('max_depth', 5, 20, 1)),
            'min_samples_split': scope.int(hp.quniform(
                'min_samples_split', 3, 7, 1)),
            'min_samples_leaf': scope.int(hp.quniform(
                'min_samples_leaf', 1, 9, 1)),
            'max_features':  hp.choice(
                'max_features', ['auto', 'sqrt', 'log2', None]),
            'criterion':  hp.choice('criterion', ["gini", "entropy"]),
        }
        return params

    def parameters_catboost(self):
        # CatBoost model parameters for hyperopt
        params = {
            'iterations': scope.int(hp.quniform(
                    'iterations', 40, 400, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
            'depth': scope.int(hp.choice('depth', [4, 6, 7])),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.01, .5),
            'border_count': scope.int(hp.choice(
                    'border_count', [32, 64, 128, 256])),
            "od_type": "Iter",
            "od_wait":200,
            "eval_metric":  'AUC', # only if validation set is provided
            "use_best_model": True,
            "random_seed": 4,
            "class_weights": (1, 10)
        }
        return params
        
    def evaluate_h_measure(self):
        # Calculate n1 and n0
        n1 = self.y.sum()
        n0 = self.train.shape[0] - n1
        # Calculate severity ratio
        severity_ratio = n1 / n0
        # Calculate h-score
        hscore_validation = h_score(self.ytest, 
                self.predictions, severity_ratio=severity_ratio)
        # Calculate h-score for OOD dataset
        hscore_ood = h_score(self.yood_test, 
                self.predictions, severity_ratio=severity_ratio)
        
        return hscore_validation, hscore_ood

    def hyperparameter_optimization(self, model_type='dt', evals=10, verbose=0):
        # Initialize model based on the provided flag
        if model_type == 'dt':
            param_space = self.parameters_decision_tree()
        elif model_type == 'rf':
            param_space = self.parameters_rf_classifier()            
        elif model_type == 'catboost':
            param_space = self.parameters_catboost()
        else:
            raise ValueError("Invalid model type. Choose from 'dt', 'rf', 'catboost', or 'xgboost'.")
        
        # Define objective function
        def objective(params):
            if model_type == 'rf':
                model = RandomForestClassifier(verbose=0)
            elif model_type == 'catboost':
                model = CatBoostClassifier(verbose=0)
            elif model_type == 'dt':
                model = DecisionTreeClassifier()
            
            model.set_params(**params)
            if self.val is not None:
                model.fit(self.train, self.y, eval_set=(self.val, self.yval))
                predictions = model.predict(self.val)
                predict_proba = model.predict_proba(self.val)
                score = matthews_corrcoef(self.yval, predictions)
                print(f"Score: {score}")
                self.model = model
                if verbose == 1:
                    self.eval_validation(predictions, predict_proba)
                if score > self.score:
                    self.score = score
                    self.best_model = model
            else:
                model.fit(self.train, self.y)
                predictions = model.predict(self.test)
                score = matthews_corrcoef(self.ytest, predictions)
                print(f"Score: {score}")
            return {'loss': 1-score, 'status': STATUS_OK}

        trials = Trials()
        best_params = fmin(fn=objective,
                        space=param_space,
                        algo=tpe.suggest,
                        max_evals=evals,
                        trials=trials)
        self.best_params = best_params
        self.best_params['depth'] = [4, 6, 7][best_params['depth']]
        self.best_params['border_count']= [32, 64, 128, 256][best_params['border_count']]
        self.best_params['od_type'] =  "Iter"
        self.best_params['od_wait'] = 200
        self.best_params['class_weights'] = (1, 10)
        self.best_params['random_seed'] = 4
        return best_params
    
    def eval_validation(self, predictions, predict_proba):
        eval_val = Eval(D, predictions, predict_proba, 
            self.yval.values)
        eval_val.trainy = self.y
        print('Validation set metrics:')
        eval_val.calc_metrics(verbose=True)
        print('\n')
    
    def eval_test(self, predictions, predict_proba):
        eval_val = Eval(D, predictions, predict_proba, 
            self.ytest.values)
        eval_val.trainy = self.y
        print('Test set metrics:')
        eval_val.calc_metrics(verbose=True)
        print('\n')
        
    def eval_ood_test(self, predictions, predict_proba):
        eval_val = Eval(D, predictions, predict_proba, self.yood_test.values)
        eval_val.trainy = self.y
        print('ood Test set metrics:')
        eval_val.calc_metrics(verbose=True)
        print('\n')
    
    def fit_best_params(self, model_type='dt', params=None, scale=1.02):
        if model_type == 'dt':
            model = DecisionTreeClassifier()
        elif model_type == 'rf':
            model = RandomForestClassifier()
            self.best_params['iterations'] = int(
                self.best_params['iterations']*0.9)
        elif model_type == 'catboost':
            model = CatBoostClassifier(verbose=0)
        else:
            raise ValueError("Invalid model type. Choose from 'dt', 'rf' or 'catboost'")
        if params:
            model.set_params(**params)
        else:
            self.best_params['iterations'] = int(
                self.best_params['iterations']*scale)
            model.set_params(**self.best_params)
        if self.val is not None:
            self.train = pd.concat([self.train, self.val])
            self.y = np.concatenate([self.y, self.yval])
            model.fit(self.train, self.y)
            self.model = model
        else:
            model.fit(self.train, self.y)
            self.model = model
        
        self.eval_model(self.model)
        
    def eval_model(self, model):    
        self.yhat = model.predict(self.test)
        self.probabilities = model.predict_proba(self.test)
        self.ood_yhat = model.predict(self.ood_test)
        self.ood_probabilities = model.predict_proba(self.ood_test)
        self.eval_test(self.yhat, self.probabilities)
        self.eval_ood_test(self.ood_yhat, self.ood_probabilities)
        predictions = model.predict(self.val)
        predict_proba = model.predict_proba(self.val)
        self.eval_validation(predictions, predict_proba)
        mcc = matthews_corrcoef(self.ytest, self.yhat
            )*0.75 + matthews_corrcoef(self.yood_test, self.ood_yhat)*0.25
        return mcc
    
    # calculate and plot roc auc
    def plot_roc_auc(self):
        # calculate roc curves
        fpr, tpr, thresholds = roc_curve(self.ytest, self.yhat)
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
        precision, recall, thresholds = precision_recall_curve(
            self.ytest, self.yhat)
        # plot the roc curve for the model
        no_skill = len(self.ytest[eval.testy==1]) / len(self.ytest)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', 
                label=self.model.__class__.__name__)
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        # show the plot
        plt.show()

if __name__ == "__main__":
    D = Dataset()
    D.load_data()
    D.Label.value_counts(normalize=True)
    D.preprocess()
    D.categorical = D.train.drop('Label', axis=1).columns ^ D.numerical
    model = MLModel(CatBoostClassifier(), D)
    model.clean_data()
    model.oversample_smotenc(0.1, val=True, test_size=0.25, seed=4)
    model.hyperparameter_optimization(model_type='catboost', evals=5, verbose=1)
    model.eval_model(model.best_model)