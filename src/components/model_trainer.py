import os
import sys
from dataclasses import dataclass

from sklearn.metrics import f1_score,recall_score,accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "SVC":SVC(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "LogisticRegression":LogisticRegression(),
                "CatBoostClassifier":CatBoostClassifier(),
                "XGBClassifier":XGBClassifier()
            }

            #Decision tree params
            criterion = ['gini', 'entropy', 'log_loss']
            max_depth = [3,5,9,15,20]
            max_features = ['sqrt','log2']

            #Random Forest params
            n_estimators = [50,100,200]





            penalty = ['l1', 'l2', 'elasticnet',None]
            C = [0.01,0.1,1,0.0001,10]
            class_weight = ['balanced',None]
            solver = [ 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
            kernel = ['linear', 'poly', 'rbf', 'sigmoid']
            gamma = ['scale','auto']


            #catboost_params
            learning_rate = [0.001,0.01,0.1]

            #xgb_params
            sampling_method = ['uniform','gradient_based']
            lma = [1,2,0.1,1.1]


            #gbclassifier
            loss = ['log_loss', 'exponential']


            params_svc = dict(C=C, kernel=kernel, class_weight=class_weight, gamma=gamma)
            params_lr = dict(penalty=penalty, C=C, class_weight=class_weight, solver=solver)
            params_dt = dict(criterion=criterion, max_depth=max_depth, max_features=max_features)
            params_rf = dict(criterion=criterion, max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
            params_ctb = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
            params_xgb = dict(max_depth=max_depth, reg_lambda=lma, sampling_method=sampling_method)
            params_ada = dict(n_estimators=n_estimators, learning_rate=learning_rate)
            params_gbst = dict(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

            params = {
                    "SVC": params_svc,
                    "LogisticRegression": params_lr,
                    "DecisionTreeClassifier": params_dt,
                    "RandomForestClassifier": params_rf,
                    "CatBoostClassifier": params_ctb,
                    "XGBClassifier": params_xgb,
                    "AdaBoostClassifier": params_ada,
                    "GradientBoostingClassifier": params_gbst
                }




            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name =  list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'Best found model on both training and test dataset')


            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj = best_model)
            predicted = best_model.predict(X_test)

            recall_scoring = recall_score(y_test,predicted)
            
            return recall_scoring


        except Exception as e:
            raise CustomException(e,sys)
        
