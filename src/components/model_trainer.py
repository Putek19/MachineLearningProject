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

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

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
        
