import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        Age: int,
        Sex: str,
        ChestPainType: str,
        RestingBP: float,
        Cholesterol: float,
        FastingBS: int,
        RestingECG: str,
        MaxHR: int,
        ExerciseAngina: str,
        Oldpeak: int,
        ST_Slope: str
    ):
        self.Age = Age
        self.Sex = Sex
        self.ChestPainType = ChestPainType
        self.RestingBP = RestingBP
        self.Cholesterol = Cholesterol
        self.FastingBS = FastingBS
        self.RestingECG = RestingECG
        self.MaxHR = MaxHR
        self.ExerciseAngina = ExerciseAngina
        self.Oldpeak = Oldpeak
        self.ST_Slope = ST_Slope

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Sex": [self.Sex],
                "ChestPainType": [self.ChestPainType],
                "RestingBP": [self.RestingBP],
                "Cholesterol": [self.Cholesterol],
                "FastingBS": [self.FastingBS],
                "RestingECG": [self.RestingECG],
                "MaxHR": [self.MaxHR],
                "ExerciseAngina": [self.ExerciseAngina],
                "Oldpeak": [self.Oldpeak],
                "ST_Slope": [self.ST_Slope]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

