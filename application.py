from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Mapowanie nazw z formularza HTML
            data = CustomData(
                Age=float(request.form.get('Age')),
                Sex=request.form.get('Sex'),
                ChestPainType=request.form.get('ChestPainType'),
                RestingBP=float(request.form.get('RestingBP')),
                Cholesterol=float(request.form.get('Cholesterol')),
                FastingBS=float(request.form.get('FastingBS')),
                RestingECG=request.form.get('RestingECG'),
                MaxHR=float(request.form.get('MaxHR')),
                ExerciseAngina=request.form.get('ExerciseAngina'),
                Oldpeak=float(request.form.get('Oldpeak')),
                ST_Slope=request.form.get('ST_Slope')
            )
            
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template(
                'home.html', 
                prediction_text="Wykryto chorobę serca" if results[0] == 1 else "Brak oznak choroby serca",
                input_data=request.form.to_dict()
            )
            
        except Exception as e:
            return render_template(
                'home.html', 
                error=f"Błąd przetwarzania: {str(e)}"
            )
        

if __name__ == '__main__':
    app.run(host = "0.0.0.0")