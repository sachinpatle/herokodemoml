from flask import Flask,request,render_template
# print(flask.__version__)

import numpy as np
import pandas as pd
import joblib
model=joblib.load("studentmarkprediction.pkl")
df=pd.DataFrame()
app = Flask(__name__)
@app.route("/")
def home():
     return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
     global df
     input_features = [int(i) for i in request.form.values()]
     features_values = np.array(input_features)
     output = model.predict([features_values])
     return  render_template('index.html',predicted_marks=output)
if __name__ == "__main__":
     app.run()


