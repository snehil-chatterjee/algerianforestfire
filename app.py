import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

scaler=pickle.load(open("/config/workspace/models/scaler.pkl","rb"))
model=pickle.load(open("/config/workspace/models/ridge_model.pkl","rb"))


@app.route('/',methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get("Temperature"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        RH=float(request.form.get("RH"))
        Ws=float(request.form.get("Ws"))
        Rain=float(request.form.get("Rain"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))
        value=scaler.transform([[Temperature,FFMC,DMC,ISI,RH,Ws,Rain,Classes,Region]])
        prediction=model.predict(value)
        return render_template("Home_Page.html",result=prediction[0])
    else:
        return render_template("Home_Page.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
