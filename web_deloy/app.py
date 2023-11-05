from flask import Flask, render_template, request
import pandas as pd
import joblib

model = joblib.load('web_deloy/model/rf_mushroom_pre.joblib')

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index():
    result = None
    if request.method == "POST":
        attr=[]
        for i in range(1, 21):
            attr.append(int(request.form[f"attr{i}"]))
        if attr:
            x1 = pd.DataFrame([attr])
            result = model.predict(x1)
            if result == 1:
                result = "ăn được"
            else: result = "không ăn được"
    return render_template("main.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)