from flask import Flask, render_template, request
from utils import get_y_pred

app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def index():
    result = None
    if request.method == "POST":
        attr=[]
        for i in range(1, 21):
            attr.append(int(request.form[f"attr{i}"]))
        if attr:
            result = get_y_pred(attr)
    return render_template("main.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
