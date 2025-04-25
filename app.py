from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = ""
    if request.method == "POST":
        message = request.form["email"]
        pred = model.predict([message])[0]
        prediction = "Spam" if pred == 1 else "Not Spam"
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
