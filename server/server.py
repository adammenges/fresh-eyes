from flask import Flask
app = Flask(__name__)

@app.route("/getPrediction")
def getPrediction():
	return "Prediction!"
