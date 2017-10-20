from flask import Flask
from server import dianpin

app = Flask(__name__)

model = None
@app.route('/dianpin')
def create_model():
	global model
	model = dianpin.Dianpin()
	model.model_built()
	return "Model Loaded"
	
@app.route('/predict')
def predict():
	global model
    return model.final_predict()

@app.route('/')
def hello_world():
    return 'Hello, World!'