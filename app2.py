from flask import Flask
from server import dianpin

app = Flask(__name__)

model = None
@app.route('/dianpin')
def create_model():
	model = dianpin.Dianpin()
	model.model_built()
	
@app.route('/predict')
def predict():
    return model.final_predict()

@app.route('/')
def hello_world():
    return 'Hello, World!'