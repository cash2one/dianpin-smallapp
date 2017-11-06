from flask import Flask
from server.char_rnn import simple

app = Flask(__name__)
global model
	
model = dianpin.Dianpin()
model.model_built()

@app.route('/predict')
def predict():
	global model
	return model.final_predict()

if __name__ == '__main__':
	global model
	app.run('0.0.0.0', port = 45723)
