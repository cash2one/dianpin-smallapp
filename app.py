from flask import Flask
from server.char_rnn import sample

app = Flask(__name__)
global tfmodel
	
tfmodel = sample.Dianpin()
tfmodel.model_built()

@app.route('/predict')
def predict():
	global tfmodel
	return tfmodel.final_predict()

if __name__ == '__main__':
	global tfmodel
	app.run('0.0.0.0', port = 45723)
