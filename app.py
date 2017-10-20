from flask import Flask
from server import dianpin_class

app = Flask(__name__)
@app.route('/dianpin')
def dianpin():
    return dianpin_class.test_output()

@app.route('/ye')
def ye():
    return 'Ye Big Face'

@app.route('/')
def hello_world():
    return 'Hello, World!'