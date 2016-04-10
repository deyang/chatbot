__author__ = 'Deyang'
from flask import Flask


app = Flask(__name__)
app.secret_key = '/\xf4sH\xaa\xe2\xb9\xb3'
app.config['_secret_key'] = app.secret_key
app.config['DEBUG'] = True

import views


