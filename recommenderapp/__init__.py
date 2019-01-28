from flask import Flask
from dash import Dash


server = Flask(__name__)
app = Dash(__name__, server = server)

# Init the layout of the app
from recommenderapp import index

# app = Flask(__name__)
from recommenderapp import routes
