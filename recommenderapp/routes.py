from recommenderapp import app
import flask
import os

ASSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

@app.server.route('/')
@app.server.route('/assets/<resource>')
def img(resource):
    return flask.send_from_directory(ASSET_PATH, resource)

