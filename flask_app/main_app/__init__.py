from flask import Flask
import os

app = Flask(__name__, template_folder='templates', instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY='dev',
    DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
)

import main_app.routes

