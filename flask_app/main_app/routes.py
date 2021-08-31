from . import app
from flask import render_template


@app.route('/')
def index():
    # text = 'hello world'
    # return render_template('templates/base.html')
    user = {'username': 'roman'}
    # return render_template('index.html', title='Home111', user=user)
    return render_template('register.html', title='app v1', user=user)
    # return 'index1'


