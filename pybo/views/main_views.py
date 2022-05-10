from flask import Blueprint, render_template

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('input/input.html')

@bp.route('/predict')
def hello_pybo():
    return 'Hello, Pybo!'

