from flask import Blueprint, request, render_template
from pybo.model import toxicCommentClassficiationPredict

bp = Blueprint('predict-classification', __name__, url_prefix='/')

@bp.route('/predict-classification', methods=['POST'])
def predict():
    inputText = request.form.get('inputText')
    outputText = toxicCommentClassficiationPredict(inputText)
    return render_template('output/output.html', inputText = inputText, outputText = outputText)