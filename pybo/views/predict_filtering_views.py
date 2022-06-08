from flask import Blueprint, request, render_template
from pybo.model import toxicCommentFilteringPredict

bp = Blueprint('predict-filtering', __name__, url_prefix='/')

@bp.route('/predict-filtering', methods=['POST'])
def predict():
    inputText = request.form.get('inputText')
    outputText = toxicCommentFilteringPredict(inputText)
    return render_template('output/output2.html', outputText = outputText)