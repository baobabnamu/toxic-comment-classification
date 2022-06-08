# from .views import main_views, predict_classification_views, predict_filtering_views
from .views import main_views, predict_classification_views, predict_filtering_views
from flask import Flask

def create_app():
    app = Flask(__name__)

    app.register_blueprint(main_views.bp)
    app.register_blueprint(predict_classification_views.bp)
    app.register_blueprint(predict_filtering_views.bp)

    return app