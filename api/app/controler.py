import logging

import pandas as pd
from flask import request
from werkzeug.utils import secure_filename

from app import app
from app.model import setup
from app.config import MAIN_DIR

logging.basicConfig(level=logging.INFO)


@app.route("/api/model/retrain", methods=["GET"])
def retrain_model():
    pass


@app.route("/api/prediction", methods=["POST"])
def make_prediciton():
    content = request.get_json()

    feature_arr = {
        "experience_level": [content.get("experience_level")],
        "employment_type": [content.get("employment_type")],
        "job_title": [content.get("job_title")],
        "remote_ratio": [content.get("remote_ratio")],
        "company_size": [content.get("company_size")],
    }

    feature_arr = pd.DataFrame.from_dict(feature_arr)

    model, ct = setup()

    if model is not None:
        feature_arr = ct.transform(feature_arr)
        prediction = int(model.predict(feature_arr)[0])

        logging.info(f"Prediction successful = {prediction}")

        return {"success": True, "data": {"predicted_salary": prediction}}

    return {"success": False, "data": {}}
