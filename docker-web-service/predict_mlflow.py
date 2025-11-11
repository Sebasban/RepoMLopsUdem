import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Las columnas esperadas por el modelo (según la firma)
COLUMNS = [
    "age","education_num","capital_gain","capital_loss","hours_per_week",
    "capital_net","has_capital_gain","has_capital_loss",
    "workclass=__other__","workclass=local-gov","workclass=missing","workclass=private",
    "workclass=self-emp-not-inc","workclass=state-gov",
    "marital_status=__other__","marital_status=divorced","marital_status=married-civ-spouse",
    "marital_status=never-married","marital_status=separated","marital_status=widowed",
    "occupation=__other__","occupation=adm-clerical","occupation=craft-repair",
    "occupation=exec-managerial","occupation=prof-specialty","occupation=sales",
    "relationship=__other__","relationship=husband","relationship=not-in-family",
    "relationship=own-child","relationship=unmarried","relationship=wife",
    "native_country=__other__","native_country=germany","native_country=mexico",
    "native_country=missing","native_country=philippines","native_country=united-states",
]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)
    df = df.reindex(columns=COLUMNS, fill_value=0)

    preds = model.predict(df)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df).tolist()

    response = {"predictions": preds.tolist()}
    if proba is not None:
        response["probabilities"] = proba

    return jsonify(response)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)
