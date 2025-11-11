import requests
import json

# Entrada dummy para probar la API del modelo
payload_0 = {
  "age": 35,
  "education_num": 12,
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 40,
  "capital_net": 0,
  "has_capital_gain": 0,
  "has_capital_loss": 0,

  "workclass=__other__": 0,
  "workclass=local-gov": 0,
  "workclass=missing": 0,
  "workclass=private": 1,
  "workclass=self-emp-not-inc": 0,
  "workclass=state-gov": 0,

  "marital_status=__other__": 0,
  "marital_status=divorced": 0,
  "marital_status=married-civ-spouse": 0,
  "marital_status=never-married": 1,
  "marital_status=separated": 0,
  "marital_status=widowed": 0,

  "occupation=__other__": 0,
  "occupation=adm-clerical": 1,
  "occupation=craft-repair": 0,
  "occupation=exec-managerial": 0,
  "occupation=prof-specialty": 0,
  "occupation=sales": 0,

  "relationship=__other__": 0,
  "relationship=husband": 0,
  "relationship=not-in-family": 1,
  "relationship=own-child": 0,
  "relationship=unmarried": 0,
  "relationship=wife": 0,

  "native_country=__other__": 0,
  "native_country=germany": 0,
  "native_country=mexico": 0,
  "native_country=missing": 0,
  "native_country=philippines": 0,
  "native_country=united-states": 1
}
payload_1 = {
  "age": 45,
  "education_num": 16,
  "capital_gain": 50000,
  "capital_loss": 0,
  "hours_per_week": 50,

  "capital_net": 50000,
  "has_capital_gain": 1,
  "has_capital_loss": 0,

  "workclass=__other__": 0,
  "workclass=local-gov": 0,
  "workclass=missing": 0,
  "workclass=private": 0,
  "workclass=self-emp-not-inc": 0,
  "workclass=state-gov": 1,

  "marital_status=__other__": 0,
  "marital_status=divorced": 0,
  "marital_status=married-civ-spouse": 1,
  "marital_status=never-married": 0,
  "marital_status=separated": 0,
  "marital_status=widowed": 0,

  "occupation=__other__": 0,
  "occupation=adm-clerical": 0,
  "occupation=craft-repair": 0,
  "occupation=exec-managerial": 1,
  "occupation=prof-specialty": 0,
  "occupation=sales": 0,

  "relationship=__other__": 0,
  "relationship=husband": 1,
  "relationship=not-in-family": 0,
  "relationship=own-child": 0,
  "relationship=unmarried": 0,
  "relationship=wife": 0,

  "native_country=__other__": 0,
  "native_country=germany": 0,
  "native_country=mexico": 0,
  "native_country=missing": 0,
  "native_country=philippines": 0,
  "native_country=united-states": 1
}

url = "http://localhost:9696/predict"

response = requests.post(url, json=payload_1)

print("\n=== Payload enviado ===")
print(json.dumps(payload_1, indent=4))

print("\n=== Respuesta de la API ===")
print(response.json())