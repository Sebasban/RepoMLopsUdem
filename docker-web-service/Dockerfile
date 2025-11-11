FROM python:3.12-slim

# Actualizar pip e instalar UV
RUN pip install -U pip
RUN pip install uv

WORKDIR /app

# Copiar solo el archivo de dependencias
COPY [ "pyproject.toml", "./" ]

# Instalar dependencias con UV
RUN uv pip install --system -e .

# Copiar solo los archivos necesarios para la aplicación
COPY [ "predict_mlflow.py", "model.pkl", "./" ]

# Exponer puerto
EXPOSE 9696

# Ejecutar la aplicación con gunicorn
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict_mlflow:app" ]