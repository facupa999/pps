FROM continuumio/miniconda3:latest
#FROM python:3.7-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos y el script al contenedor
COPY requirements.txt requirements.txt
COPY extract_any_label.py extract_any_label.py

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Definir el comando por defecto para ejecutar el script
#CMD ["python", "extract_any_label.py"]

