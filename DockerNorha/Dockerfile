FROM continuumio/miniconda3:latest
#FROM python:3.7-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos y el script al contenedor
RUN pip install numpy
RUN pip install torch
RUN pip install nibabel
RUN pip install matplotlib


COPY test.py test.py
COPY thalamus.pt /app/thalamus.pt
COPY Models /app/Models


# Define el punto de entrada y el comando por defecto
COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh
ENTRYPOINT ["/app/run_pipeline.sh"]