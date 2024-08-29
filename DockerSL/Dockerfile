# Imagen de FastSurfer como base
FROM deepmi/fastsurfer:cpu-latest

# Instala las dependencias necesarias para el script de Python
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install nibabel numpy

# Copia el script de Python y el archivo de licencia al contenedor
COPY extract_any_label.py /app/extract_any_label.py
COPY fs_license/license.txt /fs_license/license.txt

# puerto en el que correrá Streamlit (8501 por defecto)
EXPOSE 8501

# Define el punto de entrada y el comando por defecto
COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh
ENTRYPOINT ["/app/run_pipeline.sh"]
