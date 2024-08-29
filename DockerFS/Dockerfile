# Imagen de FastSurfer como base
FROM deepmi/fastsurfer:cpu-latest

WORKDIR /app

# Instala las dependencias necesarias para el script de Python
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install nibabel numpy
RUN pip install streamlit==1.29.0

# Copia el script de Python y el archivo de licencia al contenedor
COPY extract_any_label.py /app/extract_any_label.py
COPY fs_license/license.txt /fs_license/license.txt

# Define el punto de entrada y el comando por defecto
COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh
ENTRYPOINT ["/app/run_pipeline.sh"]
