# Imagen de FastSurfer como base
FROM deepmi/fastsurfer:cpu-latest

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Instala las dependencias necesarias para el script de Python
RUN apt-get update && apt-get install -y python3-pip
RUN pip install nibabel numpy 
RUN pip install streamlit==1.29.0
RUN pip install scikit-image
RUN pip install torch


# Copia los scripts de Python, el archivo de licencia y otros archivos necesarios al contenedor
COPY Una_imagen.py /app/Una_imagen.py
COPY pages/Muchas_imagenes.py /app/pages/Muchas_imagenes.py
COPY fs_license/license.txt /app/fs_license/license.txt
COPY fast_surfer.sh /app/fast_surfer.sh
COPY run_fastsurfer.sh /app/run_fastsurfer.sh
COPY thalamus.pt /app/thalamus.pt
COPY Models /app/Models
ENV FASTSURFER_HOME=/fastsurfer





# Establecer permisos y variable de entorno para la licencia
RUN chmod 644 /app/fs_license/license.txt
ENV FS_LICENSE=/app/fs_license/license.txt

# puerto en el que correrá Streamlit (8501 por defecto)
EXPOSE 8501

# Health check para el servicio
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Comando para iniciar Streamlit
ENTRYPOINT ["streamlit", "run", "Una_imagen.py", "--server.port=8501", "--server.address=0.0.0.0"]
