# Imagen de FastSurfer como base
FROM deepmi/fastsurfer:cpu-latest

# Establecer el directorio de trabajo en /fastsurfer
WORKDIR /fastsurfer

# Cambiar al usuario root para instalar dependencias
USER root

# Instala las dependencias necesarias para el script de Python
RUN apt-get update && apt-get install -y python3-pip
RUN pip install nibabel numpy 
RUN pip install streamlit==1.29.0
RUN pip install scikit-image
RUN pip install torch
RUN pip install trimesh
RUN pip install stpyvista
RUN pip install pyvista
RUN pip install scipy
RUN pip install matplotlib

# Instala dependencias de sistema
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y libglib2.0-0 
RUN apt-get install -y libxrender1 
RUN apt-get install -y libxext6 
RUN apt-get install -y libsm6 
RUN apt-get install -y xvfb

# Copia los scripts de Python, el archivo de licencia y otros archivos necesarios al contenedor
COPY Una_imagen.py /fastsurfer/Una_imagen.py
COPY pages/Muchas_imagenes.py /fastsurfer/pages/Muchas_imagenes.py
COPY fs_license/license.txt /fastsurfer/fs_license/license.txt
COPY fast_surfer.sh /fastsurfer/fast_surfer.sh
COPY Models /fastsurfer/Models

# Establecer permisos y variable de entorno para la licencia
RUN chmod 644 /fastsurfer/fs_license/license.txt
ENV FS_LICENSE=/fastsurfer/fs_license/license.txt
ENV FASTSURFER_HOME=/fastsurfer

# puerto en el que correrá Streamlit (8501 por defecto)
EXPOSE 8501

# Health check para el servicio
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Inicia un servidor X virtual y luego ejecuta Streamlit
#CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && streamlit run Una_imagen.py --server.port=8501 --server.address=0.0.0.0"]

# Comando para iniciar Streamlit
ENTRYPOINT ["streamlit", "run", "Una_imagen.py", "--server.port=8501", "--server.address=0.0.0.0"]
