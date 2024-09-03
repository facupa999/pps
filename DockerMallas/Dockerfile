# Usa la imagen base de Python 3.9
FROM python:3.9-slim

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Instala las dependencias necesarias para el script de Python
RUN apt-get update 
RUN apt-get install -y python3-pip
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y libglib2.0-0 
RUN apt-get install -y libxrender1 
RUN apt-get install -y libxext6 
RUN apt-get install -y libsm6 
RUN apt-get install -y xvfb

# Instala las bibliotecas de Python por separado
RUN pip install nibabel
RUN pip install numpy
RUN pip install scikit-image
RUN pip install trimesh
RUN pip install streamlit==1.29.0
RUN pip install stpyvista
RUN pip install pyvista
RUN pip install scipy
RUN pip install matplotlib

# Copia los scripts de Python al contenedor
COPY Hello.py /app/Hello.py
COPY pages/malla.py /app/pages/malla.py

# Puerto en el que correrá Streamlit (8501 por defecto)
EXPOSE 8501

# Health check para el servicio
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Inicia un servidor X virtual y luego ejecuta Streamlit
#CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && streamlit run Hello.py --server.port=8501 --server.address=0.0.0.0"]

# Comando para iniciar Streamlit
ENTRYPOINT ["streamlit", "run", "Hello.py", "--server.port=8501", "--server.address=0.0.0.0"]

