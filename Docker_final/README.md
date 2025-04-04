# Proyecto de Segmentación y Procesamiento con Streamlit

## Descripción
Este proyecto utiliza Docker Compose para desplegar tres servicios:
- **Streamlit App**: Interfaz web interactiva.
- **Segmentación (FastSurfer)**: Servicio para segmentación de imágenes cerebrales.
- **Preprocesamiento (NPPY)**: Servicio para preprocesar los datos.

Los contenedores comparten un volumen (`shared`) para intercambiar datos.

## Requisitos
- [Docker](https://docs.docker.com/get-docker/)  
- [Docker Compose] (incluido en Docker Desktop o instalable por separado en algunas versiones de Linux) (https://docs.docker.com/compose/install/)  

## Instalación y Uso
1. Clonar el repositorio:  

   git clone https://github.com/facupa999/pps.git
   cd mi-proyecto


## Para construir y ejecutar los servicios:
2. docker-compose up --build

3. desde windows al navegador ir a "http://localhost:8501"

	en linux a "http://0.0.0.0:8501"

