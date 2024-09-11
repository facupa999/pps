# Imagen de FastSurfer como base
FROM deepmi/fastsurfer:cpu-latest

WORKDIR /fastsurfer

# Copia el archivo de licencia al contenedor
COPY fs_license/license.txt /fastsurfer/fs_license/license.txt

# Define el punto de entrada y el comando por defecto
COPY run_pipeline.sh /app/run_pipeline.sh
ENTRYPOINT ["/app/run_pipeline.sh"]
