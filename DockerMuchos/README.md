Modo de uso:
 
1. Construir la imagen Docker:

docker build -t labelsmuchos .

2. puedes usar el siguiente comando para ejecutar el script y extraer las regiones de interés:

docker run --rm -v /ruta/a/la/carpeta/con/imagenes:/app/imagen 
           -v /ruta/a/la/carpeta/de/salida:/app/salida 
           labels python extract_any_label.py /app/imagen/nombre_imagen.mgz /app/salida label

donde: 

-"/ruta/a/la/carpeta/con/imagenes" es la carpeta en tu máquina local donde se encuentran las imagenes de entrada

-"/ruta/a/la/carpeta/de/salida" es la carpeta en tu máquina local donde quieres guardar las imágenes resultantes

-"label" es La etiqueta que deseas extraer (por ejemplo, amygdala), pueden ser tambien "putamen", "pallidum", "hippocampus" o "thalamus"

ejemplo completo:

docker run --rm -v D:/Documents/FastSurfer/TODOS:/app/imagenes -v D:\Documents\FastSurfer\pruebas:/app/salida labelstodos python extract_any_label.py /app/imagenes /app/salida amygdala

en este ejemplo de la carpeta con el path "D:/Documents/FastSurfer/TODOS" se van a extraer de cada imagen (.mgz) los labels "left_amygdala" y "rigth_amygdala" y van a quedar en "D:\Documents\FastSurfer\pruebas" 
como 2 archivos .nii por cada imagen.

 
