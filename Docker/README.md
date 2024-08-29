Modo de uso:
 
1. Construir la imagen Docker:

docker build -t labels .

2. puedes usar el siguiente comando para ejecutar el script y extraer las regiones de interés:

docker run --rm -v /ruta/a/la/carpeta/con/imagenes:/app/imagen 
           -v /ruta/a/la/carpeta/de/salida:/app/salida 
           labels python extract_any_label.py /app/imagen/nombre_imagen.mgz /app/salida label

donde: 

-"/ruta/a/la/carpeta/con/imagenes" es la carpeta en tu máquina local donde se encuentra la imagen de entrada

-"nombre_imagen.mgz" es el nombre de la imagen de entrada

-"/ruta/a/la/carpeta/de/salida" es la carpeta en tu máquina local donde quieres guardar las imágenes resultantes

-"label" es La etiqueta que deseas extraer (por ejemplo, amygdala), pueden ser tambien "putamen", "pallidum", "hippocampus" o "thalamus"

ejemplo completo:


docker run --rm -v D:/Documents/FastSurfer/imagenes:/app/imagen
           -v D:/Documents/FastSurfer/output:/app/salida
           labels python extract_any_label.py /app/imagen/r_HEC_HSL_018.mgz /app/salida amygdala

en este ejemplo de la imagen "r_HEC_HSL_018.mgz" se van a extraer los labels "left_amygdala" y "rigth_amygdala" y van a quedar en "/Documents/FastSurfer/output" 
como 2 archivos .nii 

 
