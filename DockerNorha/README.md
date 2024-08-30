Modo de uso:
 
1. Construir la imagen Docker:

docker build -t norha .

2. el siguiente comando imprime el indice norha:

docker run --rm -v /ruta/a/la/carpeta/con/imagen/izquierda:/left -v /ruta/a/la/carpeta/con/imagen/derecha:/right norha /left/nombre_imagen_izq.nii /right/nombre_imagen_der.nii label

donde: 

-"/ruta/a/la/carpeta/con/imagen/izquierda" es la carpeta en tu máquina local donde se encuentra la imagen del lado izquierdo

-"/ruta/a/la/carpeta/con/imagen/derecha" es la carpeta en tu máquina local donde se encuentra la imagen del lado derecho

-"/ruta/a/la/carpeta/de/salida" es la carpeta en tu máquina local donde quieres guardar las imágenes resultantes

-"nombre_imagen_izq.nii" nombre del archivo de la imagen izquierda

-"nombre_imagen_der.nii" nombre del archivo de la imagen derecha

-"label" es La estructura que deseas analizar (por ejemplo, amygdala), pueden ser tambien "putamen", "pallidum", "hippocampus" o "thalamus"

ejemplo completo:

docker run --rm -v D:\Documents\FastSurfer\DUILIO\thalamus:/left -v D:\Documents\FastSurfer\DUILIO\thalamus:/right norha /left/r_ADNI_001_left_thalamus.nii /right/r_ADNI_001_right_thalamus.nii thalamus
 

 
