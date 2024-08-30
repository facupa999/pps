Modo de uso:
 
1. Construir la imagen Docker:

docker build -t labels .

2. puedes usar el siguiente comando para ejecutar el script y extraer las regiones de interés:

docker run --rm -v /ruta/a/la/carpeta/con/imagenes:/data 
	   -v ruta/a/la/carpeta/de/salida:/output 
	   fslabel /data/nombre_imagen.nii /output label


donde: 

-"/ruta/a/la/carpeta/con/imagenes" es la carpeta en tu máquina local donde se encuentra la imagen de entrada

-"nombre_imagen.mgz" es el nombre de la imagen de entrada

-"/ruta/a/la/carpeta/de/salida" es la carpeta en tu máquina local donde quieres guardar las imágenes resultantes

-"label" es La etiqueta que deseas extraer (por ejemplo, amygdala), pueden ser tambien "putamen", "pallidum", "hippocampus" o "thalamus"

ejemplo completo:


docker run --rm -v D:\Documents\FastSurfer\Imagenes\ixi\GUY:/data -v D:\Documents\FastSurfer\duilio\pruebas:/output fslabel /data/r_IXI023-Guys-0699-IXI3DMPRAG_-s231_-0401-00004-000001-01.nii /output amygdala

en este ejemplo de la imagen "r_IXI002-Guys-0828-MPRAGESEN_-s256_-0301-00003-000001-01.nii" se va a segmentar para proximamente extraer los labels "left_amygdala" y "rigth_amygdala" y van a quedar en "/Documents/FastSurfer/output" 
como 2 archivos .nii 

 
