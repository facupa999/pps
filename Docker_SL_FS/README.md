Modo de uso:
 
1. Construir la imagen Docker:

docker build -t stlabels .

2. correrlo en el puerto 8501:

docker run -p 8501:8501 stlabel
 
3. desde windows al navegador ir a "http://localhost:8501"

	en linux a "http://0.0.0.0:8501"