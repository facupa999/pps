#!/bin/bash

# Variables de entrada
T1_IMAGE=$1
OUTPUT_DIR=$2
STRUCTURE=$3

# Ejecuta FastSurfer dentro del mismo contenedor (con opción para permitir root)
./run_fastsurfer.sh --t1 $T1_IMAGE --sd $OUTPUT_DIR --fs_license /fs_license/license.txt --sid subjectX --seg_only --allow_root

# Ruta del archivo segmentado
SEGMENTED_IMAGE="$OUTPUT_DIR/subjectX/mri/aparc.DKTatlas+aseg.deep.mgz"

# Verificar si el archivo segmentado existe antes de continuar
if [ ! -f "$SEGMENTED_IMAGE" ]; then
    echo "Error: El archivo segmentado $SEGMENTED_IMAGE no se encontró."
    exit 1
fi

# Ejecuta el script de Python para extraer estructuras
python3 /app/extract_any_label.py $SEGMENTED_IMAGE $OUTPUT_DIR $STRUCTURE

#aca se podria llamar a otro .py que borre la carpeta "subjectX"
