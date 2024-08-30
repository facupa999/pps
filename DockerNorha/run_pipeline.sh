#!/bin/bash

# Variables de entrada
IZQ_IMAGE_PATH=$1
DER_IMAGE_PATH=$2
STRUCTURE=$3

# Ejecuta el script de Python ejecutar norha
python /app/test.py "$IZQ_IMAGE_PATH" "$DER_IMAGE_PATH" "$STRUCTURE"

