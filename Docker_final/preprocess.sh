#!/bin/bash

# Verificar que se proporcionó el argumento necesario
if [ -z "$1" ]; then
    echo "Uso: $0 <input_image>"
    exit 1
fi

input_image="$1"
filename=$(basename "$input_image")  # Extraer solo el nombre del archivo
working_dir="/tmp/nppy_work"
output_folder="$working_dir/output"

# Crear un directorio de trabajo temporal dentro del contenedor
mkdir -p "$output_folder"

# Copiar la imagen de entrada al directorio de trabajo
cp "$input_image" "$working_dir/$filename"

# Ejecutar nppy dentro del directorio de trabajo
nppy -i "$working_dir/$filename" -o "$output_folder" -s -w -1 || {
    echo "❌ Error en el preprocesamiento"
    exit 1
}

# Buscar el archivo generado en la carpeta de salida
processed_image=$(find "$output_folder" -type f -name "*.nii" | head -n 1)

if [ -z "$processed_image" ]; then
    echo "❌ No se generó ningún archivo de salida"
    exit 1
fi

# Mover el archivo procesado al lugar original y sobrescribir
mv "$processed_image" "$input_image" || {
    echo "❌ Error al sobrescribir el archivo original"
    exit 1
}

# Limpiar archivos temporales
rm -rf "$working_dir"

echo "✅ Procesamiento completado y sobrescrito en: $input_image"
