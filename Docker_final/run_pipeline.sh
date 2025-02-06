#!/bin/bash

# Variables de entorno
export FS_LICENSE=${FS_LICENSE:-"/fastsurfer/fs_license/license.txt"}

# Obtener el número de hilos disponibles con nproc
NUM_THREADS=$(nproc)

# Ejecutar FastSurfer en segundo plano
./run_fastsurfer.sh --t1 $1 --sd $2 --fs_license $FS_LICENSE --sid $3 --seg_only --no_cereb --no_hypothal --allow_root --threads $NUM_THREADS &

# Guardar el ID del proceso de FastSurfer
FASTSURFER_PID=$!

# Ruta al archivo que deseas monitorear (reemplazar con la ruta correcta)
FILE_PATH="$2/$3/mri/aparc.DKTatlas+aseg.deep.mgz"

START_TIME=$(date +%s)



# Monitorear la creación del archivo
while [ ! -f "$FILE_PATH" ]; do
    # Verificar si el proceso de FastSurfer sigue corriendo
    if ! kill -0 $FASTSURFER_PID 2>/dev/null; then
        echo "El proceso de FastSurfer ha terminado, pero el archivo no fue generado."
        echo "Revisar errores en fastsurfer_errors.log:"
        cat fastsurfer_errors.log
        exit 1
    fi
    echo "Todavía no termina."
    sleep 10
done



END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo "Tiempo total transcurrido: $(date -u -d @${TOTAL_TIME} +"%H:%M:%S")"

# Matar el proceso de FastSurfer una vez que se haya generado el archivo
kill $FASTSURFER_PID 2>/dev/null

echo "El archivo $FILE_PATH ha sido generado. El proceso de FastSurfer ha sido detenido."

# Mover el archivo generado al directorio de salida y renombrarlo
NEW_FILE_PATH="$2/$3.mgz"
mv "$FILE_PATH" "$NEW_FILE_PATH"
echo "Archivo movido y renombrado a $NEW_FILE_PATH."

# Borrar la carpeta generada ($3)
rm -rf "$2/$3"
echo "La carpeta $2/$3 ha sido eliminada."

# Borrar los errores temporales ya que no son necesarios
rm fastsurfer_errors.log
