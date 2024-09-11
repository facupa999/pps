#!/bin/bash

# Variables de entorno
export FS_LICENSE=${FS_LICENSE:-"/fastsurfer/fs_license/license.txt"}

# Obtener el número de hilos disponibles con nproc
NUM_THREADS=$(nproc)

# Ejecutar FastSurfer con los parámetros proporcionados y el número de hilos
./run_fastsurfer.sh --t1 $1 --sd $2 --fs_license $FS_LICENSE --sid $3 --seg_only --no_cereb --no_hypothal --allow_root --threads $NUM_THREADS
