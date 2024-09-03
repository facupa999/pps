#!/bin/bash

# Variables de entorno
export FS_LICENSE=${FS_LICENSE:-"/fastsurfer/fs_license/license.txt"}

# Ejecutar FastSurfer con los parámetros proporcionados
./run_fastsurfer.sh --t1 $1 --sd $2 --fs_license $FS_LICENSE --sid $3 --seg_only --allow_root

#aca se podria llamar a otro .py que borre la carpeta "subjectX"