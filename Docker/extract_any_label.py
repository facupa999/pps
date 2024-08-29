import nibabel as nib
import numpy as np
import sys
import os
import argparse

# Estructuras válidas
VALID_STRUCTURES = ["amygdala", "putamen", "pallidum", "hippocampus", "thalamus"]

def extract_structure(img_path, dest_path, structure):
    if structure not in VALID_STRUCTURES:
        print(f"Estructura {structure} no es válida. Las estructuras válidas son: {', '.join(VALID_STRUCTURES)}.")
        return

    # Cargar imagen
    img = nib.load(img_path)
    data = img.get_fdata()

    filename = os.path.basename(img_path)

    if structure == "amygdala":
        left_amygdala = np.where(data == 18, data, 0)
        left_img = nib.Nifti1Image(left_amygdala, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_amygdala.nii"))
        nib.save(left_img, left_output_path)

        right_amygdala = np.where(data == 54, data, 0)
        right_img = nib.Nifti1Image(right_amygdala, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_amygdala.nii"))
        nib.save(right_img, right_output_path)

    elif structure == "putamen":
        left_putamen = np.where(data == 12, data, 0)
        left_img = nib.Nifti1Image(left_putamen, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_putamen.nii"))
        nib.save(left_img, left_output_path)

        right_putamen = np.where(data == 51, data, 0)
        right_img = nib.Nifti1Image(right_putamen, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_putamen.nii"))
        nib.save(right_img, right_output_path)

    elif structure == "pallidum":
        left_pallidum = np.where(data == 13, data, 0)
        left_img = nib.Nifti1Image(left_pallidum, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_pallidum.nii"))
        nib.save(left_img, left_output_path)

        right_pallidum = np.where(data == 52, data, 0)
        right_img = nib.Nifti1Image(right_pallidum, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_pallidum.nii"))
        nib.save(right_img, right_output_path)

    elif structure == "hippocampus":
        left_hippocampus = np.where(data == 17, data, 0)
        left_img = nib.Nifti1Image(left_hippocampus, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_hippocampus.nii"))
        nib.save(left_img, left_output_path)

        right_hippocampus = np.where(data == 53, data, 0)
        right_img = nib.Nifti1Image(right_hippocampus, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_hippocampus.nii"))
        nib.save(right_img, right_output_path)

    elif structure == "thalamus":
        left_thalamus = np.where(data == 10, data, 0)
        left_img = nib.Nifti1Image(left_thalamus, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_thalamus.nii"))
        nib.save(left_img, left_output_path)

        right_thalamus = np.where(data == 49, data, 0)
        right_img = nib.Nifti1Image(right_thalamus, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_thalamus.nii"))
        nib.save(right_img, right_output_path)

    print(f"Estructuras extraídas y guardadas en {dest_path}")

parser = argparse.ArgumentParser(description="Extraer estructuras de una imagen .mgz")
parser.add_argument("PATH1", help="Ruta de la imagen de entrada (salida de FastSurfer) .mgz")
parser.add_argument("PATH2", help="Ruta del directorio de salida")
parser.add_argument("structure", help="Estructura a extraer (amygdala, putamen, pallidum, hippocampus, thalamus)")

args = parser.parse_args()

img_path = args.PATH1
dest_path = args.PATH2
estructura = args.structure.lower()

extract_structure(img_path, dest_path, estructura)
