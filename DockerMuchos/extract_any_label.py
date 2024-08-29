import nibabel as nib
import numpy as np
import sys
import os
import argparse
import os
import skimage.transform as skTrans


def margin(x,tolerance):
    y = (x*tolerance)
    left = right = int(y)
    return left,right

def crop_hipp_image_around_boundig_box (raw_image,tolerance):
    image = raw_image.get_fdata()
    mask = image == 0
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_left = np.max(coords, axis=1)

    span_x = bottom_left[0]-top_left[0]
    span_y = bottom_left[1]-top_left[1]
    span_z = bottom_left[2]-top_left[2]
    var_center_x = top_left[0] + int(span_x/2)
    var_center_y = top_left[1] + int(span_y/2)
    var_center_z = top_left[2] + int(span_z/2)
  
    crop_image = image[var_center_x - 32:var_center_x + 32,
                      var_center_y - 32:var_center_y + 32,
                      var_center_z - 32:var_center_z + 32]

    return nib.Nifti1Image(crop_image.astype(float), raw_image.affine) 

def resize_image_to_match_hippocampal_boundig_box (new_image, image_size):
    im = new_image.get_fdata()
    result1 = skTrans.resize(im, image_size, order=1, preserve_range=True)
    return nib.Nifti1Image(result1.astype(float), new_image.affine) 
  
def preprocess_image_file_for_anomaly_detection(hippocampal_mri: np.array, image_size: int = (64,64,64), is_test: bool = False, tolerance : float = 0.25):
    """Preprocess an input image for anomaly detection.
    """
      # crop the image
    new_image = crop_hipp_image_around_boundig_box(hippocampal_mri, tolerance=tolerance)
      # and downsize it
    new_image = resize_image_to_match_hippocampal_boundig_box(new_image, image_size)

    return new_image

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
        left_img = preprocess_image_file_for_anomaly_detection(left_img)
        nib.save(left_img, left_output_path)

        right_amygdala = np.where(data == 54, data, 0)
        right_img = nib.Nifti1Image(right_amygdala, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_amygdala.nii"))
        right_img = preprocess_image_file_for_anomaly_detection(right_img)
        nib.save(right_img, right_output_path)

    elif structure == "putamen":
        left_putamen = np.where(data == 12, data, 0)
        left_img = nib.Nifti1Image(left_putamen, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_putamen.nii"))
        left_img = preprocess_image_file_for_anomaly_detection(left_img)
        nib.save(left_img, left_output_path)

        right_putamen = np.where(data == 51, data, 0)
        right_img = nib.Nifti1Image(right_putamen, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_putamen.nii"))
        right_img = preprocess_image_file_for_anomaly_detection(right_img)
        nib.save(right_img, right_output_path)

    elif structure == "pallidum":
        left_pallidum = np.where(data == 13, data, 0)
        left_img = nib.Nifti1Image(left_pallidum, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_pallidum.nii"))
        left_img = preprocess_image_file_for_anomaly_detection(left_img)
        nib.save(left_img, left_output_path)

        right_pallidum = np.where(data == 52, data, 0)
        right_img = nib.Nifti1Image(right_pallidum, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_pallidum.nii"))
        right_img = preprocess_image_file_for_anomaly_detection(right_img)
        nib.save(right_img, right_output_path)

    elif structure == "hippocampus":
        left_hippocampus = np.where(data == 17, data, 0)
        left_img = nib.Nifti1Image(left_hippocampus, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_hippocampus.nii"))
        left_img = preprocess_image_file_for_anomaly_detection(left_img)
        nib.save(left_img, left_output_path)

        right_hippocampus = np.where(data == 53, data, 0)
        right_img = nib.Nifti1Image(right_hippocampus, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_hippocampus.nii"))
        right_img = preprocess_image_file_for_anomaly_detection(right_img)
        nib.save(right_img, right_output_path)

    elif structure == "thalamus":
        left_thalamus = np.where(data == 10, data, 0)
        left_img = nib.Nifti1Image(left_thalamus, img.affine, img.header)
        left_output_path = os.path.join(dest_path, filename.replace(".mgz", "_left_thalamus.nii"))
        left_img = preprocess_image_file_for_anomaly_detection(left_img)
        nib.save(left_img, left_output_path)

        right_thalamus = np.where(data == 49, data, 0)
        right_img = nib.Nifti1Image(right_thalamus, img.affine, img.header)
        right_output_path = os.path.join(dest_path, filename.replace(".mgz", "_right_thalamus.nii"))
        right_img = preprocess_image_file_for_anomaly_detection(right_img)
        nib.save(right_img, right_output_path)

    

def process_directory(input_dir, dest_dir, structure):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mgz"):
                img_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(dest_dir, relative_path)
                os.makedirs(output_path, exist_ok=True)
                extract_structure(img_path, output_path, structure)
                print(f"Estructuras extraídas y guardadas en {output_path}")

parser = argparse.ArgumentParser(description="Extraer estructuras de imágenes .mgz en una carpeta")
parser.add_argument("input_dir", help="Ruta del directorio de entrada con imágenes .mgz")
parser.add_argument("dest_dir", help="Ruta del directorio de salida")
parser.add_argument("structure", help="Estructura a extraer (amygdala, putamen, pallidum, hippocampus, thalamus)")

args = parser.parse_args()

input_dir = args.input_dir
dest_dir = args.dest_dir
estructura = args.structure.lower()

process_directory(input_dir, dest_dir, estructura)
