import streamlit as st
import nibabel as nib
import numpy as np
import os
import subprocess
import pandas as pd
import torch
import shutil
import skimage.transform as skTrans

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap
from Models.LeNet_ELU_SVDD import Encoder



# Estructuras válidas
VALID_STRUCTURES = ["amygdala", "putamen", "pallidum", "hippocampus", "thalamus"]

st.markdown("# Una sola imagen")
st.sidebar.markdown("# Una sola imagen:")
st.sidebar.subheader("Aqui podras trabajar de a una imagen")


# Directorio donde se guardarán los archivos CSV
CSV_DIRECTORY = "/fastsurfer/csv_results"

# Crear el directorio si no existe
os.makedirs(CSV_DIRECTORY, exist_ok=True)

def extract_structure(img_path, structure):
    if structure not in VALID_STRUCTURES:
        st.error(f"Estructura {structure} no es válida. Las estructuras válidas son: {', '.join(VALID_STRUCTURES)}.")
        return

    # Cargar imagen
    img = nib.load(img_path)
    data = img.get_fdata()

    filename = os.path.basename(img_path)
    output_files = []

    # Determinar los valores correspondientes a cada estructura
    if structure == "amygdala":
        left_value, right_value = 18, 54
    elif structure == "putamen":
        left_value, right_value = 12, 51
    elif structure == "pallidum":
        left_value, right_value = 13, 52
    elif structure == "hippocampus":
        left_value, right_value = 17, 53
    elif structure == "thalamus":
        left_value, right_value = 10, 49

    # Extraer las estructuras
    left_structure = np.where(data == left_value, data, 0)
    left_img = nib.Nifti1Image(left_structure, img.affine, img.header)
    left_output_path = os.path.join("/fastsurfer", filename.replace(".mgz", f"_left_{structure}.nii"))
    left_img = preprocess_image_file_for_anomaly_detection(left_img)
    nib.save(left_img, left_output_path)
    output_files.append(left_output_path)



    right_structure = np.where(data == right_value, data, 0)
    right_img = nib.Nifti1Image(right_structure, img.affine, img.header)
    right_output_path = os.path.join("/fastsurfer", filename.replace(".mgz", f"_right_{structure}.nii"))
    right_img = preprocess_image_file_for_anomaly_detection(right_img)
    nib.save(right_img, right_output_path)
    output_files.append(right_output_path)

    st.success(f"Estructuras {structure} extraídas")

    left_volume = (left_img.get_fdata() > 0).sum()
    right_volume = (right_img.get_fdata() > 0).sum()
    volume_difference = abs(left_volume - right_volume)

    # Nombre del archivo CSV para la estructura específica
    csv_file_path = os.path.join(CSV_DIRECTORY, f"{structure}_results.csv")

    # Agregar fila al archivo CSV específico
    new_row = {
        "Nombre de Archivo": filename,
        "Volumen Izquierdo": left_volume,
        "Volumen Derecho": right_volume,
        "Diferencia de Volumen": volume_difference
    }
    
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame([new_row])
    else:
        df = pd.read_csv(csv_file_path)
        df = df.append(new_row, ignore_index=True)

    df.to_csv(csv_file_path, index=False)

    st.write(f"La diferencia entre volúmenes es de: {volume_difference}")
    asimetria(left_output_path,right_output_path,structure)

    # Agregar archivos extraídos a la lista de archivos en session_state
    st.session_state.output_files.extend(output_files)


def segmentar(img_path, name_subject):
    output_segmentation_files = []
    # Mostrar spinner mientras se ejecuta el script
    with st.spinner('Segmentando la imagen, por favor espera...'):
        try:
            # Ruta completa al script dentro del contenedor
            command = ["bash", "/fastsurfer/fast_surfer.sh", img_path, "/fastsurfer", name_subject]
            subprocess.run(command, capture_output=True, text=True, check=True)
            st.success("Segmentación completada exitosamente.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error al ejecutar el script: {e.stderr}")

    segmentation_path = os.path.join("/fastsurfer", name_subject, "mri", "aparc.DKTatlas+aseg.deep.mgz")
    final_path = os.path.join("/fastsurfer", f"{name_subject}.mgz")
    shutil.move(segmentation_path, final_path)
    shutil.rmtree(os.path.join("/fastsurfer", name_subject))
    output_segmentation_files.append(final_path)
    st.session_state.output_segmentation_files.extend(output_segmentation_files)


def tab_1():
    st.title("Segmentación de imagen")

    # Inicializar la lista de archivos extraídos en session_state si no existe
    if "output_segmentation_files" not in st.session_state:
        st.session_state.output_segmentation_files = []

    # Subir archivo
    uploaded_file = st.file_uploader("Subir archivo .nii", type=["nii"])
    if uploaded_file is not None:
        img_path = os.path.join("/fastsurfer", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    name_subject = st.text_input("Ingrese el nombre del paciente")

    if st.button("Segmentar"):
        if uploaded_file and name_subject:
            segmentar(img_path, name_subject)
        else:
            st.error("Por favor, sube un archivo e ingresa el nombre del paciente.")

    # Mostrar archivos extraídos
    st.subheader("Archivos Segmentados")
    for file_path in st.session_state.output_segmentation_files:
        with open(file_path, "rb") as file:
            st.download_button(
                label=f"Descargar {os.path.basename(file_path)}",
                data=file,
                file_name=os.path.basename(file_path)
            )


def tab_2():
    st.title("Extracción de Estructuras Cerebrales")

    # Inicializar la lista de archivos extraídos en session_state si no existe
    if "output_files" not in st.session_state:
        st.session_state.output_files = []

    # Opciones de archivos generados en `tab_1` y archivo subido manualmente
    file_options = [None] + st.session_state.output_segmentation_files
    selected_file = st.selectbox("Seleccionar archivo de los que segmentaste", file_options)

    # Subir archivo
    uploaded_file2 = st.file_uploader("O subir archivo .mgz directamente", type=["mgz"])
    img_path2 = None

    if uploaded_file2 is not None:
        img_path2 = os.path.join("/fastsurfer", uploaded_file2.name)
        with open(img_path2, "wb") as f:
            f.write(uploaded_file2.getbuffer())
    elif selected_file is not None:
        img_path2 = selected_file

    structure = st.selectbox("Selecciona una estructura a extraer", VALID_STRUCTURES)

    if st.button("Extraer"):
        if img_path2:
            extract_structure(img_path2, structure)
        else:
            st.error("Por favor, selecciona un archivo segmentado o sube uno manualmente y selecciona una estructura.")

    # Mostrar archivos extraídos
    st.subheader("Archivos Extraídos")
    for file_path in st.session_state.output_files:
        with open(file_path, "rb") as file:
            st.download_button(
                label=f"Descargar {os.path.basename(file_path)}",
                data=file,
                file_name=os.path.basename(file_path)
            )


def tab_3():
    st.title("Resultados")

    # Iterar sobre cada estructura válida y mostrar el CSV si existe y no está vacío
    for structure in VALID_STRUCTURES:
        csv_file_path = os.path.join(CSV_DIRECTORY, f"{structure}_results.csv")
        
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            
            if not df.empty:
                st.subheader(f"Resultados para {structure.capitalize()}")
                st.write(df)
                st.download_button(
                    label=f"Descargar resultados de {structure.capitalize()} en CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{structure}_results.csv",
                    mime='text/csv'
                )


def asimetria(image,imagepair,structure):
    image = nib.load(image)
    image = np.array(image.get_fdata())

    imagepair = nib.load(imagepair)
    imagepair = np.array(imagepair.get_fdata())
    imagepair = np.flip(imagepair,axis=0).copy()


    image = torch.tensor(image)
    imagepair = torch.tensor(imagepair)

    left_volume, right_volume = np.count_nonzero(imagepair), np.count_nonzero(image)
    left_volume = torch.unsqueeze(torch.from_numpy(np.array(left_volume)), 0)
    right_volume = torch.unsqueeze(torch.from_numpy(np.array(right_volume)), 0)


    Example_input = [torch.unsqueeze(torch.cat((torch.unsqueeze(image, 0),torch.unsqueeze(imagepair, 0)), 0), 0),left_volume, right_volume]

    
    # Model class must be defined somewhere
    model = torch.load(f"/fastsurfer/Models/{structure}.pt")
    model.eval()

    inference = model(Example_input).cpu().detach().numpy()
    if structure == "amygdala":
        c=torch.from_numpy(np.array([ 1.1995e-03, -9.0719e-05,  2.2601e-04, -1.6296e-04,  1.2889e-04,
     	7.9593e-04, -4.6107e-04,  9.3010e-05,  2.6274e-04, -4.4553e-04,
     	2.0372e-04, -2.7011e-04, -7.7813e-04, -3.4863e-04,  4.8828e-04,
     	2.0638e-04, -7.3941e-04, -8.4966e-04, -1.0506e-03, -2.0882e-04,
    	-7.3310e-04,  8.1595e-05, -2.3642e-04,  1.3975e-04, -6.8821e-05,
     	3.7211e-04, -9.4548e-04,  3.9856e-04,  3.9226e-04, -2.1122e-05,
    	-4.1044e-04,  1.0118e-03]))
    elif structure == "putamen":
        c=torch.from_numpy(np.array([-6.0157e-04, -2.4639e-03, -1.1748e-04, -2.6048e-04,  1.0129e-03,
    	-1.5707e-03,  1.4672e-03,  4.5265e-04, -4.3584e-04,  1.3319e-04,
    	-1.9649e-03, -2.6121e-04,  7.0604e-04, -1.2275e-03, -4.4771e-04,
    	-5.7263e-04,  5.2476e-04, -1.4224e-03, -1.1021e-03,  3.9382e-04,
     	1.6061e-03, -1.3993e-03, -1.2486e-04, -1.4451e-03,  1.3612e-05,
     	2.1086e-05,  1.0144e-03, -9.1855e-04, -1.8682e-03,  4.3085e-04,
     	1.2417e-03, -9.9061e-04]))
    elif structure == "pallidum":
        c=torch.from_numpy(np.array([ 1.2615e-03, -9.8757e-04,  3.3690e-04,  8.8020e-04,  1.9845e-03,
     	2.8372e-04, -1.5581e-03, -2.5303e-04, -2.8048e-05, -5.8833e-04,
     	2.2088e-04, -1.0752e-03, -9.9092e-04, -8.4401e-04,  4.2088e-04,
    	-2.7957e-04, -2.0193e-04,  3.4747e-04,  8.4916e-05,  2.2653e-04,
    	-5.0316e-04, -1.5247e-03,  1.3997e-03,  1.6661e-03, -5.5025e-04,
    	-7.6762e-04,  1.0199e-03,  1.7472e-04,  2.8011e-04,  1.3499e-04,
     	1.5556e-03,  4.8420e-04]))
    elif structure == "hippocampus":
        c=torch.from_numpy(np.array([ 0.0118,  0.0097,  0.0113, -0.0100, -0.0113, -0.0136,  0.0107,  0.0092,
    	-0.0130,  0.0136, -0.0119, -0.0129, -0.0119, -0.0127, -0.0107,  0.0116,
    	-0.0111, -0.0151,  0.0108,  0.0113, -0.0118, -0.0112, -0.0104,  0.0136,
     	0.0112, -0.0115,  0.0129,  0.0114,  0.0115, -0.0120, -0.0074,  0.0113]))
    elif structure == "thalamus":
        c=torch.from_numpy(np.array([-7.3644e-04,  2.5405e-04, -3.1788e-03, -1.6109e-03, -7.6053e-04,
                -5.0173e-04,  2.2761e-03,  1.0202e-03, -2.8073e-03,  1.0238e-03,
                3.1727e-04, -1.2766e-03,  3.5283e-05, -7.2480e-04, -2.5971e-04,
                9.8939e-04, -1.7135e-03,  1.4254e-03,  1.3150e-05, -1.0366e-03,
                -9.3122e-04, -1.1257e-03, -1.0345e-03,  1.4835e-03,  1.4995e-03,
                1.2943e-03, -9.0330e-04,  1.8791e-03,  6.6228e-04, -2.2420e-04,
                3.5120e-03,  3.4326e-03]))
        

    NORAH_index = torch.sum((torch.from_numpy(inference) - c) ** 2, dim=1).numpy()
    st.write(f"El indice Norah es: {NORAH_index*200000}")

    # Configuración de la figura
    fig, ax = plt.subplots()

    # Definir colores con transparencia
    colors_outer =  ['red', 'yellow', 'green']
    colors_inner = ['yellow', 'green']
    alphas = [0.3, 0.3, 0.3]

    # Crear mapas de colores personalizados para los gradientes
    cmap_outer = LinearSegmentedColormap.from_list("custom_outer", colors_outer, N=100)
    cmap_inner = LinearSegmentedColormap.from_list("custom_inner", colors_inner, N=100)

    # Creación de los semicírculos con gradiente
    for i, radius in enumerate([20, 10]):
        for r in np.linspace(radius, radius-9, 10):
            if radius == 10:
                color = cmap_inner((radius - r) / 10)  # Green to yellow for inner circle
            else:
                color = cmap_outer((radius - r) / 20)  # Red to yellow for outer circles
            semicirculo = Wedge((0, 0), r, 0, 180, color=color, alpha=alphas[i], clip_on=False)
            ax.add_artist(semicirculo)

    # Coordenadas para el punto negro a 25 unidades en un ángulo de 30 grados
    angle_rad = np.radians(90)
    point_x = NORAH_index*200000 * np.cos(angle_rad)
    point_y = NORAH_index*200000 * np.sin(angle_rad)

    # Dibujar el punto negro
    ax.plot(point_x, point_y, 'd')  # 'ko' indica un punto negro

    # Configuración del gráfico
    ax.set_aspect('equal')
    ax.set_xlim(-35, 35)
    ax.set_ylim(0, 35)  # Cambiado para mostrar solo la parte superior
    ax.axis('off')

    # Mostrar el gráfico en streamlit
    st.pyplot(fig)

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



tab1, tab2, tab3 = st.tabs(["Segmentar Imagen", "Imagen ya Segmentada", "Resultados"])
with tab1:
    tab_1()

with tab2:
    tab_2()

with tab3:
    tab_3()









