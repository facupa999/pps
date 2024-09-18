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


#para malla
import trimesh
from skimage import measure
from trimesh import Trimesh, smoothing
from stpyvista import stpyvista
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import matplotlib.colors as mcolors

import requests
from streamlit_autorefresh import st_autorefresh  
from datetime import datetime



# Inicializar Xvfb para PyVista
pv.start_xvfb()

# Estructuras válidas
VALID_STRUCTURES = ["amygdala", "putamen", "pallidum", "hippocampus", "thalamus"]

st.markdown("# Una sola imagen")

if 'proceso_en_curso' not in st.session_state:
    st.session_state['proceso_en_curso'] = False
if 'proceso_recien_empezado' not in st.session_state:
    st.session_state['proceso_recien_empezado'] = False
if 'Segmentacion_terminada' not in st.session_state:
    st.session_state['Segmentacion_terminada'] = False


# Directorio donde se guardarán los archivos CSV
CSV_DIRECTORY = "/app/csv_results"

# Crear el directorio si no existe
os.makedirs(CSV_DIRECTORY, exist_ok=True)

def extract_structure(img_path, structure):
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
    left_output_path = os.path.join("/app", filename.replace(".mgz", f"_left_{structure}.nii"))
    left_img = preprocess_image_file_for_anomaly_detection(left_img)
    nib.save(left_img, left_output_path)
    output_files.append(left_output_path)



    right_structure = np.where(data == right_value, data, 0)
    right_img = nib.Nifti1Image(right_structure, img.affine, img.header)
    right_output_path = os.path.join("/app", filename.replace(".mgz", f"_right_{structure}.nii"))
    right_img = preprocess_image_file_for_anomaly_detection(right_img)
    nib.save(right_img, right_output_path)
    output_files.append(right_output_path)

    st.success(f"Estructuras {structure} extraídas")

    left_volume = (left_img.get_fdata() > 0).sum()
    right_volume = (right_img.get_fdata() > 0).sum()
    volume_difference = abs(left_volume - right_volume)

    # Nombre del archivo CSV para la estructura específica
    csv_file_path = os.path.join(CSV_DIRECTORY, f"{structure}_results.csv")


    st.write(f"La diferencia entre volúmenes es de: {volume_difference}")
    asimetry = asimetria(left_output_path,right_output_path,structure)
    st.session_state.norha_indexs[left_output_path] = asimetry
    # Agregar fila al archivo CSV específico
    new_row = {
        "Nombre de Archivo": filename,
        "Volumen Izquierdo": left_volume,
        "Volumen Derecho": right_volume,
        "Diferencia de Volumen": volume_difference,
        "Asimetria": asimetry
    }
    


    archivos_reemplazados = []

    # Comprobar si algún archivo ya existe
    for nuevo_archivo in output_files:
        for archivo_existente in st.session_state.output_files:
            if os.path.basename(nuevo_archivo) == os.path.basename(archivo_existente):
                # El archivo ya existe, se elimina del sistema y de session_state
                st.session_state.output_files.remove(archivo_existente)  # Elimina el archivo de session_state
                archivos_reemplazados.append(nuevo_archivo)  # Añadir a la lista de reemplazados

    if not archivos_reemplazados:
        if not os.path.exists(csv_file_path):
            df = pd.DataFrame([new_row])
        else:
            df = pd.read_csv(csv_file_path)
            new_row_df = pd.DataFrame([new_row])  # Convertir la nueva fila en un DataFrame
            df = pd.concat([df, new_row_df], ignore_index=True)
        # Guardar de nuevo el DataFrame en el archivo CSV
        df.to_csv(csv_file_path, index=False)

    # Agregar archivos extraídos a la lista de archivos en session_state
    st.session_state.output_files.extend(output_files)
    return left_output_path,right_output_path



def segmentar(img_path, name_subject):
    # Ruta de destino en el volumen compartido
    shared_img_path = os.path.join("/shared", os.path.basename(img_path))
    
    # Mover la imagen al volumen compartido
    shutil.move(img_path, shared_img_path)

    # Hacer una solicitud HTTP al contenedor de segmentación
    response = requests.post("http://segmentacion_service:5000/segmentar", json={
        "img_path": shared_img_path,  # Usar la nueva ruta en el volumen compartido
        "name_subject": name_subject  # Identificador del sujeto
    })

    if response.status_code == 200:
        st.session_state['proceso_en_curso'] = True
        st.session_state['proceso_recien_empezado'] = True
    else:
        st.error(f"Error al iniciar la segmentación: {response.json().get('error')}")

# Función para verificar si la segmentación ha terminado
def verificar_segmentacion(name_subject):
    # Ruta del archivo final
    final_path = os.path.join("/shared", f"{name_subject}.mgz")
    
    # Verificar si el archivo existe en el volumen compartido
    if os.path.exists(final_path):
        st.session_state['proceso_en_curso'] = False
        # Guardar el archivo de salida en el estado de sesión
        if 'output_segmentation_files' not in st.session_state:
            st.session_state['output_segmentation_files'] = []
        st.session_state['output_segmentation_files'].append(final_path)
        st.session_state['Segmentacion_terminada'] = True
        st.rerun()

    else:
        st.info("La segmentación aún está en curso. Por favor, vuelve a comprobar más tarde.")




@st.cache_data
def load_structure(path):
    # Cargar y procesar la estructura (esta parte es costosa)
    structure_data = nib.load(path).get_fdata()
    structure_data_flip = np.flip(structure_data, axis=0).copy()
    verts, faces, normals, values = measure.marching_cubes(structure_data_flip, 0.5)
    faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh = smoothing.filter_laplacian(
        mesh.subdivide_loop(),
        lamb=0.5,
        iterations=10,
        implicit_time_integration=False,
        volume_constraint=True
    )
    return mesh

def visor3D(path_izq, path_der):
    # Obtener el nombre de la estructura
    structure = os.path.splitext(os.path.basename(path_izq).split('_left_')[1])[0]
    cols = st.columns([1, 0.2, 1], gap="medium")

    # Cachear la carga y el procesamiento de las estructuras
    mesh_left = load_structure(path_izq)
    mesh_right = load_structure(path_der)

    # Renderizado para la estructura izquierda
    with cols[0]:
        plotter_left = pv.Plotter(window_size=[300, 300])  # Ventana cuadrada
        plotter_left.add_mesh(mesh_left, cmap='bwr', line_width=1, label=f"Left {structure}")
        plotter_left.add_scalar_bar()
        plotter_left.view_isometric()
        plotter_left.add_text(f"Left {structure}", position='upper_left', font_size=12, color='black')
        stpyvista(plotter_left)

    # Espacio entre los plots
    with cols[1]:
        st.write("")

    # Renderizado para la estructura derecha
    with cols[2]:
        plotter_right = pv.Plotter(window_size=[300, 300])  # Ventana cuadrada
        plotter_right.add_mesh(mesh_right, cmap='bwr', line_width=1, label=f"Right {structure}")
        plotter_right.add_scalar_bar()
        plotter_right.view_isometric()
        plotter_right.add_text(f"Right {structure}", position='upper_left', font_size=12, color='black')
        stpyvista(plotter_right)


@st.cache_data
def process_volume(path_izq):
    """Carga y procesa el volumen cerebral, devolviendo la malla procesada."""
    brain_vol = nib.load(path_izq)
    brain_vol_data = brain_vol.get_fdata()
    brain_vol_data_flip = np.flip(brain_vol_data, axis=0).copy()

    # Aplicar el algoritmo de Marching Cubes
    verts, faces, normals, values = measure.marching_cubes(brain_vol_data_flip, 0.5)
    faces = faces[:, ::-1]

    # Crear la malla y aplicar suavizado
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    subdivided_mesh = mesh.subdivide_loop()
    smoothed_mesh = trimesh.smoothing.filter_laplacian(
        subdivided_mesh, lamb=0.5, iterations=10,
        implicit_time_integration=False, volume_constraint=True
    )
    return smoothed_mesh

@st.cache_data
def process_scalar_field(scalar_field):
    """Aplica un filtro Gaussiano al campo escalar."""
    sigma = 1  # Parámetro del filtro Gaussiano
    return gaussian_filter(scalar_field, sigma)

def malla(path_izq, file_npy):
    st.header("malla")

    # Cachear la carga y procesamiento del volumen cerebral
    smoothed_mesh = process_volume(path_izq)

    # Cachear el suavizado Gaussiano del campo escalar
    smoothed_scalar_field = process_scalar_field(file_npy)

    # Mapear los valores escalares a los vértices de la malla
    vertex_coords = smoothed_mesh.vertices - 0.5
    scalar_values_at_vertices = map_coordinates(smoothed_scalar_field, vertex_coords.T, order=1, mode='nearest')

    # Calcular el valor escalar promedio por cara de la malla
    average_scalar_per_face = scalar_values_at_vertices[smoothed_mesh.faces.astype("int")].mean(axis=1)

    # Convertir la malla a formato PyVista
    pv_mesh = pv.PolyData(smoothed_mesh.vertices, np.hstack((np.full((smoothed_mesh.faces.shape[0], 1), 3), smoothed_mesh.faces)).astype('int'))
    pv_mesh["average_scalar"] = average_scalar_per_face

    # Normalizar y mapear los valores escalares a colores para la visualización
    norm = mcolors.Normalize(vmin=average_scalar_per_face.min(), vmax=average_scalar_per_face.max())
    colormap = plt.cm.plasma
    colors = colormap(norm(average_scalar_per_face))

    # Inicializar un plotter de PyVista
    plotter = pv.Plotter(window_size=[600, 600])
    plotter.add_mesh(pv_mesh, scalars='average_scalar', cmap='plasma', show_edges=True)
    plotter.view_isometric()
    plotter.add_scalar_bar()
    plotter.background_color = 'white'

    # Mostrar el gráfico en Streamlit
    stpyvista(plotter)






def tab_1():
    st.title("Segmentación de imagen")

    # Inicializar la lista de archivos extraídos en session_state si no existe
    if "output_segmentation_files" not in st.session_state:
        st.session_state.output_segmentation_files = []
    if 'comienzo' not in st.session_state:
        st.session_state.comienzo = None

    # Subir archivo

    with st.form("form_segmentar"):
        # Subir archivo
        uploaded_file = st.file_uploader("Subir archivo .nii", type=["nii"])
        if uploaded_file is not None:
            img_path = os.path.join("/app", uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        name_subject = st.text_input("Ingrese el nombre del paciente")

        boton_segmentar = st.form_submit_button('Segmentar', disabled=st.session_state['proceso_en_curso'])

        if boton_segmentar:
            if uploaded_file and name_subject:
                if os.path.exists(f"/shared/{name_subject}.mgz"):
                    st.warning(f"El archivo {name_subject}.mgz ya existe.")
                else:
                    st.session_state['name_subject'] = name_subject
                    segmentar(img_path, name_subject)
                    st.session_state.comienzo = datetime.now()
                    st.rerun()
            else:
                st.error("Por favor, sube un archivo e ingresa el nombre del paciente.")

    if st.session_state['Segmentacion_terminada']:
        st.success(f"Segmentación completada. Archivo disponible en el lado izquierdo superior de la pagina para descargar")
        st.session_state['Segmentacion_terminada'] = False

    if st.session_state['proceso_en_curso'] and not st.session_state['proceso_recien_empezado']:
        st.write('Segmentacion en proceso, esto tarda alrededor de 15 minutos')
        verificar_segmentacion(st.session_state['name_subject'])
        delta = datetime.now() - st.session_state.comienzo
        minutos = delta.seconds // 60
        segundos = delta.seconds % 60
        st.write(f"Han transcurrido {minutos} minutos y {segundos} segundos")
        if st.button('Actualizar'):
            pass
    
    if st.session_state['proceso_recien_empezado']:
        st.success("Segmentación iniciada exitosamente.")
        st.write('Este proceso tarda alrededor de 15 minutos')
        st.session_state['proceso_recien_empezado'] = False
        if st.button('Actualizar'):
            pass
            


def tab_2():
    st.title("Extracción de Estructuras Cerebrales")

    # Inicializar la lista de archivos extraídos en session_state si no existe
    if "output_files" not in st.session_state:
        st.session_state.output_files = []
    if 'norha_indexs' not in st.session_state:
        st.session_state.norha_indexs = {}
    if 'mapa_npys' not in st.session_state:
        st.session_state.mapa_npys = {}

    # Inicializar las variables para almacenar los paths si no existen en session_state
    if 'path_izq' not in st.session_state:
        st.session_state.path_izq = None
    if 'path_der' not in st.session_state:
        st.session_state.path_der = None

    if 'extraido' not in st.session_state:
        st.session_state.extraido = False  # Flag para controlar si se ha extraído
    if 'malla_vista' not in st.session_state:
        st.session_state.malla_vista = False  # Flag para controlar si se vio la malla

    # Opciones de archivos generados en `tab_1` y archivo subido manualmente
    with st.form("form_extraer"):
        #file_options = [None] + st.session_state.output_segmentation_files
        file_options = [None] + [file.replace("/shared/", "") for file in st.session_state.output_segmentation_files]
        selected_file = None
        if st.session_state.output_segmentation_files:
            selected_file = st.selectbox("Seleccionar archivo de los que segmentaste", file_options)

        # Subir archivo
        uploaded_file2 = st.file_uploader("Subir archivo .mgz", type=["mgz"])
        img_path2 = None

        if selected_file is not None:
            img_path2 = f'/shared/{selected_file}'
        elif uploaded_file2 is not None:
            img_path2 = os.path.join("/app", uploaded_file2.name)
            with open(img_path2, "wb") as f:
                f.write(uploaded_file2.getbuffer())

        structure = st.selectbox("Selecciona una estructura a extraer", VALID_STRUCTURES)
        extraer = st.form_submit_button('Extraer')

    if extraer:
        if img_path2:
            # Extraer las estructuras y guardar los paths en session_state
            path_izq, path_der = extract_structure(img_path2, structure)
            st.session_state.path_izq = path_izq
            st.session_state.path_der = path_der
            nombre = os.path.basename(path_izq).split('_left')[0]
            structure = os.path.splitext(os.path.basename(path_izq).split('_left_')[1])[0]
            st.session_state.structure = structure
            st.session_state.extraido = True
            st.session_state.malla_vista = False
            with st.expander(f"{structure} de la imagen {nombre}"):
                visor3D(path_izq, path_der)

            # Establecer la bandera de extracción en True y resetear la de malla
             
        else:
            st.error("Por favor, selecciona un archivo segmentado o sube uno manualmente y selecciona una estructura.")
    
    # Mostrar el botón "Ver malla" solo si ya se realizó la extracción y aún no se visualizó la malla
    if st.session_state.extraido:
        if st.button("Ver malla",disabled=st.session_state.malla_vista):
            # Usar los paths almacenados en session_state
            if st.session_state.path_izq is not None and st.session_state.path_der is not None:
                with st.spinner('Creando la malla de la imagen, por favor espera...'):
                    file_npy = generar_npy(st.session_state.path_izq, st.session_state.path_der,st.session_state.structure)
                st.session_state.malla_vista = True
                st.session_state.mapa_npys[st.session_state.path_izq] = file_npy
                st.rerun()
    
            else:
                st.error("ya se hizo la malla de esta imagen")

    if st.session_state.malla_vista:
        st.session_state.malla_vista = False
        st.session_state.extraido = False
        nombre = os.path.basename(st.session_state.path_izq).split('_left')[0]
        structure = os.path.splitext(os.path.basename(st.session_state.path_izq).split('_left_')[1])[0]
        grafico_norah(st.session_state.norha_indexs[st.session_state.path_izq])
        with st.expander(f"{structure} de la imagen {nombre}"):
            visor3D(st.session_state.path_izq, st.session_state.path_der)
            malla(st.session_state.path_izq, st.session_state.mapa_npys[st.session_state.path_izq])
        st.session_state.path_izq = None
        st.session_state.path_der = None
        st.write('Se podra ver todo esto en la pestaña "imagenes"')
    






    
def tab_3():
    if 'output_files' in st.session_state and len(st.session_state.output_files) > 0:
        # Suponiendo que los paths están en st.session_state.output_files
        paths = st.session_state.output_files
        # Recorremos los paths de a 2
        for i in range(0, len(paths), 2):
            if i + 1 < len(paths):  # Asegurarse de que exista un par
                path_izquierdo = paths[i]
                path_derecho = paths[i + 1]
                nombre = os.path.basename(path_izquierdo).split('_left')[0]
                structure = os.path.splitext(os.path.basename(path_izquierdo).split('_left_')[1])[0]
                with st.expander(f"{structure} de la imagen {nombre}"):
                    visor3D(path_izquierdo, path_derecho)
                    if path_izquierdo in st.session_state.mapa_npys:
                        malla(path_izquierdo, st.session_state.mapa_npys[path_izquierdo])
                    grafico_norah(st.session_state.norha_indexs[path_izquierdo])
    else:
        st.write("Aun no hay imagenes para mostrar")




def tab_4():
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
    NORAH_index,c,model,Example_input = norah(image,imagepair,structure)
    st.write(f"El indice Norah es: {NORAH_index}")
    grafico_norah(NORAH_index)
    #print('entrando')
    #control_right_occluded, infra_array, ultra_array = negative_occlusion(Example_input, 8, 8, c, model)
    #print('occlusion finished')
    ##############################################
    #file_npy = control_right_occluded[:][:][:].cpu().detach().numpy()
    ##############################################
    return NORAH_index #, file_npy

def generar_npy(image,imagepair,structure):
    file_npy = occlusion(image,imagepair,structure)
    return file_npy


def occlusion(image,imagepair,structure):
    NORAH_index,c,model,Example_input = norah(image,imagepair,structure)
    print('entrando')
    control_right_occluded, infra_array, ultra_array = negative_occlusion(Example_input, 8, 8, c, model)
    print('occlusion finished')
    ##############################################
    file_npy = control_right_occluded[:][:][:].cpu().detach().numpy()
    ##############################################
    return file_npy


def norah(image,imagepair,structure):
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
    model = torch.load(f"/app/Models/{structure}.pt")
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
    return  NORAH_index,c,model,Example_input


def grafico_norah(NORAH_index):
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
    point_x = NORAH_index * np.cos(angle_rad)
    point_y = NORAH_index * np.sin(angle_rad)

    # Dibujar el punto negro
    ax.plot(point_x, point_y, 'd')  # 'ko' indica un punto negro

    # Configuración del gráfico
    ax.set_aspect('equal')
    ax.set_xlim(-35, 35)
    ax.set_ylim(0, 35)  # Cambiado para mostrar solo la parte superior
    ax.axis('off')

    # Mostrar el gráfico en streamlit
    #st.pyplot(fig)
    st.pyplot(plt)

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

def negative_occlusion(img, patch, stride, c, trainedModel):
    infra_array, ultra_array = [], []
    image = img[:][0][0][0]
    counter_image = img[:][0][0][1]
    occluded_baseline = torch.zeros(64,64,64)
    negative_occluded_baseline = torch.zeros(64,64,64)
    positive_occluded_baseline = torch.zeros(64,64,64)

    H, W, L = image.shape
    patch_H, patch_W, patch_L = patch,patch,patch
    stride=stride
    mean=0

    anchors = []
    grid_l = 0
    while grid_l < L :
        grid_h = 0
        while grid_h < H :
            grid_w = 0
            while grid_w <= W - patch_W:
                anchors.append((grid_l, grid_h, grid_w))
                grid_w += stride  
            grid_h += stride
        grid_l += stride
    
    result = trainedModel.predict(img).cpu().detach().numpy()
    infra_array.append(c.reshape(1,32)) #center is saved in the list
    infra_array.append(result) #Original distance is also saved in the list
    original_distance = torch.sum((torch.from_numpy(result) - c) ** 2, dim=1).numpy()
    
        
    for i in anchors:
        grid_l, grid_h, grid_w = i[0],i[1],i[2]
        images_ = image.clone()
        counter_images_ = counter_image.clone()

        images_[..., grid_l : grid_l + patch_L, grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
        counter_images_[..., grid_l : grid_l + patch_L, grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean

        x=torch.unsqueeze(torch.vstack((torch.unsqueeze(images_,dim=0),torch.unsqueeze(counter_images_,dim=0))),dim=0)

        control_right = [x,img[:][1],img[:][2]]
        result = trainedModel.predict(control_right).cpu().detach().numpy()

        new_distance = torch.sum((torch.from_numpy(result) - c) ** 2, dim=1).numpy()
        A = original_distance - new_distance
        occluded_baseline[..., grid_l : grid_l + patch_L, grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] += A
        
        if A > 0:
            negative_occluded_baseline[..., grid_l : grid_l + patch_L, grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] += A
        if A < 0:
            positive_occluded_baseline[..., grid_l : grid_l + patch_L, grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] += A

    negative_occluded_baseline = occluded_baseline
    negative_occluded_baseline[negative_occluded_baseline < 0] = 0
    print('end of the road')
    return occluded_baseline, negative_occluded_baseline, positive_occluded_baseline


tab1, tab2, tab3, tab4 = st.tabs(["Segmentar Imagen", "Imagen ya Segmentada","imagenes" ,"Resultados"])
with tab1:
    tab_1()

with tab2:
    tab_2()

with tab3:
    tab_3()

with tab4:
    tab_4()


if "output_segmentation_files" in st.session_state or "output_files" in st.session_state:
    st.sidebar.subheader("Archivos de Salida")

with st.sidebar.expander("Segmentaciones"):
    if "output_segmentation_files" in st.session_state:
        # Mostrar archivos extraídos
        for file_path in st.session_state.output_segmentation_files:
            with open(file_path, "rb") as file:
                st.download_button(
                    label=f"Descargar {os.path.basename(file_path)}",
                    data=file,
                    file_name=os.path.basename(file_path)
                )

with st.sidebar.expander("Extracciones"):
    if "output_files" in st.session_state:
        # Mostrar archivos extraídos
        for file_path in st.session_state.output_files:
            with open(file_path, "rb") as file:
                st.download_button(
                    label=f"Descargar {os.path.basename(file_path)}",
                    data=file,
                    file_name=os.path.basename(file_path)
                )








