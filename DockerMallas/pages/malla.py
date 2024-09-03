import nibabel as nib
import os
import numpy as np
from skimage import measure
import trimesh
import streamlit as st
from stpyvista import stpyvista
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from trimesh import Trimesh, smoothing
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Inicializar Xvfb para PyVista
pv.start_xvfb()

def tab_1():
    st.title("Resultados")
    current_dir = os.getcwd()
    # Columns for left and right thalamus with increased separation
    cols = st.columns([1, 0.2, 1], gap="medium")


    uploaded_file = st.file_uploader("Subir archivo .nii del lado izquierdo", type=["nii"])
    if uploaded_file is not None:
        img_path = os.path.join("/app", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    uploaded_file2 = st.file_uploader("Subir archivo .nii del lado derecho", type=["nii"])
    if uploaded_file2 is not None:
        img_path2 = os.path.join("/app", uploaded_file2.name)
        with open(img_path2, "wb") as f:
            f.write(uploaded_file2.getbuffer())


    if uploaded_file and uploaded_file2:
        # File paths for left and right thalamus
        left_thalamus_path = img_path
        right_thalamus_path = img_path2


        # Load left thalamus data
        left_thalamus = nib.load(left_thalamus_path)
        left_thalamus_data = left_thalamus.get_fdata()
        left_thalamus_data_flip = np.flip(left_thalamus_data, axis=0).copy()

        # Extract surface mesh for left thalamus
        verts_left, faces_left, normals_left, values_left = measure.marching_cubes(left_thalamus_data_flip, 0.5)
        faces_left = faces_left[:, ::-1]
        mesh_left = trimesh.Trimesh(vertices=verts_left, faces=faces_left)
        mesh_left = smoothing.filter_laplacian(
            mesh_left.subdivide_loop(),
            lamb=0.5,
            iterations=10,
            implicit_time_integration=False,
            volume_constraint=True
        )

        # Load right thalamus data
        right_thalamus = nib.load(right_thalamus_path)
        right_thalamus_data = right_thalamus.get_fdata()
        right_thalamus_data_flip = np.flip(right_thalamus_data, axis=0).copy()

        # Extract surface mesh for right thalamus
        verts_right, faces_right, normals_right, values_right = measure.marching_cubes(right_thalamus_data_flip, 0.5)
        faces_right = faces_right[:, ::-1]
        mesh_right = trimesh.Trimesh(vertices=verts_right, faces=faces_right)
        mesh_right = smoothing.filter_laplacian(
            mesh_right.subdivide_loop(),
            lamb=0.5,
            iterations=10,
            implicit_time_integration=False,
            volume_constraint=True
        )

        # Plot left thalamus
        with cols[0]:
            plotter_left = pv.Plotter(window_size=[300, 300])  # Set to square window
            plotter_left.add_mesh(mesh_left, cmap='bwr', line_width=1, label="Left Thalamus")
            plotter_left.add_scalar_bar()
            plotter_left.view_isometric()
            plotter_left.add_text("Left Thalamus", position='upper_left', font_size=12, color='black')  # Add title
            stpyvista(plotter_left)

        # Add a space between plots for separation
        with cols[1]:
            st.write("")  # This column is used to create space between the plots

        # Plot right thalamus
        with cols[2]:
            plotter_right = pv.Plotter(window_size=[300, 300])  # Set to square window
            plotter_right.add_mesh(mesh_right, cmap='bwr', line_width=1, label="Right Thalamus")
            plotter_right.add_scalar_bar()
            plotter_right.view_isometric()
            plotter_right.add_text("Right Thalamus", position='upper_left', font_size=12, color='black')  # Add title
            stpyvista(plotter_right)


def tab_2():
    st.title("Malla")

    ## Initialize a plotter object
    plotter = pv.Plotter(window_size=[600,600])

    uploaded_file3 = st.file_uploader("Subir archivo .nii del lado izquierdo ", type=["nii"])
    if uploaded_file3 is not None:
        img_path3 = os.path.join("/app", uploaded_file3.name)
        with open(img_path3, "wb") as f:
            f.write(uploaded_file3.getbuffer())

    # Load the brain volume data
    #brain_vol = nib.load('D:/Documents/gitProyectos/pps/DockerMallas/pages/r_ADNI_003left.nii')
    if uploaded_file3:
        brain_vol = nib.load(img_path3)

        # Check the type of the brain volume object
        print(type(brain_vol))


        # Extract data from the brain volume object
        brain_vol_data = brain_vol.get_fdata()
        brain_vol_data_flip = np.flip(brain_vol_data,axis=0).copy()
        print(type(brain_vol_data))
        print(brain_vol_data.shape)


        # Apply the marching cubes algorithm to extract surface mesh
        verts, faces, normals, values = measure.marching_cubes(brain_vol_data_flip, 0.5)

        # Orient faces
        faces = faces[:, ::-1]


        # Load the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Apply Loop subdivision
        subdivided_mesh = mesh.subdivide_loop()
        # Apply Laplacian smoothing
        smoothed_mesh = trimesh.smoothing.filter_laplacian(subdivided_mesh, lamb=0.5, iterations=10, implicit_time_integration=False, volume_constraint=True, laplacian_operator=None)



        uploaded_file4 = st.file_uploader("Subir archivo .npy del lado derecho  ", type=["npy"])
        if uploaded_file4 is not None:
            img_path = os.path.join("/app", uploaded_file4.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file4.getbuffer())
        # Load a 3D scalar field to overlay onto the surface mesh
        #scalar_field = np.load('D:/Documents/gitProyectos/pps/DockerMallas/pages/r_ADNI_003right_EMMA_pixel.npy')
        if uploaded_file4:
            scalar_field = np.load(uploaded_file4)


            # Apply Gaussian smoothing
            sigma = 1  # Standard deviation of the Gaussian kernel, adjust as needed
            smoothed_scalar_field = gaussian_filter(scalar_field, sigma)

            # Map scalar values from the field onto the mesh vertices
            vertex_coords = smoothed_mesh.vertices - 0.5
            scalar_values_at_vertices = map_coordinates(smoothed_scalar_field, vertex_coords.T, order=1, mode='nearest')

            # Calculate average scalar value per face of the mesh
            average_scalar_per_face = scalar_values_at_vertices[smoothed_mesh.faces.astype("int")].mean(axis=1)
            # Convert the mesh to PyVista format
            pv_mesh = pv.PolyData(smoothed_mesh.vertices, np.hstack((np.full((smoothed_mesh.faces.shape[0], 1), 3), smoothed_mesh.faces)).astype('int'))

            # Add scalar data to PyVista mesh
            pv_mesh["average_scalar"] = average_scalar_per_face

            # Normalize and map the scalar values to colors for visualization
            norm = mcolors.Normalize(vmin=average_scalar_per_face.min(), vmax=average_scalar_per_face.max())
            colormap = plt.cm.plasma
            colors = colormap(norm(average_scalar_per_face))

            # Initialize a PyVista plotter
            plotter = pv.Plotter(window_size=[600, 600])
            plotter.add_mesh(pv_mesh, scalars='average_scalar', cmap='plasma', show_edges=True)

            # Final touches
            plotter.view_isometric()
            plotter.add_scalar_bar()
            plotter.background_color = 'white'

            # Display the PyVista plot using stpyvista
            stpyvista(plotter, key="pv_mesh")

tab1, tab2 = st.tabs(["Resultados", "Malla"])
with tab1:
    tab_1()

with tab2:
    tab_2()

