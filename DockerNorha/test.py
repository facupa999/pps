import numpy as np
import torch
import nibabel as nb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("izq_dir", help="ruta imagen izquierda")
parser.add_argument("der_dir", help="ruta imagen derecha")
parser.add_argument("structure", help="Estructura a analizar (amygdala, putamen, pallidum, hippocampus, thalamus)")

args = parser.parse_args()

izq_dir = args.izq_dir
der_dir = args.der_dir
estructura = args.structure.lower()



image = nb.load(izq_dir)
image = np.array(image.get_fdata())

imagepair = nb.load(der_dir)
imagepair = np.array(imagepair.get_fdata())
imagepair = np.flip(imagepair,axis=0).copy()


image = torch.tensor(image)
imagepair = torch.tensor(imagepair)

left_volume, right_volume = np.count_nonzero(imagepair), np.count_nonzero(image)
left_volume = torch.unsqueeze(torch.from_numpy(np.array(left_volume)), 0)
right_volume = torch.unsqueeze(torch.from_numpy(np.array(right_volume)), 0)


Example_input = [torch.unsqueeze(torch.cat((torch.unsqueeze(image, 0),torch.unsqueeze(imagepair, 0)), 0), 0),left_volume, right_volume]

from Models.LeNet_ELU_SVDD import Encoder  # Importar la clase o módulo necesario
# Model class must be defined somewhere
model = torch.load(f"/app/Models/{estructura}.pt")
model.eval()

inference = model(Example_input).cpu().detach().numpy()

c=torch.from_numpy(np.array([-7.3644e-04,  2.5405e-04, -3.1788e-03, -1.6109e-03, -7.6053e-04,
        -5.0173e-04,  2.2761e-03,  1.0202e-03, -2.8073e-03,  1.0238e-03,
         3.1727e-04, -1.2766e-03,  3.5283e-05, -7.2480e-04, -2.5971e-04,
         9.8939e-04, -1.7135e-03,  1.4254e-03,  1.3150e-05, -1.0366e-03,
        -9.3122e-04, -1.1257e-03, -1.0345e-03,  1.4835e-03,  1.4995e-03,
         1.2943e-03, -9.0330e-04,  1.8791e-03,  6.6228e-04, -2.2420e-04,
         3.5120e-03,  3.4326e-03]))

NORAH_index = torch.sum((torch.from_numpy(inference) - c) ** 2, dim=1).numpy()

print(NORAH_index*200000)


import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap

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

# Mostrar el gráfico
plt.show()
