import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

def show_train_subsample(train_images_path:Path, titles=True):
    '''
    Muestra una cuadrícula de 6 imágenes seleccionadas aleatoriamente desde un directorio.

    Params:
        train_images_path: Ruta al directorio que contiene las imágenes en formato JPG.
        titles: Si es True, muestra como título el nombre de cada archivo de imagen.

    Returns:
        None
    '''

    image_files = list(train_images_path.glob("*.jpg"))
    sampled_images = np.random.choice(image_files, size=6, replace=False)

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    for ax, img_path in zip(axs.flatten(), sampled_images):

        img = plt.imread(str(img_path)) 
        ax.imshow(img)
        
        if titles:
            ax.set_title(img_path.name, fontsize=20)
        
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_class_histogram(train_labels_path:Path, labels:list|dict):
    '''
    Genera y muestra un histograma del número de instancias por clase a partir de archivos de etiquetas en formato YOLO.

    Params:
        train_labels_path: Ruta al directorio que contiene los archivos de etiquetas (.txt), donde cada archivo incluye líneas con formato YOLO:
            `<class_id> <x_center> <y_center> <width> <height>`.
        labels: Estructura que mapea `class_id` (entero) a su nombre de clase. Puede ser una lista (índice = id) o un diccionario.

    Returns:
        None
    '''

    all_classes = []
    label_files = list(train_labels_path.glob("*.txt"))

    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    class_id = int(line.split()[0])
                    all_classes.append(labels[class_id])

    class_counts = Counter(all_classes)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')

    plt.xlabel('Clase')
    plt.ylabel('Cantidad de Instancias')
    plt.title('Balance de Clases')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def show_instance_histogram(train_labels_path:Path, nbins=27):
    '''
    Muestra un histograma de la cantidad de instancias por imagen en un conjunto de etiquetas en formato YOLO.

    Params:
        train_labels_path: Ruta al directorio que contiene los archivos de etiquetas (.txt), donde cada línea representa una instancia anotada.
        nbins: Número de bins a utilizar en el histograma.

    Returns:
        None
    '''

    instances_per_image = []
    label_files = list(train_labels_path.glob("*.txt"))


    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = [line for line in f.readlines() if line.strip()]
            instances_per_image.append(len(lines))

    plt.figure(figsize=(10, 6))
    plt.hist(instances_per_image, bins=nbins, color='salmon', edgecolor='black', alpha=0.7)

    plt.title('Distribución de Instancias por Frame')
    plt.xlabel('Cantidad de objetos en la foto')
    plt.ylabel('Cantidad de fotos')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    mean_instances = np.mean(instances_per_image)
    max_instances = np.max(instances_per_image)

    textstr = '\n'.join((
        f'Total Frames: {len(instances_per_image)}',
        f'Promedio: {mean_instances:.2f} objetos/imágen',
        f'Máximo: {max_instances} objetos/imágen'
        ))

    plt.gca().text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.show()

