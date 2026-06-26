import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("graficos_ppt", exist_ok=True)

def draw_block(ax, x, y, width, height, color, text):
    rect = patches.FancyBboxPatch((x, y), width, height, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=12, weight='bold', color='white')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.01, width=2, headwidth=10))

def plot_vertical_architecture(filename, title, labels, colors, widths):
    n_blocks = len(labels)
    # Ajustamos el alto de la figura segun la cantidad de bloques
    fig, ax = plt.subplots(figsize=(6, 1.5 * n_blocks + 1))
    
    # Fijamos limites. X va de -4 a 4, Y se ajusta.
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 1.8 * n_blocks + 0.5)
    ax.axis('off')
    
    y_pos = 1.8 * n_blocks - 1
    
    for i in range(n_blocks):
        w = widths[i]
        h = 1.0
        x = -w / 2
        
        draw_block(ax, x, y_pos, w, h, colors[i], labels[i])
        
        if i < n_blocks - 1:
            # Draw arrow pointing down
            draw_arrow(ax, 0, y_pos, 0, y_pos - 0.8)
            
        y_pos -= 1.8
        
    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'graficos_ppt/{filename}', dpi=300)
    plt.close()

def plot_custom_cnn():
    colors = ['#34495e', '#2980b9', '#2980b9', '#2980b9', '#2980b9', '#e67e22', '#c0392b', '#27ae60']
    labels = ['Imágen de Entrada\n(224x224 px)', 
              'Conv2D + MaxPooling\n(32 filtros)', 
              'Conv2D + MaxPooling\n(64 filtros)', 
              'Conv2D + MaxPooling\n(128 filtros)', 
              'Conv2D + Normalización\n(256 filtros)', 
              'Global Average Pooling\n(Promedio espacial)', 
              'Capa Densa Oculta\n(256 neuronas)', 
              'Clasificación Final\n(10 clases)']
    widths = [4.5] * len(labels)
    plot_vertical_architecture('5_arq_custom.png', 'Arquitectura: Custom CNN\n(Construida desde cero)', labels, colors, widths)

def plot_vgg16():
    colors = ['#34495e', '#8e44ad', '#e67e22', '#c0392b', '#27ae60']
    labels = ['Imágen de Entrada\n(224x224 px)', 
              'Extractor de VGG16\n(13 Capas Convolucionales)\nPreentrenado en ImageNet', 
              'Global Average Pooling', 
              'Capa Densa Oculta\n(256 neuronas)', 
              'Clasificación Final\n(10 clases)']
    widths = [4.5, 6.0, 4.5, 4.5, 4.5]
    plot_vertical_architecture('6_arq_vgg16.png', 'Arquitectura: VGG16\n(Transfer Learning)', labels, colors, widths)

def plot_mobilenet():
    colors = ['#34495e', '#16a085', '#e67e22', '#c0392b', '#27ae60']
    labels = ['Imágen de Entrada\n(224x224 px)', 
              'Extractor MobileNetV2\n(53 Capas Convolucionales)\nPreentrenado en ImageNet', 
              'Global Average Pooling', 
              'Capa Densa Oculta\n(256 neuronas)', 
              'Clasificación Final\n(10 clases)']
    widths = [4.5, 6.5, 4.5, 4.5, 4.5]
    plot_vertical_architecture('7_arq_mobilenet.png', 'Arquitectura: MobileNetV2\n(Transfer Learning Eficiente)', labels, colors, widths)

if __name__ == "__main__":
    print("Generando diagramas verticales...")
    plot_custom_cnn()
    plot_vgg16()
    plot_mobilenet()
    print("Diagramas guardados.")
