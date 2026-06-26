import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración de estilo para presentaciones (fuentes grandes, sin bordes feos)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk") # Hace que las fuentes y líneas sean más gruesas (ideal para PPT)

# Crear carpeta si no existe
os.makedirs("graficos_ppt", exist_ok=True)

# ==========================================
# 1. Gráfico de Desbalance (Dataset)
# ==========================================
def plot_dataset_distribution():
    # Valores aproximados multiplicando el support del test set x10
    clases = ['Araña', 'Ardilla', 'Caballo', 'Elefante', 'Gallina', 
              'Gato', 'Mariposa', 'Oveja', 'Perro', 'Vaca']
    cantidades = [4830, 1870, 2630, 1460, 3110, 1680, 2120, 1820, 4870, 1880]
    
    # Ordenar para que el desbalance sea más evidente visualmente
    sorted_indices = sorted(range(len(cantidades)), key=lambda k: cantidades[k], reverse=True)
    clases_sorted = [clases[i] for i in sorted_indices]
    cantidades_sorted = [cantidades[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    
    # Colores: Resaltar la clase mayoritaria (Perro) y minoritaria (Elefante)
    colores = ['#e74c3c' if c in ['Perro', 'Elefante'] else '#3498db' for c in clases_sorted]
    
    ax = sns.barplot(x=clases_sorted, y=cantidades_sorted, palette=colores)
    
    plt.title('Distribución de Imágenes por Clase (El problema del desbalance)', fontsize=20, pad=20, weight='bold')
    plt.ylabel('Cantidad de Imágenes', fontsize=16)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    
    # Agregar los números arriba de cada barra
    for i, v in enumerate(cantidades_sorted):
        ax.text(i, v + 100, str(v), ha='center', fontsize=14, color='black', weight='bold')

    plt.tight_layout()
    plt.savefig('graficos_ppt/1_desbalance_clases.png', dpi=300)
    plt.close()
    print("Gráfico de desbalance guardado.")

# ==========================================
# 2. Gráfico de División de Datos (Pie Chart)
# ==========================================
def plot_data_split():
    labels = ['Entrenamiento\n(Train - 70%)', 'Validación\n(Val - 15%)', 'Prueba\n(Test - 15%)']
    sizes = [70, 15, 15]
    colors = ['#2ecc71', '#f1c40f', '#9b59b6']
    explode = (0.05, 0, 0)  # Separar un poco el Train para destacar

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140, 
            textprops={'fontsize': 16, 'weight': 'bold'})
    
    plt.title('División del Dataset', fontsize=22, pad=20, weight='bold')
    
    plt.tight_layout()
    plt.savefig('graficos_ppt/2_division_datos.png', dpi=300)
    plt.close()
    print("Gráfico de división guardado.")

# ==========================================
# 3. Gráfico de Comparativa de Modelos
# ==========================================
def plot_model_comparison():
    modelos = ['Custom CNN', 'MobileNetV2', 'VGG16']
    accuracy = [71.68, 95.58, 96.12]
    
    plt.figure(figsize=(10, 6))
    
    # Colores: Gris para Custom (para mostrar que quedó atrás), verdes para los exitosos
    colores = ['#95a5a6', '#27ae60', '#2ecc71']
    
    ax = sns.barplot(x=modelos, y=accuracy, palette=colores)
    
    plt.title('Comparativa Final de Accuracy en Test', fontsize=20, pad=20, weight='bold')
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.ylim(0, 110) # Dar un poco de espacio arriba
    
    # Agregar porcentajes en las barras
    for i, v in enumerate(accuracy):
        ax.text(i, v + 2, f"{v}%", ha='center', fontsize=18, color='black', weight='bold')

    # Línea horizontal para marcar el 95% de referencia
    plt.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    plt.text(-0.4, 96, 'Barrera del 95%', color='red', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('graficos_ppt/3_comparativa_modelos.png', dpi=300)
    plt.close()
    print("Gráfico de comparativa guardado.")

# ==========================================
# 4. Gráfico de Pesos de Clases (Class Weights)
# ==========================================
def plot_class_weights():
    clases = ['Araña', 'Ardilla', 'Caballo', 'Elefante', 'Gallina', 
              'Gato', 'Mariposa', 'Oveja', 'Perro', 'Vaca']
    cantidades = [4830, 1870, 2630, 1460, 3110, 1680, 2120, 1820, 4870, 1880]
    
    total_samples = sum(cantidades)
    n_classes = len(clases)
    
    # Calcular class weights: n_samples / (n_classes * class_count)
    pesos = [total_samples / (n_classes * count) for count in cantidades]
    
    # Ordenar por peso de mayor a menor para que visualmente se vea a quién se ayuda más
    sorted_indices = sorted(range(len(pesos)), key=lambda k: pesos[k], reverse=True)
    clases_sorted = [clases[i] for i in sorted_indices]
    pesos_sorted = [pesos[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    # Color: Naranja intenso para mostrar "penalización/peso extra"
    ax = sns.barplot(x=clases_sorted, y=pesos_sorted, color='#e67e22')
    
    plt.title('Pesos Asignados a cada Clase (Class Weights)', fontsize=20, pad=20, weight='bold')
    plt.ylabel('Multiplicador de Peso', fontsize=16)
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    
    # Agregar los valores numéricos arriba de cada barra
    for i, v in enumerate(pesos_sorted):
        ax.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=14, color='black', weight='bold')

    plt.tight_layout()
    plt.savefig('graficos_ppt/4_pesos_clases.png', dpi=300)
    plt.close()
    print("Gráfico de pesos guardado.")

if __name__ == "__main__":
    print("Generando gráficos para la presentación...")
    plot_dataset_distribution()
    plot_data_split()
    plot_model_comparison()
    plot_class_weights()
    print("¡Listo! Los gráficos están en la carpeta 'graficos_ppt'.")
