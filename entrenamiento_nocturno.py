import subprocess
import sys
import time

# Lista de los notebooks en el orden que querés que se ejecuten.
# Puse VGG16 al final porque suele ser el más pesado.
notebooks = [
    "TP3_IA-CM.ipynb",
    "TP3_IA-MBNet.ipynb",
    "TP3_IA-VGG16.ipynb",
    "TP3_IA-Comparacion.ipynb"
]

def run_notebook(notebook_path):
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO ENTRENAMIENTO: {notebook_path}")
    print(f"{'='*60}\n")
    
    # jupyter nbconvert permite ejecutar un notebook desde la terminal.
    # --inplace guarda los outputs (gráficos, logs) en el mismo archivo para que los veas mañana.
    comando = [
        sys.executable, "-m", "jupyter", "nbconvert", 
        "--to", "notebook", 
        "--execute", notebook_path, 
        "--inplace",
        "--ExecutePreprocessor.timeout=-1"  # Desactiva el timeout (clave para entrenamientos largos)
    ]
    
    try:
        # Ejecuta el comando y bloquea hasta que termine
        subprocess.run(comando, check=True)
        print(f"\n✅ FINALIZADO CON ÉXITO: {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR AL EJECUTAR: {notebook_path}")
        print("El script continuará con el siguiente modelo para no perder la noche.")
        print(f"Detalle del error: {e}\n")

if __name__ == "__main__":
    print("🌙 Iniciando pipeline de entrenamiento nocturno secuencial...")
    start_time = time.time()
    
    for nb in notebooks:
        run_notebook(nb)
        
    end_time = time.time()
    horas = (end_time - start_time) / 3600
    
    print(f"\n{'='*60}")
    print(f"🎉 ¡PIPELINE TERMINADO!")
    print(f"Tiempo total transcurrido: {horas:.2f} horas.")
    print("Ya podés revisar los archivos .keras y los gráficos generados en cada notebook.")
    print(f"{'='*60}\n")
