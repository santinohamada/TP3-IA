import json
import sys
import re

def limpiar_notebook_agresivo(file_path):
    print(f"Limpiando a fondo logs de {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_path}")
        return

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
            
        new_outputs = []
        stdout_text = ""
        
        # Juntar todo el stdout en un solo string gigante
        for output in cell.get('outputs', []):
            if output.get('name') == 'stdout' and output.get('output_type') == 'stream':
                # text can be a string or a list of strings
                if isinstance(output['text'], list):
                    stdout_text += "".join(output['text'])
                else:
                    stdout_text += output['text']
            else:
                new_outputs.append(output) # Mantener display_data, errores, etc.
                
        if not stdout_text:
            continue
            
        # Ahora limpiar el stdout
        # Primero, aplicar los \r (carriage returns) como haría una consola real
        # Si hay un \r, nos quedamos solo con lo que está a la derecha del último \r en esa línea
        clean_lines = []
        for line in stdout_text.split('\n'):
            if '\r' in line:
                line = line.split('\r')[-1]
                
            # Criterio estricto: 
            # Guardamos la linea si es "Epoch X/Y" o si tiene "val_loss"
            # O si no parece ser una linea de keras progress bar en absoluto
            line_stripped = line.strip()
            
            es_progreso = 'ms/step' in line or 's/step' in line or '━' in line or 'loss:' in line
            tiene_val = 'val_loss' in line or 'val_accuracy' in line
            es_epoch = line_stripped.startswith('Epoch')
            
            if es_epoch:
                clean_lines.append(line)
            elif es_progreso:
                if tiene_val:
                    clean_lines.append(line)
                # Si es progreso pero NO tiene val_, la ignoramos (es intermedia)
            else:
                if line_stripped != "":
                    clean_lines.append(line)
                    
        if clean_lines:
            # Recrear un solo bloque de stdout
            final_text = [l + '\n' for l in clean_lines]
            new_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": final_text
            })
            
        cell['outputs'] = new_outputs

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"-> ¡Limpieza profunda completada en {file_path}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Archivos a limpiar")
    args = parser.parse_args()
    
    if args.notebooks:
        for nb_path in args.notebooks:
            limpiar_notebook_agresivo(nb_path)
