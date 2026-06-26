import re

with open('informe_tp3.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix "peces" to "animales"
content = content.replace("Clasificación de Imágenes de Peces mediante Redes Neuronales", "Clasificación de Imágenes de Animales mediante Redes Neuronales")
content = content.replace("abordar el problema de clasificación multiclase de imágenes de peces: una red", "abordar el problema de clasificación multiclase de imágenes de animales: una red")


# 2. Build the new tables matching the adjustbox format
def make_report(model_name, data):
    rows = []
    for d in data[:10]:
        rows.append(f"{d[0]:<11} & {d[1]} & {d[2]} & {d[3]} & {d[4]} \\\\")
    
    table = fr'''\begin{{table}}[h]
\centering
\begin{{adjustbox}}{{max width=\textwidth}}
\begin{{tabular}}{{lcccc}}
\hline
\textbf{{Clase}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1-Score}} &
\textbf{{Support}} \\
\hline
{rows[0]}
{rows[1]}
{rows[2]}
{rows[3]}
{rows[4]}
{rows[5]}
{rows[6]}
{rows[7]}
{rows[8]}
{rows[9]}
\hline
\textbf{{Macro avg}}    & {data[10][1]} & {data[10][2]} & {data[10][3]} & {data[10][4]} \\
\textbf{{Weighted avg}} & {data[11][1]} & {data[11][2]} & {data[11][3]} & {data[11][4]} \\
\hline
\end{{tabular}}
\end{{adjustbox}}
\caption{{Classification Report del modelo {model_name}.}}
\label{{tab:report_{model_name.lower().replace(" ", "_")}}}
\end{{table}}'''
    return table

data_custom = [
    ("Araña", "0.90", "0.78", "0.83", "483"),
    ("Ardilla", "0.68", "0.73", "0.71", "187"),
    ("Caballo", "0.86", "0.63", "0.73", "263"),
    ("Elefante", "0.41", "0.95", "0.57", "146"),
    ("Gallina", "0.81", "0.83", "0.82", "311"),
    ("Gato", "0.58", "0.61", "0.59", "168"),
    ("Mariposa", "0.73", "0.91", "0.81", "212"),
    ("Oveja", "0.63", "0.77", "0.70", "182"),
    ("Perro", "0.87", "0.48", "0.62", "487"),
    ("Vaca", "0.63", "0.73", "0.67", "188"),
    ("Macro avg", "0.71", "0.74", "0.70", "2627"),
    ("Weighted avg", "0.76", "0.72", "0.72", "2627")
]

data_vgg = [
    ("Araña", "1.00", "0.97", "0.98", "483"),
    ("Ardilla", "0.96", "0.96", "0.96", "187"),
    ("Caballo", "0.95", "0.93", "0.94", "263"),
    ("Elefante", "0.95", "0.99", "0.97", "146"),
    ("Gallina", "0.99", "0.99", "0.99", "311"),
    ("Gato", "0.95", "0.96", "0.96", "168"),
    ("Mariposa", "0.94", "1.00", "0.97", "212"),
    ("Oveja", "0.89", "0.97", "0.93", "182"),
    ("Perro", "0.98", "0.94", "0.96", "487"),
    ("Vaca", "0.90", "0.92", "0.91", "188"),
    ("Macro avg", "0.95", "0.96", "0.96", "2627"),
    ("Weighted avg", "0.96", "0.96", "0.96", "2627")
]

data_mbnet = [
    ("Araña", "1.00", "0.98", "0.99", "483"),
    ("Ardilla", "0.97", "0.94", "0.96", "187"),
    ("Caballo", "0.96", "0.92", "0.94", "263"),
    ("Elefante", "0.93", "0.96", "0.95", "146"),
    ("Gallina", "0.97", "0.99", "0.98", "311"),
    ("Gato", "0.95", "0.97", "0.96", "168"),
    ("Mariposa", "0.97", "0.99", "0.98", "212"),
    ("Oveja", "0.86", "0.93", "0.89", "182"),
    ("Perro", "0.98", "0.94", "0.96", "487"),
    ("Vaca", "0.87", "0.90", "0.88", "188"),
    ("Macro avg", "0.95", "0.95", "0.95", "2627"),
    ("Weighted avg", "0.96", "0.96", "0.96", "2627")
]

rep_custom = make_report("Custom CNN", data_custom)
rep_vgg = make_report("VGG16", data_vgg)
rep_mbnet = make_report("MobileNetV2", data_mbnet)


# Find the current MobileNetV2 table and replace it with ALL THREE tables
pattern = r"\\begin{table}\[h\]\n\\centering\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{lcccc}.*?\\end{table}"

all_tables = f"{rep_custom}\n\n{rep_vgg}\n\n{rep_mbnet}"

content = re.sub(pattern, all_tables.replace('\\', '\\\\'), content, flags=re.DOTALL)


# 3. Replace the final comparativa table '---' values
old_comp = r"""VGG16       & \textbf{0.9612} & 0.96 & 0.96 & 0.96 & 0.96 & 0.96 \\
MobileNetV2 & 0.9558          & 0.95 & 0.95 & 0.95 & 0.96 & 0.96 \\
Custom CNN  & 0.7168          & 0.71 & 0.74 & 0.70 & 0.76 & 0.72 \\"""

if old_comp not in content:
    # If the user reset it to ---
    old_comp = r"""VGG16       & \textbf{0.9612} & --- & --- & --- & --- & --- \\
MobileNetV2 & 0.9558          & --- & --- & --- & --- & --- \\
Custom CNN  & 0.7168          & --- & --- & --- & --- & --- \\"""

new_comp = r"""VGG16       & \textbf{0.9612} & 0.96 & 0.96 & 0.96 & 0.96 & 0.96 \\
MobileNetV2 & 0.9558          & 0.95 & 0.95 & 0.95 & 0.96 & 0.96 \\
Custom CNN  & 0.7168          & 0.71 & 0.74 & 0.70 & 0.76 & 0.72 \\"""

content = content.replace(old_comp, new_comp)

with open('informe_tp3.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("LaTeX content updated successfully.")
