with open('c:\\IA-FAC\\informe_tp3.tex', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\\begin{table}[H]', '\\begin{table}[h]')

old_text = r"""Los pesos de las clases se calcularon matemáticamente mediante la heurística de balanceo de scikit-learn (\texttt{compute\_class\_weight}), cuya fórmula es inversamente proporcional a la frecuencia de cada clase. Al pasar este diccionario de pesos a la red neuronal, la función de pérdida penaliza de forma mucho más severa los errores cometidos en clases minoritarias, obligando al modelo a prestarles igual atención durante el descenso de gradiente."""

new_text = r"""Los pesos de las clases se calcularon matemáticamente mediante la heurística de balanceo de scikit-learn (\texttt{compute\_class\_weight}), cuya fórmula es inversamente proporcional a la frecuencia de cada clase:

\begin{equation}
\label{eq:class_weight}
w_j = \frac{N}{K \cdot n_j}
\end{equation}

donde $w_j$ es el peso asignado a la clase $j$, $N$ es el total de muestras del dataset, $K$ es el número total de clases, y $n_j$ es la cantidad de muestras pertenecientes a la clase $j$. Al pasar este diccionario de pesos a la red neuronal, la función de pérdida penaliza de forma mucho más severa los errores cometidos en clases minoritarias, obligando al modelo a prestarles igual atención durante el descenso de gradiente."""

content = content.replace(old_text, new_text)

with open('c:\\IA-FAC\\informe_tp3.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("LaTeX format fixed")
