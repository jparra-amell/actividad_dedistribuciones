# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - Paso 1: ANÁLISIS DESCRIPTIVO DE LOS DATOS
Caso: Ejemplo 6.1 (lavado e inspección de piezas A, B y C)
Ejecute celda por celda en Spyder (Ctrl + Enter).
Para analizar otra variable cambie VARIABLE (ej.: "llegadas_A", "inspeccion_C").
"""

#%% 1. Librerías
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Trabajar SIEMPRE en la carpeta donde está guardado este script ---
# (evita el error FileNotFoundError cuando Spyder tiene otra carpeta de trabajo)
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
# Los .csv pueden estar en la subcarpeta datos/ o junto a los scripts
CARPETA_DATOS = "datos/datos" if os.path.isdir("datos/datos") else ("datos" if os.path.isdir("datos") else ".")
print("Carpeta de trabajo:", os.getcwd())
print("Carpeta de datos:", CARPETA_DATOS)

os.makedirs("figuras", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"

#%% 2. Lista de las 9 variables a analizar
variables = [
    "llegadas_A", "llegadas_B", "llegadas_C",
    "lavado_A", "lavado_B", "lavado_C",
    "inspeccion_A", "inspeccion_B", "inspeccion_C"
]

# Diccionario para almacenar todos los resúmenes y exportarlos si deseas
resumenes_totales = {}

for VARIABLE in variables:
    print("\n" + "="*50)
    print(f"ANALIZANDO VARIABLE: {VARIABLE}")
    print("="*50)
    
    ruta_archivo = f"{CARPETA_DATOS}/{VARIABLE}.csv"
    
    try:
        df = pd.read_csv(ruta_archivo)
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo: {ruta_archivo}. Saltando...")
        continue
        
    # Verificar si la columna esperada existe, si no, tomar la primera columna numérica
    col_tiempo = "tiempo_min" if "tiempo_min" in df.columns else df.columns[0]
    x = df[col_tiempo].to_numpy()
    n = len(x)
    
    print(f"Número de datos (n): {n}")

    #%% 3. Medidas de resumen
    q1, mediana, q3 = np.percentile(x, [25, 50, 75])
    resumen = pd.Series({
        "n": n,
        "Media": x.mean(),
        "Mediana": mediana,
        "Desviación estándar": x.std(ddof=1),
        "Coef. de variación (CV)": x.std(ddof=1) / x.mean(),
        "Mínimo": x.min(),
        "Q1 (25 %)": q1,
        "Q3 (75 %)": q3,
        "Máximo": x.max(),
        "Rango": x.max() - x.min(),
        "Asimetría": stats.skew(x, bias=False),
        "Curtosis (exceso)": stats.kurtosis(x, bias=False),
    })
    resumenes_totales[VARIABLE] = resumen
    print(resumen.round(3).to_string())

    #%% 4. Diagrama de caja (boxplot) y datos atípicos
    ric = q3 - q1
    lim_inf = q1 - 1.5 * ric
    lim_sup = q3 + 1.5 * ric
    atipicos = x[(x < lim_inf) | (x > lim_sup)]
    print(f"RIC = {ric:.2f}  |  Límites: [{lim_inf:.2f} ; {lim_sup:.2f}]")
    print(f"Cantidad de atípicos: {len(atipicos)}")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.boxplot(x, vert=False, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#D6E6EA", color=AZUL, linewidth=1.5),
               medianprops=dict(color=NARANJA, linewidth=2.5),
               whiskerprops=dict(color=AZUL, linewidth=1.5),
               capprops=dict(color=AZUL, linewidth=1.5),
               flierprops=dict(marker="o", markerfacecolor=NARANJA, markeredgecolor=NARANJA, markersize=7))
    for valor, texto in [(q1, "Q1"), (mediana, "Mediana"), (q3, "Q3")]:
        ax.annotate(f"{texto}\n{valor:.2f}", (valor, 1.27), ha="center", fontsize=9, color=AZUL)
    ax.axvline(lim_sup, color=GRIS, ls="--", lw=1)
    ax.set_yticks([])
    ax.set_xlabel("Tiempo (min)")
    ax.set_title(f"Diagrama de caja: {VARIABLE}", loc="left", fontsize=12)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figuras/{VARIABLE}_boxplot.png", dpi=200)
    plt.close() # Cierra la figura para liberar memoria

    #%% 5. Histograma (Regla de Sturges)
    k = int(round(1 + 3.322 * np.log10(n)))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(x, bins=k, color=AZUL, edgecolor="white", alpha=0.9)
    ax.axvline(x.mean(), color=NARANJA, lw=2, label=f"Media = {x.mean():.2f}")
    ax.axvline(mediana, color=GRIS, lw=2, ls="--", label=f"Mediana = {mediana:.2f}")
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Histograma: {VARIABLE} (k = {k} clases)", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figuras/{VARIABLE}_histograma.png", dpi=200)
    plt.close()

    #%% 6. Gráfico de serie temporal
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(np.arange(1, n + 1), x, marker="o", ms=3.5, color=AZUL, lw=1)
    ax.axhline(x.mean(), color=NARANJA, lw=1.5, label="Media")
    ax.set_xlabel("Número de observación")
    ax.set_ylabel("Tiempo (min)")
    ax.set_title(f"{VARIABLE} en orden de recolección", loc="left", fontsize=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figuras/{VARIABLE}_serie.png", dpi=200)
    plt.close()

# Opcional: Guardar todas las tablas de resumen juntas en un archivo CSV o Excel
df_resumen_global = pd.DataFrame(resumenes_totales)
df_resumen_global.to_csv("figuras/resumen_todas_las_variables.csv")
print("\n🎉 ¡Análisis completado para todas las variables disponibles!")
print("Los gráficos se guardaron en la carpeta 'figuras/' con el nombre de cada variable.")