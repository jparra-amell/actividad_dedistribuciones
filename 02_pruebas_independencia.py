# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - Paso 2: PRUEBAS DE INDEPENDENCIA (AUTOMATIZADO PARA LAS 9 VARIABLES)
"""

#%% 1. Librerías y configuración inicial
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Trabajar SIEMPRE en la carpeta donde está guardado este script ---
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

# Detección automática de la ruta de datos
CARPETA_DATOS = "datos/datos" if os.path.isdir("datos/datos") else ("datos" if os.path.isdir("datos") else ".")
print("Carpeta de trabajo:", os.getcwd())
print("Carpeta de datos:", CARPETA_DATOS)

os.makedirs("figuras", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"
ALFA = 0.05

# Lista de las 9 variables del caso
variables = [
    "llegadas_A", "llegadas_B", "llegadas_C",
    "lavado_A", "lavado_B", "lavado_C",
    "inspeccion_A", "inspeccion_B", "inspeccion_C"
]

resumen_independencia_global = {}

for VARIABLE in variables:
    print("\n" + "="*60)
    print(f"PRUEBAS DE INDEPENDENCIA PARA: {VARIABLE}")
    print("="*60)
    
    ruta_archivo = f"{CARPETA_DATOS}/{VARIABLE}.csv"
    
    try:
        df = pd.read_csv(ruta_archivo)
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo: {ruta_archivo}. Saltando...")
        continue
        
    col_tiempo = "tiempo_min" if "tiempo_min" in df.columns else df.columns[0]
    x = df[col_tiempo].to_numpy()
    n = len(x)
    print(f"Número de datos (n): {n}")

    #%% 2. (a) GRÁFICO DE DISPERSIÓN x_i vs x_i+1
    rng = np.random.default_rng(1)
    dep = np.zeros(n); dep[0] = x.mean()
    for i in range(1, n):
        dep[i] = 0.85 * dep[i - 1] + 0.15 * x.mean() + rng.normal(0, 1.2)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, serie, titulo, color in [(axs[0], x, f"Datos reales ({VARIABLE})", AZUL),
                                     (axs[1], dep, "Ejemplo DEPENDIENTE", NARANJA)]:
        r = np.corrcoef(serie[:-1], serie[1:])[0, 1]
        ax.scatter(serie[:-1], serie[1:], s=22, color=color, alpha=0.8)
        ax.set_title(f"{titulo}\nr = {r:.3f}", loc="left", fontsize=11)
        ax.set_xlabel("$x_i$"); ax.set_ylabel("$x_{i+1}$")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figuras/{VARIABLE}_02_dispersion.png", dpi=200)
    plt.close()

    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    print(f"Correlación entre x_i y x_i+1 (r1): {r1:.3f}")

    #%% 3. (b) ANÁLISIS DE CORRELACIÓN: autocorrelograma
    banda = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    plot_acf(x, lags=20, alpha=ALFA, ax=ax, title="", color=AZUL, vlines_kwargs={"colors": AZUL})
    ax.axhline(banda, color=NARANJA, ls="--", lw=1); ax.axhline(-banda, color=NARANJA, ls="--", lw=1)
    ax.set_title(f"Autocorrelograma: {VARIABLE}", loc="left", fontsize=12)
    ax.set_xlabel("Retardo k"); ax.set_ylabel("$r_k$")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figuras/{VARIABLE}_02_autocorrelograma.png", dpi=200)
    plt.close()

    xc = x - x.mean()
    rk = np.array([np.sum(xc[:-k] * xc[k:]) / np.sum(xc ** 2) for k in range(1, 21)])
    fuera = np.where(np.abs(rk) > banda)[0] + 1
    print(f"Retardos fuera de las bandas (de 20): {list(fuera) if len(fuera) else 'ninguno'}")

    #%% 4. (c-1) PRUEBA DE RACHAS SOBRE Y BAJO LA MEDIANA
    mediana = np.median(x)
    signos = np.where(x > mediana, "+", "-")[x != mediana]
    n1, n2 = np.sum(signos == "+"), np.sum(signos == "-")
    
    if n1 > 0 and n2 > 0:
        rachas = 1 + np.sum(signos[1:] != signos[:-1])
        media_R = 2 * n1 * n2 / (n1 + n2) + 1
        var_R = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z_med = (rachas - media_R) / np.sqrt(var_R)
        p_med = 2 * (1 - stats.norm.cdf(abs(z_med)))
    else:
        z_med, p_med = 0.0, 1.0

    #%% 5. (c-2) PRUEBA DE RACHAS ASCENDENTES Y DESCENDENTES
    d = np.sign(np.diff(x)); d = d[d != 0]
    if len(d) > 0:
        rachas_ad = 1 + np.sum(d[1:] != d[:-1])
        media_ad = (2 * n - 1) / 3
        var_ad = (16 * n - 29) / 90
        z_ad = (rachas_ad - media_ad) / np.sqrt(var_ad)
        p_ad = 2 * (1 - stats.norm.cdf(abs(z_ad)))
    else:
        z_ad, p_ad = 0.0, 1.0

    #%% 6. (c-3) PRUEBA DE LJUNG-BOX
    lags_test = min(10, n - 1) if n > 1 else 1
    lb = acorr_ljungbox(x, lags=[lags_test], return_df=True)
    q_lb, p_lb = lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]

    #%% 7. TABLA RESUMEN Y DECISIÓN POR VARIABLE
    tabla = pd.DataFrame({
        "Prueba": ["Rachas mediana", "Rachas asc/desc", f"Ljung-Box ({lags_test} retardos)"],
        "Estadístico": [f"Z = {z_med:.3f}", f"Z = {z_ad:.3f}", f"Q = {q_lb:.3f}"],
        "Valor p": [p_med, p_ad, p_lb],
    })
    tabla["Decisión (α = 0.05)"] = np.where(tabla["Valor p"] >= ALFA, "No se rechaza H0 (Independientes)", "Se rechaza H0 (Dependientes)")
    print("\n" + tabla.round(4).to_string(index=False))
    
    resumen_independencia_global[VARIABLE] = tabla["Decisión (α = 0.05)"].values

# Opcional: Exportar un resumen global si lo requieres
print("\n🎉 ¡Pruebas de independencia finalizadas para todas las variables!")