# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - Paso 3: SELECCIÓN Y EVALUACIÓN DE LA DISTRIBUCIÓN (bondad de ajuste)
Replica la lógica del "Auto::Fit" de Stat::Fit:
  1. Se ajustan varias distribuciones candidatas (máxima verosimilitud).
  2. A cada una se le aplican las pruebas Chi-cuadrado, Kolmogorov-Smirnov
     y Anderson-Darling.
  3. Se construye un RANKING (100 = mejor ajuste) y se marca cuáles se
     aceptan ("No se rechaza") y cuáles se rechazan.
  4. Se grafican SOLO las distribuciones que no se rechazan.
  5. Se guarda el resultado en resultados/ajuste_<VARIABLE>.json
Caso: Ejemplo 6.1. Para analizar otra variable cambie VARIABLE.
"""

#%% 1. Librerías, datos y parámetros
import os
import json
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
CARPETA_DATOS = "datos" if os.path.isdir("datos") else "."
print("Carpeta de trabajo:", os.getcwd())

os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"
COLORES = [NARANJA, AZUL, "#5B8E7D", "#9A348E", "#B08D57", GRIS]
ALFA = 0.05

VARIABLE = "lavado_B"                  # <- archivo de la carpeta datos/ (sin .csv)
x = np.sort(pd.read_csv(f"{CARPETA_DATOS}/{VARIABLE}.csv")["tiempo_min"].to_numpy())
n = len(x)
COTA_INF = 0.0          # cota inferior: un tiempo no puede ser negativo
DELTA = (x.max() - x.min()) / (n - 1)   # pequeño margen para las distribuciones acotadas

#%% 2. Ajuste de las distribuciones candidatas
# Cada función devuelve: (distribución de scipy "congelada", texto de parámetros, nº de parámetros estimados)
def ajustar(nombre):
    if nombre == "Normal":
        m, s = stats.norm.fit(x)
        return stats.norm(m, s), f"μ={m:.3f}, σ={s:.3f}", 2
    if nombre == "Exponencial":
        loc, esc = stats.expon.fit(x, floc=COTA_INF)
        return stats.expon(loc, esc), f"media={esc:.3f}", 1
    if nombre == "Gamma":
        a, loc, esc = stats.gamma.fit(x, floc=COTA_INF)
        return stats.gamma(a, loc, esc), f"α={a:.3f}, β={esc:.3f}", 2
    if nombre == "Erlang":
        a, loc, esc = stats.gamma.fit(x, floc=COTA_INF)
        k = max(1, int(round(a)))                    # la forma debe ser ENTERA
        esc = (x.mean() - COTA_INF) / k
        return stats.gamma(k, COTA_INF, esc), f"k={k}, β={esc:.3f}", 2
    if nombre == "Lognormal":
        s, loc, esc = stats.lognorm.fit(x, floc=COTA_INF)
        return stats.lognorm(s, loc, esc), f"μ={np.log(esc):.3f}, σ={s:.3f}", 2
    if nombre == "Weibull":
        c, loc, esc = stats.weibull_min.fit(x, floc=COTA_INF)
        return stats.weibull_min(c, loc, esc), f"forma={c:.3f}, escala={esc:.3f}", 2
    if nombre == "Beta":
        a_, b_ = COTA_INF, x.max() + 0.05 * (x.max() - x.min())
        p, q, loc, esc = stats.beta.fit(x, floc=a_, fscale=b_ - a_)
        return stats.beta(p, q, loc, esc), f"p={p:.3f}, q={q:.3f}, [{a_:.2f}; {b_:.2f}]", 4
    if nombre == "Triangular":
        a_, b_ = max(COTA_INF, x.min() - DELTA), x.max() + DELTA
        c, loc, esc = stats.triang.fit(x, floc=a_, fscale=b_ - a_)   # se estima la moda
        moda = a_ + c * (b_ - a_)
        return stats.triang(c, a_, b_ - a_), f"mín={a_:.2f}, moda={moda:.2f}, máx={b_:.2f}", 3
    if nombre == "Uniforme":
        a_, b_ = max(COTA_INF, x.min() - DELTA), x.max() + DELTA
        return stats.uniform(a_, b_ - a_), f"mín={a_:.2f}, máx={b_:.2f}", 2
    if nombre == "Logística":
        m, s = stats.logistic.fit(x)
        return stats.logistic(m, s), f"μ={m:.3f}, s={s:.3f}", 2

candidatas = ["Normal", "Exponencial", "Gamma", "Erlang", "Lognormal",
              "Weibull", "Beta", "Triangular", "Uniforme", "Logística"]

#%% 3. Las tres pruebas de bondad de ajuste
def chi_cuadrado(dist, n_par):
    """Chi-cuadrado con clases EQUIPROBABLES (todas con la misma frecuencia esperada)."""
    k = int(round(1 + 3.322 * np.log10(n)))           # nº de clases (Sturges)
    limites = dist.ppf(np.linspace(0, 1, k + 1))       # límites de clase según la distribución
    limites[0], limites[-1] = -np.inf, np.inf
    obs, _ = np.histogram(x, bins=limites)
    esp = np.full(k, n / k)
    chi2 = np.sum((obs - esp) ** 2 / esp)
    gl = k - 1 - n_par                                  # grados de libertad
    p = 1 - stats.chi2.cdf(chi2, gl) if gl > 0 else np.nan
    return chi2, gl, p

def kolmogorov_smirnov(dist):
    res = stats.kstest(x, dist.cdf)
    return res.statistic, res.pvalue

def anderson_darling(dist):
    F = np.clip(dist.cdf(x), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n

#%% 4. Tabla de resultados y RANKING
filas, modelos = [], {}
for nombre in candidatas:
    dist, txt, n_par = ajustar(nombre)
    modelos[nombre] = dist
    chi2, gl, p_chi = chi_cuadrado(dist, n_par)
    ks, p_ks = kolmogorov_smirnov(dist)
    filas.append([nombre, txt, chi2, gl, p_chi, ks, p_ks, anderson_darling(dist)])

tabla = pd.DataFrame(filas, columns=["Distribución", "Parámetros", "Chi²", "gl",
                                     "p Chi²", "K-S", "p K-S", "A-D"])

# Puntaje = promedio de los valores p de Chi² y K-S (mientras más alto, mejor ajuste)
tabla["Puntaje"] = tabla[["p Chi²", "p K-S"]].mean(axis=1)
# Ranking: la mejor distribución recibe 100 y las demás se escalan respecto a ella
tabla["Ranking"] = 100 * tabla["Puntaje"] / tabla["Puntaje"].max()
# Aceptación: se rechaza si CUALQUIERA de las dos pruebas tiene valor p < alfa
acepta = (tabla["p Chi²"] >= ALFA) & (tabla["p K-S"] >= ALFA)
tabla["Aceptación"] = np.where(acepta, "No se rechaza", "Rechazada")
tabla = tabla.sort_values("Ranking", ascending=False).reset_index(drop=True)

pd.set_option("display.width", 200)
print(tabla[["Distribución", "Parámetros", "p Chi²", "p K-S", "A-D", "Ranking", "Aceptación"]]
      .round({"p Chi²": 4, "p K-S": 4, "A-D": 3, "Ranking": 2}).to_string(index=False))
tabla.to_csv(f"resultados/ranking_{VARIABLE}.csv", index=False)

aceptadas = tabla.loc[tabla["Aceptación"] == "No se rechaza", "Distribución"].tolist()
graficar = aceptadas[:4]          # para que los gráficos se lean bien: las 4 mejores aceptadas
# La mejor es la primera del ranking que NO se rechaza
if aceptadas:
    mejor = aceptadas[0]
else:
    mejor = tabla.loc[0, "Distribución"]
    print("¡ATENCIÓN! Ninguna distribución pasó las pruebas: revise los datos o use una empírica.")
fila_mejor = tabla.index[tabla["Distribución"] == mejor][0]
print("\nNo se rechazan:", aceptadas)
print("Mejor ajuste:", mejor)

#%% 5. Gráfico 1: histograma con las densidades de las distribuciones aceptadas
k = int(round(1 + 3.322 * np.log10(n)))
malla = np.linspace(max(COTA_INF, x.min() - 1), x.max() + 1, 400)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.hist(x, bins=k, density=True, color="#D6E6EA", edgecolor="white", label="Datos")
for nombre, color in zip(graficar, COLORES):
    ax.plot(malla, modelos[nombre].pdf(malla), color=color, lw=2.2, label=nombre)
ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Densidad")
ax.set_title(f"{VARIABLE}: histograma y las mejores distribuciones aceptadas", loc="left", fontsize=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figuras/03_densidades.png", dpi=200); plt.show()

#%% 6. Gráfico 2: función acumulada empírica vs teóricas
Fn = np.arange(1, n + 1) / n
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.step(x, Fn, where="post", color="black", lw=1.5, label="Empírica (datos)")
for nombre, color in zip(graficar, COLORES):
    ax.plot(malla, modelos[nombre].cdf(malla), color=color, lw=2, label=nombre)
ax.set_xlabel("Tiempo (min)"); ax.set_ylabel("Probabilidad acumulada F(x)")
ax.set_title(f"{VARIABLE}: acumulada de los datos vs. modelos aceptados", loc="left", fontsize=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figuras/03_acumuladas.png", dpi=200); plt.show()

#%% 7. Gráfico 3: P-P y Q-Q de la MEJOR distribución
dist = modelos[mejor]
p_emp = (np.arange(1, n + 1) - 0.5) / n
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
axs[0].scatter(p_emp, dist.cdf(x), s=16, color=AZUL)
axs[0].plot([0, 1], [0, 1], color=NARANJA, lw=1.5)
axs[0].set_title(f"Gráfico P-P ({mejor})", loc="left"); axs[0].set_xlabel("Probabilidad empírica"); axs[0].set_ylabel("Probabilidad teórica")
q_teo = dist.ppf(p_emp)
axs[1].scatter(q_teo, x, s=16, color=AZUL)
lim = [min(q_teo.min(), x.min()), max(q_teo.max(), x.max())]
axs[1].plot(lim, lim, color=NARANJA, lw=1.5)
axs[1].set_title(f"Gráfico Q-Q ({mejor})", loc="left"); axs[1].set_xlabel("Cuantil teórico (min)"); axs[1].set_ylabel("Dato observado (min)")
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figuras/03_pp_qq.png", dpi=200); plt.show()

#%% 8. Expresión para FlexSim y archivo de resultados
def expresion_flexsim(nombre, d):
    """Texto para pegar en FlexSim. Verifique el orden de los parámetros en la ayuda de FlexSim."""
    a = d.args
    if nombre == "Normal":      return f"normal({a[0]:.4f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Exponencial": return f"exponential({a[0]:.2f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Gamma":       return f"gamma({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Erlang":      return f"erlang({a[1]:.2f}, {a[2]:.4f}, {int(a[0])}, getstream(current))"
    if nombre == "Lognormal":   return f"lognormal2({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Weibull":     return f"weibull({a[1]:.2f}, {a[2]:.4f}, {a[0]:.4f}, getstream(current))"
    if nombre == "Beta":        return f"beta({a[2]:.2f}, {a[2]+a[3]:.2f}, {a[0]:.4f}, {a[1]:.4f}, getstream(current))"
    if nombre == "Triangular":  return f"triangular({a[1]:.2f}, {a[1]+a[2]:.2f}, {a[1]+a[0]*a[2]:.2f}, getstream(current))"
    if nombre == "Uniforme":    return f"uniform({a[0]:.2f}, {a[0]+a[1]:.2f}, getstream(current))"
    if nombre == "Logística":   return f"logistic({a[0]:.4f}, {a[1]:.4f}, getstream(current))"

d = modelos[mejor]
expr = expresion_flexsim(mejor, d)
print("Distribución seleccionada:", mejor, "->", tabla.loc[fila_mejor, "Parámetros"])
print("Expresión en FlexSim:", expr)

# Se guarda el resultado (lo usa el modelo en Python modelo_ej61_simpy.py)
with open(f"resultados/ajuste_{VARIABLE}.json", "w", encoding="utf-8") as f:
    json.dump({"variable": VARIABLE, "distribucion": mejor, "parametros": tabla.loc[fila_mejor, "Parámetros"],
               "scipy": d.dist.name, "args": [float(v) for v in d.args], "flexsim": expr},
              f, ensure_ascii=False, indent=2)
