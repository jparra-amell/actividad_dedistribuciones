# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Fase 3 - VALIDACIÓN DEL MODELO (se aplica SOLO al modelo base)
Caso: Ejemplo 6.1. Simulación TERMINAL de un turno de 8 h.
Medida de desempeño: piezas terminadas por turno.
"""

#%% 1. Librerías y datos
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# --- Trabajar SIEMPRE en la carpeta donde está guardado este script ---
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

# Detección automática de la ruta de datos (maneja 'datos/datos/', 'datos/' o directorio actual)
CARPETA_DATOS = "datos/datos" if os.path.isdir("datos/datos") else ("datos" if os.path.isdir("datos") else ".")
print("Carpeta de trabajo:", os.getcwd())
print("Carpeta de datos:", CARPETA_DATOS)

os.makedirs("figuras", exist_ok=True)
AZUL, NARANJA, GRIS = "#0F4C5C", "#E36414", "#6B7B83"
ALFA = 0.05

try:
    real = pd.read_csv(f"{CARPETA_DATOS}/validacion_real.csv")["piezas_turno"].to_numpy()
    piloto = pd.read_csv(f"{CARPETA_DATOS}/validacion_piloto.csv")["piezas_turno"].to_numpy()
    simulado = pd.read_csv(f"{CARPETA_DATOS}/validacion_corridas.csv")["piezas_turno"].to_numpy()
except FileNotFoundError as e:
    print(f"⚠️ Error al cargar los archivos de validación: {e}")
    print("Verifica que existan 'validacion_real.csv', 'validacion_piloto.csv' y 'validacion_corridas.csv' en la carpeta de datos.")

#%% 2. PASO 1 - Prueba piloto
n0 = len(piloto)
s0 = piloto.std(ddof=1)
print(f"\nCorridas piloto n0 = {n0} | media = {piloto.mean():.3f} | s = {s0:.3f}")

#%% 3. PASO 2 - Número de corridas n = (s·Z/ε)²
EPSILON = 1.5                        # error máximo aceptado: ± 1.5 piezas por turno
z = stats.norm.ppf(1 - ALFA / 2)  # Z_{α/2} = 1.96 para 95 % de confianza
n_z = math.ceil((s0 * z / EPSILON) ** 2)          # siempre se redondea HACIA ARRIBA
print(f"Z = {z:.3f} | ε = {EPSILON} | n calculado = {n_z}")

if len(simulado) < n_z:
    print(f"⚠️ ¡ATENCIÓN! Solo hay {len(simulado)} corridas simuladas; se necesitan al menos {n_z}.")
else:
    print(f"✅ El número de corridas simuladas ({len(simulado)}) cumple con el mínimo requerido ({n_z}).")

#%% 4. Estadísticos de las dos muestras
resumen = pd.DataFrame({
    "Real": [len(real), real.mean(), real.var(ddof=1), real.std(ddof=1)],
    "Simulado": [len(simulado), simulado.mean(), simulado.var(ddof=1), simulado.std(ddof=1)],
}, index=["n", "Media", "Varianza", "Desv. estándar"])
print("\n--- ESTADÍSTICOS DESCRIPTIVOS ---")
print(resumen.round(4))

#%% 5. PASO 4a - Prueba F de igualdad de varianzas
v_sim, v_real = simulado.var(ddof=1), real.var(ddof=1)
F = v_sim / v_real
gl1, gl2 = len(simulado) - 1, len(real) - 1
p_F = 2 * min(stats.f.cdf(F, gl1, gl2), 1 - stats.f.cdf(F, gl1, gl2))  # dos colas
var_iguales = p_F >= ALFA
print(f"\n--- PRUEBA F (Varianzas) ---")
print(f"F = {F:.4f} | gl = ({gl1}, {gl2}) | valor p = {p_F:.4f}")
print("-> Varianzas IGUALES" if var_iguales else "-> Varianzas DIFERENTES")

#%% 6. PASO 4b - Prueba t de diferencia de medias
res_t = stats.ttest_ind(simulado, real, equal_var=var_iguales)
tipo = "varianzas iguales" if var_iguales else "varianzas desiguales (Welch)"
t_crit = stats.t.ppf(1 - ALFA / 2, res_t.df)
print(f"\n--- PRUEBA t DE MEDIAS ({tipo}) ---")
print(f"t calculado = {res_t.statistic:.4f} | gl = {res_t.df:.1f} | t crítico = ±{t_crit:.4f} | valor p = {res_t.pvalue:.4f}")

ic = res_t.confidence_interval(confidence_level=1 - ALFA)
print(f"IC 95 % para μ(modelo) - μ(real): [{ic.low:.3f} ; {ic.high:.3f}]")

#%% 7. PASO 4c - Prueba z (como en Excel)
dif = simulado.mean() - real.mean()
z_cal = dif / math.sqrt(v_sim / len(simulado) + v_real / len(real))
p_z = 2 * (1 - stats.norm.cdf(abs(z_cal)))
print(f"\n--- PRUEBA Z ---")
print(f"z calculado = {z_cal:.4f} | z crítico = ±{z:.3f} | valor p = {p_z:.4f}")

#%% 8. DECISIÓN FINAL
print("\n" + "="*50)
if res_t.pvalue >= ALFA:
    print("✅ NO se rechaza H0: el modelo REPRESENTA adecuadamente al sistema real -> MODELO VÁLIDO")
else:
    print("❌ Se rechaza H0: el modelo NO representa al sistema real -> revisar supuestos y datos")
print("="*50)

#%% 9. Gráfico comparativo real vs simulado
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
bp = axs[0].boxplot([real, simulado], widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
for caja, color in zip(bp["boxes"], [AZUL, NARANJA]):
    caja.set_facecolor(color); caja.set_alpha(0.75)
axs[0].set_xticks([1, 2], ["Real", "Simulado"])
axs[0].set_ylabel("Piezas terminadas por turno")
axs[0].set_title("Distribución de los resultados", loc="left")

medias = [real.mean(), simulado.mean()]
err = [stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m)) for m in (real, simulado)]
axs[1].errorbar([0, 1], medias, yerr=err, fmt="o", ms=9, capsize=8, color=GRIS, ecolor=GRIS)
axs[1].scatter([0, 1], medias, s=90, color=[AZUL, NARANJA], zorder=3)
axs[1].set_xticks([0, 1], ["Real", "Simulado"]); axs[1].set_xlim(-0.6, 1.6)
axs[1].set_title("Medias con IC 95 %", loc="left")
for i, m in enumerate(medias):
    axs[1].annotate(f"{m:.2f}", (i + 0.12, m), va="center")
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figuras/04_validacion.png", dpi=200)
plt.show()