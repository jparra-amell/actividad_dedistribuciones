# -*- coding: utf-8 -*-
"""
SIMULACIÓN DE SISTEMAS - CECAR
Modelo del Ejemplo 6.1 (lavado e inspección de piezas A, B y C) en Python (SimPy)

¿Para qué sirve este archivo?
  En clase el modelo se construye en FlexSim. Este script es un EQUIVALENTE en
  Python que permite generar las corridas piloto y las corridas definitivas sin
  FlexSim (para practicar la validación o para comprobar resultados).

Requisito: haber ejecutado 03_bondad_ajuste.py para LAS 9 VARIABLES
           (cada ejecución guarda resultados/ajuste_<variable>.json).
Instalar SimPy la primera vez:  pip install simpy

Sistema: 3 tipos de pieza -> almacén -> 1 lavadora -> almacén -> 2 inspectores -> salida
Horizonte: un turno de 8 horas (480 min) que empieza vacío  (simulación TERMINAL)
Medida de desempeño: número de piezas terminadas en el turno
"""

#%% 1. Librerías
import os
import json
import math
import numpy as np
import pandas as pd
import simpy
from scipy import stats

DURACION_TURNO = 480          # minutos
TIPOS = ["A", "B", "C"]


#%% 2. El modelo: una función que simula UN turno
def simular_turno(dist, semilla, duracion=DURACION_TURNO):
    """dist: diccionario con 9 funciones generadoras, por ejemplo dist["lavado_B"](rng)."""
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    lavadora = simpy.Resource(env, capacity=1)
    inspectores = simpy.Resource(env, capacity=2)
    terminadas = []

    def pieza(tipo):
        llegada = env.now
        with lavadora.request() as turno:                 # espera en el almacén 1
            yield turno
            yield env.timeout(max(0.0, dist[f"lavado_{tipo}"](rng)))
        with inspectores.request() as turno:              # espera en el almacén 2
            yield turno
            yield env.timeout(max(0.0, dist[f"inspeccion_{tipo}"](rng)))
        terminadas.append(env.now - llegada)

    def llegadas(tipo):
        while True:
            yield env.timeout(max(0.0, dist[f"llegadas_{tipo}"](rng)))
            env.process(pieza(tipo))

    for t in TIPOS:
        env.process(llegadas(t))
    env.run(until=duracion)
    return len(terminadas)


def cargar_modelo(carpeta="resultados"):
    """Lee las 9 distribuciones ajustadas en el script 03 y las convierte en generadores."""
    dist = {}
    for etapa in ["llegadas", "lavado", "inspeccion"]:
        for t in TIPOS:
            nombre = f"{etapa}_{t}"
            with open(f"{carpeta}/ajuste_{nombre}.json", encoding="utf-8") as f:
                info = json.load(f)
            d = getattr(stats, info["scipy"])(*info["args"])
            dist[nombre] = (lambda d: (lambda rng: float(d.rvs(random_state=rng))))(d)
            print(f"{nombre:13s} -> {info['distribucion']:11s} {info['parametros']}")
    return dist


#%% 3. Ejecutar el modelo: corridas piloto y corridas definitivas
if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    CARPETA_DATOS = "datos" if os.path.isdir("datos") else "."
    modelo = cargar_modelo()

    # Prueba piloto: 20 corridas (turnos) independientes
    N_PILOTO = 20
    piloto = [simular_turno(modelo, semilla=1000 + i) for i in range(N_PILOTO)]
    pd.DataFrame({"piezas_turno": piloto}).to_csv(f"{CARPETA_DATOS}/validacion_piloto.csv", index=False)
    s0 = np.std(piloto, ddof=1)
    print(f"\nPiloto: media = {np.mean(piloto):.2f} piezas | s = {s0:.3f}")

    # Número de corridas  n = (s · Z / ε)²   (el mismo cálculo del script 04)
    EPSILON = 1.5                              # error aceptado: ± 1.5 piezas por turno
    n = math.ceil((s0 * 1.96 / EPSILON) ** 2)
    corridas = [simular_turno(modelo, semilla=5000 + i) for i in range(n)]
    pd.DataFrame({"piezas_turno": corridas}).to_csv(f"{CARPETA_DATOS}/validacion_corridas.csv", index=False)
    print(f"Corridas definitivas: n = {n} | media = {np.mean(corridas):.2f} piezas")