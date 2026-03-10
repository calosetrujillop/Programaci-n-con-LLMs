import pandas as pd
import numpy as np
import random
from datetime import timedelta


def generar_caso_de_uso_analizar_retencion_clientes():
    """
    Genera un caso de uso aleatorio para la función analizar_retencion_clientes(df).

    Retorna:
        tuple: (
            dict con los argumentos de la función ({"df": DataFrame}),
            DataFrame esperado como resultado de la función
        )
    """

    # --- 1. Generacion aleatoria del DataFrame de entrada ---

    planes = ["Basic", "Pro", "Premium"]
    precios = {"Basic": 9.99, "Pro": 19.99, "Premium": 39.99}

    n_clientes = random.randint(10, 30)

    datos = []
    hoy = pd.Timestamp.today().normalize()

    for i in range(n_clientes):
        cliente_id = f"C{str(i + 1).zfill(3)}"
        plan = random.choice(planes)
        precio_mensual = precios[plan] + round(random.uniform(-1.0, 1.0), 2)

        # Fecha de inicio aleatoria entre 1 y 48 meses atras
        dias_inicio = random.randint(30, 48 * 30)
        fecha_inicio = hoy - timedelta(days=dias_inicio)

        # ~30% de probabilidad de ser cliente activo (sin cancelacion)
        if random.random() < 0.30:
            fecha_cancelacion = np.nan
        else:
            # La cancelacion ocurre entre 1 mes y la fecha actual
            dias_activo = random.randint(30, dias_inicio)
            fecha_cancelacion = fecha_inicio + timedelta(days=dias_activo)
            # Aseguramos que no supere hoy
            if fecha_cancelacion > hoy:
                fecha_cancelacion = hoy

        datos.append({
            "cliente_id": cliente_id,
            "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
            "fecha_cancelacion": (
                fecha_cancelacion.strftime("%Y-%m-%d")
                if not (isinstance(fecha_cancelacion, float) and np.isnan(fecha_cancelacion))
                else np.nan
            ),
            "plan": plan,
            "precio_mensual": round(precio_mensual, 2),
        })

    df_entrada = pd.DataFrame(datos)

    # --- 2. Calculo del resultado esperado (replica la logica de analizar_retencion_clientes) ---

    df_proc = df_entrada.copy()

    # Paso 1: Conversion de fechas
    df_proc["fecha_inicio"] = pd.to_datetime(df_proc["fecha_inicio"])
    df_proc["fecha_cancelacion"] = pd.to_datetime(df_proc["fecha_cancelacion"])

    # Si fecha_cancelacion es NaT, usar la fecha actual
    hoy_ts = pd.Timestamp.today()
    df_proc["fecha_cancelacion"] = df_proc["fecha_cancelacion"].fillna(hoy_ts)

    # Paso 2: Calculo de meses_activo
    df_proc["meses_activo"] = (
        (df_proc["fecha_cancelacion"] - df_proc["fecha_inicio"]) / pd.Timedelta(days=30.44)
    ).round(2)

    # Paso 3: Agrupacion por plan
    resumen = (
        df_proc.groupby("plan")
        .agg(
            clientes_totales=("cliente_id", "count"),
            duracion_promedio_meses=("meses_activo", "mean"),
            ingreso_promedio_mensual=("precio_mensual", "mean"),
        )
        .reset_index()
    )

    resumen["duracion_promedio_meses"] = resumen["duracion_promedio_meses"].round(2)
    resumen["ingreso_promedio_mensual"] = resumen["ingreso_promedio_mensual"].round(2)

    # Paso 4: Ordenar por duracion_promedio_meses de mayor a menor
    resultado_esperado = (
        resumen
        .sort_values("duracion_promedio_meses", ascending=False)
        .reset_index(drop=True)
    )

    # --- 3. Construccion de la tupla de salida ---
    argumentos = {"df": df_entrada}

    return (argumentos, resultado_esperado)
