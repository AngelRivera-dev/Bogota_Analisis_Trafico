import os
import sys
import io
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, desc

# Configuración de entorno
os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["SPARK_HOME"] = r"C:\spark"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PATH"] = os.path.join(os.environ["SPARK_HOME"], "bin") + os.path.pathsep + os.environ["PATH"]
os.environ["TEMP"] = r"C:\SparkTemp"
os.environ["TMP"] = r"C:\SparkTemp"

spark = None

# URL pública al CSV (misma que usa app.py)
URL_GITHUB = "https://raw.githubusercontent.com/Dany601/Datasets901/refs/heads/main/incidentes.csv"

def get_spark_session():
    global spark

    if spark is None:
         spark = SparkSession.builder \
            .appName("AnalisisIncidentesBogota") \
            .master("local[*]") \
            .config("spark.driver.host", "localhost") \
            .config("spark.executor.memory", "1g") \
            .config("spark.driver.memory", "1g") \
            .getOrCreate()

    return spark


def cargar_datos():
    """
    Descarga el CSV desde GitHub y devuelve un pandas.DataFrame.
    Si no es posible, lanza excepción para que el llamador lo gestione.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(URL_GITHUB, headers=headers, timeout=20)
        resp.raise_for_status()
        text = resp.text

        # Intentar leer con separador ';' (como en app.py), y si falla, probar autodetección
        try:
            df = pd.read_csv(io.StringIO(text), delimiter=';', on_bad_lines='skip', low_memory=False)
        except Exception:
            df = pd.read_csv(io.StringIO(text), on_bad_lines='skip', low_memory=False)

        # Normalizar nombres de columnas (strip)
        df.columns = [c.strip() for c in df.columns]

        # Debug: mostrar info básica del dataframe
        try:
            print("LOG spark.py - CSV descargado: filas=", len(df), " columnas=", list(df.columns)[:20])
            print("LOG spark.py - primeras filas:\n", df.head(3).to_dict(orient='records'))
        except Exception:
            pass

        # Normalizar columnas de conteos (Cant*) a numérico
        cant_cols_all = [c for c in df.columns if c.lower().startswith('cant')]
        if cant_cols_all:
            try:
                df[cant_cols_all] = df[cant_cols_all].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            except Exception:
                # en caso de error, forzar a 0 cuando no convertible
                for c in cant_cols_all:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

        return df
    except Exception as e:
        print(f"LOG spark.py - cargar_datos error: {e}")
        raise


def obtener_resultados():
    try:
        pdf = cargar_datos()

        # 1) Incidentes por Localidad: contar filas por localidad
        if 'Localidad' in pdf.columns:
            g_local = pdf.groupby('Localidad').size().reset_index(name='total_incidentes')
            # Si existen columnas de tipo Cant* sumar como 'total_heridos'
            cant_cols = [c for c in pdf.columns if c.lower().startswith('cant') or c.lower().startswith('cant.')]
            if cant_cols:
                g_her = pdf.groupby('Localidad')[cant_cols].sum().reset_index()
                # sumar todas las columnas cant en una sola
                g_her['total_heridos'] = g_her[cant_cols].sum(axis=1)
                g_local = g_local.merge(g_her[['Localidad', 'total_heridos']], on='Localidad', how='left')
                g_local['total_heridos'] = g_local['total_heridos'].fillna(0).astype(int)
            else:
                g_local['total_heridos'] = 0
            g_local = g_local.sort_values('total_incidentes', ascending=False)
            incidentes_localidad = g_local.to_dict(orient='records')
        else:
            incidentes_localidad = []

        # 2) Incidentes por Tipo de Vehículo / implicado: buscar columnas "Tipo implicado N" y sus Cant
        tipo_cols = [c for c in pdf.columns if 'tipo implic' in c.lower() or 'tipo implicado' in c.lower() or 'tipo implic' in c.lower()]
        # Alternativamente nombres como 'Tipo implicado 1' y Cant., Cant..1, etc
        # Encontrar pares (tipo_col, cant_col)
        pairs = []
        # heurística: columnas que contienen 'Tipo' y 'implic' juntos
        for c in pdf.columns:
            low = c.lower()
            if 'tipo implic' in low or 'tipo implicado' in low or 'tipo implic' in low:
                # buscar columna numerada cercana para Cant
                pairs.append((c, None))
        # si no encontramos, buscar "Tipo implicado" usando startswith 'tipo' and 'implic'
        if not pairs:
            for c in pdf.columns:
                if c.lower().startswith('tipo') and 'implic' in c.lower():
                    pairs.append((c, None))

        # ahora buscar columnas cant (que suelen llamarse 'Cant.' 'Cant..1' 'Cant..2' o que empiecen con 'cant')
        cant_cols_all = [c for c in pdf.columns if c.lower().startswith('cant') or c.lower().startswith('cant.')]
        # asignar por orden
        for i in range(len(pairs)):
            if i < len(cant_cols_all):
                pairs[i] = (pairs[i][0], cant_cols_all[i])

        veh_rows = []
        for tipo_col, cant_col in pairs:
            if tipo_col not in pdf.columns:
                continue
            if cant_col and cant_col in pdf.columns:
                tmp = pdf[[tipo_col, cant_col]].dropna(subset=[tipo_col])
                tmp = tmp.rename(columns={tipo_col: 'TipoVehiculo', cant_col: 'count'})
            else:
                tmp = pdf[[tipo_col]].dropna(subset=[tipo_col])
                tmp = tmp.rename(columns={tipo_col: 'TipoVehiculo'})
                tmp['count'] = 1
            # normalizar strings
            tmp['TipoVehiculo'] = tmp['TipoVehiculo'].astype(str).str.strip()
            agg = tmp.groupby('TipoVehiculo')['count'].sum().reset_index().rename(columns={'count': 'total_incidentes'})
            veh_rows.append(agg)

        if veh_rows:
            veh_df = pd.concat(veh_rows, axis=0, ignore_index=True)
            veh_df = veh_df.groupby('TipoVehiculo')['total_incidentes'].sum().reset_index()
            veh_df = veh_df.sort_values('total_incidentes', ascending=False)
            incidentes_vehiculo = veh_df.to_dict(orient='records')
        else:
            incidentes_vehiculo = []

        # 3) Incidentes por clima — buscar columna que contenga 'clima' o 'condicion'
        clima_col = None
        for c in pdf.columns:
            if 'clima' in c.lower() or 'condicion' in c.lower():
                clima_col = c
                break
        if clima_col:
            # usar conteo de filas por condición
            clima_df = pdf.groupby(clima_col).size().reset_index(name='total_incidentes')
            clima_df = clima_df.sort_values('total_incidentes', ascending=False)
            clima_df = clima_df.rename(columns={clima_col: 'CondicionClimatica'})
            incidentes_clima = clima_df.to_dict(orient='records')
        else:
            incidentes_clima = []

        print("LOG spark.py - agregando por Localidad/TipoVehiculo/CondicionClimatica (si existen)")
        return {
            'incidentes_localidad': incidentes_localidad,
            'incidentes_vehiculo': incidentes_vehiculo,
            'incidentes_clima': incidentes_clima
        }
    except Exception as e:
        print(f"LOG spark.py - obtener_resultados error: {e}")
        return {
            'incidentes_localidad': [],
            'incidentes_vehiculo': [],
            'incidentes_clima': []
        }