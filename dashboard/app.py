"""Dashboard de monitoreo del agente (IL3.1, IL3.2). Ejecutar con:

    streamlit run dashboard/app.py

Lee logs/agent_trace.jsonl (generado por el notebook) y, si existe, logs/evaluacion_planificador.csv.
Tambien se pueden subir los archivos desde la barra lateral.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "observabilidad"))
import metricas as m

st.set_page_config(page_title="Monitoreo del Asistente de Investigacion", layout="wide")
st.title("Monitoreo del Asistente de Investigacion Academica")
st.caption("Observabilidad y trazabilidad del agente: latencia, errores, tokens y precision del planificador.")

# -- Carga de datos ----------------------------------------------------------------
st.sidebar.header("Datos")
subido = st.sidebar.file_uploader("agent_trace.jsonl", type=["jsonl"])
subido_eval = st.sidebar.file_uploader("evaluacion_planificador.csv", type=["csv"])
RUTA_LOG = os.path.join("logs", "agent_trace.jsonl")
RUTA_EVAL = os.path.join("logs", "evaluacion_planificador.csv")

if subido is not None:
    ev = m.cargar_eventos(subido)
elif os.path.exists(RUTA_LOG):
    ev = m.cargar_eventos(RUTA_LOG)
else:
    st.warning("No se encontro logs/agent_trace.jsonl. Ejecuta el notebook y descarga los logs, o subelos en la barra lateral.")
    st.stop()

if ev.empty:
    st.warning("El archivo de logs no tiene eventos.")
    st.stop()

df_eval = None
if subido_eval is not None:
    df_eval = pd.read_csv(subido_eval)
elif os.path.exists(RUTA_EVAL):
    df_eval = pd.read_csv(RUTA_EVAL)

# -- Filtro por estrategia -----------------------------------------------------------
estrategias = sorted(e for e in ev["estrategia"].dropna().unique())
elegidas = st.sidebar.multiselect("Estrategias", estrategias, default=estrategias)
if elegidas:
    trazas_ok = set(ev[ev["estrategia"].isin(elegidas)]["trace_id"])
    ev = ev[ev["trace_id"].isin(trazas_ok) | (ev["trace_id"] == "-")]

k = m.kpis(ev)

# -- Indicadores ---------------------------------------------------------------------
c = st.columns(5)
c[0].metric("Solicitudes", k["solicitudes"])
c[1].metric("Latencia p50 (s)", k["latencia_p50_s"])
c[2].metric("Latencia p95 (s)", k["latencia_p95_s"])
c[3].metric("Error en llamadas LLM", f"{k['tasa_error_llm']:.1%}")
c[4].metric("Fallback planificador", f"{k['tasa_fallback_planificador']:.1%}")
c = st.columns(5)
c[0].metric("Bloqueadas por seguridad", k["bloqueadas_seguridad"])
c[1].metric("Tokens de entrada", f"{k['tokens_entrada']:,}")
c[2].metric("Tokens de salida", f"{k['tokens_salida']:,}")
c[3].metric("Costo estimado (US$)", k["costo_estimado_usd"])
c[4].metric("Memoria maxima (MB)", k["memoria_max_mb"])

if df_eval is not None and len(df_eval):
    st.subheader("Calidad del planificador")
    por_caso = df_eval.groupby("texto")["elegida"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    c = st.columns(3)
    c[0].metric("Precision", f"{df_eval['acierto'].mean():.1%}")
    c[1].metric("Consistencia", f"{por_caso.mean():.1%}")
    c[2].metric("Casos x repeticiones", f"{df_eval['texto'].nunique()} x {df_eval['repeticion'].max()}")
    izq, der = st.columns(2)
    izq.caption("Precision por categoria de entrada")
    izq.bar_chart(df_eval.groupby("categoria")["acierto"].mean())
    der.caption("Matriz de confusion (esperada vs elegida)")
    der.dataframe(pd.crosstab(df_eval["esperada"], df_eval["elegida"]), use_container_width=True)
    malos = df_eval[~df_eval["acierto"]][["texto", "esperada", "elegida", "fallback"]].drop_duplicates()
    if len(malos):
        st.caption("Entradas donde el planificador se equivoco al menos una vez")
        st.dataframe(malos, use_container_width=True)

# -- Latencia --------------------------------------------------------------------------
st.subheader("Latencia")
izq, der = st.columns(2)
et = m.latencia_por_etapa(ev)
izq.caption("Latencia media por etapa (s): la barra mas larga es el cuello de botella")
izq.bar_chart(et.set_index("etapa")["media_s"])
est = m.latencia_por_estrategia(ev)
der.caption("Latencia total por estrategia (s)")
if len(est):
    der.bar_chart(est.set_index("estrategia")[["media_s", "p95_s"]])
else:
    der.info("Aun no hay respuestas completas registradas.")

serie = m.serie_latencia(ev)
if len(serie):
    st.caption("Latencia de cada respuesta en el tiempo (s)")
    st.line_chart(serie.set_index("ts")["duracion_s"])
    st.caption("Las 5 respuestas mas lentas")
    st.dataframe(m.mas_lentas(ev), use_container_width=True)

# -- Errores y tokens ---------------------------------------------------------------------
st.subheader("Errores y consumo de recursos")
izq, der = st.columns(2)
err = m.errores(ev)
izq.caption("Errores, fallbacks y bloqueos de seguridad")
izq.dataframe(err, use_container_width=True) if len(err) else izq.success("Sin errores registrados.")
tok = m.tokens_por_estrategia(ev)
der.caption("Tokens por estrategia")
if len(tok):
    der.bar_chart(tok.set_index("estrategia"))

# -- Trazabilidad -------------------------------------------------------------------------------
st.subheader("Trazabilidad de una ejecucion")
ids = list(ev[ev["evento"] == "solicitud"]["trace_id"])
if ids:
    elegido = st.selectbox("trace_id", ids[::-1])
    cols = [c for c in ("ts", "evento", "etapa", "estrategia", "estrategia_elegida", "duracion_s",
                        "tokens_in", "tokens_out", "ok", "error", "motivo") if c in ev]
    st.dataframe(m.traza(ev, elegido)[cols], use_container_width=True)
    st.caption("Cada fila es un evento del mismo mensaje: solicitud, planificador, etapas, llamadas al LLM y respuesta.")
