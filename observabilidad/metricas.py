"""Calculo de metricas de observabilidad a partir de logs/agent_trace.jsonl (IL3.1 e IL3.2).

Se usa desde el dashboard (dashboard/app.py) y desde el script de linea de comandos
(observabilidad/analizar_logs.py). No depende de Streamlit, asi se puede probar por separado.
"""
import json
import pandas as pd


def cargar_eventos(fuente):
    """Lee un archivo .jsonl (ruta o archivo abierto) y devuelve un DataFrame de eventos."""
    filas = []
    abierto = open(fuente, encoding="utf-8") if isinstance(fuente, str) else fuente
    for linea in abierto:
        if isinstance(linea, bytes):
            linea = linea.decode("utf-8")
        linea = linea.strip()
        if linea:
            filas.append(json.loads(linea))
    if isinstance(fuente, str):
        abierto.close()
    ev = pd.DataFrame(filas)
    if len(ev):
        ev["ts"] = pd.to_datetime(ev["ts"])
        for col in ("ok", "bloqueada", "fallback"):
            if col not in ev:
                ev[col] = None
    return ev


def _col(df, nombre, defecto=0):
    return df[nombre].fillna(defecto) if nombre in df else pd.Series([defecto] * len(df), index=df.index)


def llamadas_llm(ev):
    return ev[ev["evento"] == "llm_call"].copy()


def respuestas(ev, incluir_bloqueadas=False):
    r = ev[ev["evento"] == "respuesta"].copy()
    if not incluir_bloqueadas:
        r = r[r["bloqueada"] != True]
    return r


def kpis(ev):
    """Indicadores principales para las tarjetas del dashboard."""
    llm, resp = llamadas_llm(ev), respuestas(ev)
    plan = ev[ev["evento"] == "planificador"]
    total_resp = ev[ev["evento"] == "respuesta"]
    return {
        "solicitudes": int((ev["evento"] == "solicitud").sum()),
        "bloqueadas_seguridad": int((ev["evento"] == "bloqueo_seguridad").sum()),
        "respuestas_con_error": int((total_resp["ok"] == False).sum()),
        "latencia_p50_s": round(float(resp["duracion_s"].quantile(0.5)), 2) if len(resp) else None,
        "latencia_p95_s": round(float(resp["duracion_s"].quantile(0.95)), 2) if len(resp) else None,
        "llamadas_llm": len(llm),
        "tasa_error_llm": round(float((llm["ok"] == False).mean()), 3) if len(llm) else 0.0,
        "tasa_fallback_planificador": round(float(plan["fallback"].fillna(False).astype(bool).mean()), 3) if len(plan) else 0.0,
        "tokens_entrada": int(_col(llm, "tokens_in").sum()),
        "tokens_salida": int(_col(llm, "tokens_out").sum()),
        "costo_estimado_usd": round(float(_col(llm, "costo_usd", 0.0).sum()), 5),
        "memoria_max_mb": round(float(_col(resp, "rss_mb", 0.0).max()), 1) if len(resp) else None,
    }


def latencia_por_etapa(ev):
    """Latencia media / maxima por etapa. La fila de mayor media es el cuello de botella."""
    et = ev[ev["evento"].isin(["etapa", "planificador"])].copy()
    if "etapa" not in et:
        et["etapa"] = None
    et["etapa"] = et["etapa"].fillna("planificador")
    llm = llamadas_llm(ev)
    llm["etapa"] = "llm_call"
    todas = pd.concat([et, llm])
    if not len(todas):
        return pd.DataFrame(columns=["etapa", "n", "media_s", "max_s"])
    t = todas.groupby("etapa")["duracion_s"].agg(n="count", media_s="mean", max_s="max").round(3)
    return t.sort_values("media_s", ascending=False).reset_index()


def latencia_por_estrategia(ev):
    r = respuestas(ev)
    if not len(r):
        return pd.DataFrame(columns=["estrategia", "n", "media_s", "p95_s", "max_s"])
    g = r.groupby("estrategia")["duracion_s"]
    return pd.DataFrame({"n": g.count(), "media_s": g.mean().round(2),
                         "p95_s": g.quantile(0.95).round(2), "max_s": g.max().round(2)}).reset_index()


def errores(ev):
    """Errores de LLM por tipo + fallbacks del planificador + bloqueos de seguridad."""
    llm = llamadas_llm(ev)
    filas = []
    if "error" in llm:
        for tipo, n in llm[llm["ok"] == False]["error"].value_counts().items():
            filas.append({"origen": "llm_call", "tipo": tipo, "n": int(n)})
    plan = ev[ev["evento"] == "planificador"]
    n_fb = int(plan["fallback"].fillna(False).astype(bool).sum()) if len(plan) else 0
    if n_fb:
        filas.append({"origen": "planificador", "tipo": "fallback_json_invalido", "n": n_fb})
    bl = ev[ev["evento"] == "bloqueo_seguridad"]
    if len(bl):
        for motivo, n in bl["motivo"].value_counts().items():
            filas.append({"origen": "seguridad", "tipo": motivo, "n": int(n)})
    return pd.DataFrame(filas, columns=["origen", "tipo", "n"])


def tokens_por_estrategia(ev):
    llm = llamadas_llm(ev)
    if not len(llm):
        return pd.DataFrame(columns=["estrategia", "tokens_entrada", "tokens_salida"])
    llm["estrategia"] = llm["estrategia"].fillna("(planificador)")
    return (llm.groupby("estrategia")
               .agg(tokens_entrada=("tokens_in", "sum"), tokens_salida=("tokens_out", "sum"))
               .reset_index())


def serie_latencia(ev):
    r = respuestas(ev)
    return r[["ts", "duracion_s", "estrategia"]].sort_values("ts") if len(r) else r


def mas_lentas(ev, n=5):
    r = respuestas(ev)
    cols = [c for c in ("ts", "trace_id", "estrategia", "duracion_s") if c in r]
    return r.sort_values("duracion_s", ascending=False)[cols].head(n)


def traza(ev, trace_id):
    """Todos los eventos de un mensaje, en orden: sirve para reconstruir una ejecucion."""
    return ev[ev["trace_id"] == trace_id].sort_values("ts")
