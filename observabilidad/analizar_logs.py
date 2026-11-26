"""Analisis de logs por linea de comandos (IL3.2).

Uso:  python observabilidad/analizar_logs.py logs/agent_trace.jsonl
Imprime las metricas, el cuello de botella y los errores, y escribe logs/hallazgos.md.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metricas as m


def main(ruta):
    ev = m.cargar_eventos(ruta)
    if ev.empty:
        print("El archivo no tiene eventos.")
        return
    k = m.kpis(ev)
    et = m.latencia_por_etapa(ev)
    est = m.latencia_por_estrategia(ev)
    err = m.errores(ev)

    lineas = ["# Hallazgos del analisis de logs", "", f"Archivo analizado: `{ruta}`", "",
              "## Indicadores", ""] + [f"- **{a}**: {b}" for a, b in k.items()]
    lineas += ["", "## Latencia por etapa (la primera fila es el cuello de botella)", "", et.to_markdown(index=False)]
    if len(est):
        lineas += ["", "## Latencia total por estrategia", "", est.to_markdown(index=False)]
    lineas += ["", "## Errores y bloqueos", "", err.to_markdown(index=False) if len(err) else "Sin errores registrados."]
    if len(et):
        lento = et.iloc[0]
        lineas += ["", "## Lectura automatica", "",
                   f"- La etapa mas lenta en promedio es **{lento['etapa']}** ({lento['media_s']} s)."]
    texto = "\n".join(lineas)
    print(texto)
    salida = os.path.join(os.path.dirname(os.path.abspath(ruta)), "hallazgos.md")
    with open(salida, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print("\nReporte escrito en", salida)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/agent_trace.jsonl")
