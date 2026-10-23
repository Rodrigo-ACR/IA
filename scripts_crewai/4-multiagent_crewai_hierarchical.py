"""
IL2.3: Orquestación Multi-Agente JERÁRQUICA con CrewAI
=======================================================
Evolución de 3-multiagent_crewai_full.py que cambia el modelo de
orquestación de SERIAL a JERÁRQUICO.

Diferencias clave respecto al pipeline serial:
───────────────────────────────────────────────
  Serial (anterior)     → las tareas corren en orden fijo predefinido.
  Jerárquico (este)     → un agente Manager decide QUIÉN hace QUÉ y CUÁNDO,
                          delegando tareas a los agentes especializados según
                          el objetivo global. Los agentes NO se coordinan entre
                          sí directamente: todo pasa por el Manager.

Arquitectura:
                      ┌─────────────────┐
                      │  Manager Agent  │  ← orquesta y toma decisiones
                      └────────┬────────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        Investigador        Analista      Comunicador
        (hechos)          (insights)    (redacta/envía)

Capacidades de CrewAI utilizadas:
    ✅  Agent          — rol, goal, backstory, llm, tools, max_iter
    ✅  Task           — description, expected_output, agent (sin context manual)
    ✅  Crew           — manager_llm, Process.hierarchical, verbose
    ✅  BaseTool       — herramienta de clase con estado (registro de emails)
    ✅  Process        — orquestación jerárquica (manager decide el flujo)
    ✅  inputs         — inyección dinámica de variables en las tareas

Nota sobre Process.hierarchical en CrewAI 1.x:
  - Hay que proveer `manager_llm` (o `manager_agent`) al Crew.
  - El Manager interno de CrewAI genera sub-tareas y las delega; por eso NO
    se añade el manager a la lista `agents` del Crew (lo gestiona el framework).
  - El contexto entre tareas lo maneja el Manager, no las dependencias
    explícitas `context=[...]` del modo serial.
  - Se mantiene `max_iter=3` para respetar los límites OTPM de Groq.

Requiere: pip install crewai python-dotenv
Variables de entorno en .env:
    LLM_API_KEY  = tu clave de API
    LLM_BASE_URL = https://api.groq.com/openai/v1
    LLM_MODEL    = qwen/qwen3.8-27b
"""

import os
from datetime import datetime
from typing import Type

from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()   # Lee LLM_BASE_URL / LLM_API_KEY / LLM_MODEL del .env

# ---------------------------------------------------------------------------
# Configuración del LLM
# ---------------------------------------------------------------------------
# Se usan tres instancias con distintos max_tokens:
#   llm_worker  → agentes trabajadores (Investigador, Analista): 600 tokens
#   llm_comm    → Comunicador (recibe contexto acumulado): 950 tokens
#   llm_manager → Manager que coordina todo el equipo: 800 tokens
#
# El free tier de Groq tiene límite de 1000 OTPM para qwen3.8-27b.
# ---------------------------------------------------------------------------
_base = dict(
    model="openai/" + os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3,
)

llm_worker  = LLM(**_base, max_tokens=600)   # investigador y analista
llm_comm    = LLM(**_base, max_tokens=950)   # comunicador (más contexto)
llm_manager = LLM(**_base, max_tokens=800)   # manager jerárquico


# ---------------------------------------------------------------------------
# Herramienta: Simulación de correo electrónico
# Idéntica a la del pipeline serial; se mantiene para comparación didáctica.
# ---------------------------------------------------------------------------

class EmailInput(BaseModel):
    """Esquema de entrada para la herramienta de correo."""
    destinatario: str = Field(description="Dirección de correo del destinatario.")
    asunto:       str = Field(description="Asunto del correo electrónico.")
    cuerpo:       str = Field(description="Contenido completo del correo.")


class SimulatedEmailTool(BaseTool):
    """
    Herramienta que simula el envío de un correo electrónico.

    En un sistema real se conectaría a SMTP, SendGrid o la API de Gmail.
    Aquí registra el envío en un log interno y muestra la simulación en consola.
    """
    name:        str = "enviar_correo_simulado"
    description: str = (
        "Simula el envío de un correo electrónico. "
        "Requiere destinatario, asunto y cuerpo del mensaje. "
        "Devuelve confirmación del envío con timestamp."
    )
    args_schema: Type[BaseModel] = EmailInput

    # Estado interno: registro de todos los correos enviados en la sesión
    sent_emails: list = []

    def _run(self, destinatario: str, asunto: str, cuerpo: str) -> str:
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        self.sent_emails.append({
            "timestamp":    timestamp,
            "destinatario": destinatario,
            "asunto":       asunto,
            "cuerpo":       cuerpo,
        })

        sep = "─" * 60
        print(f"\n{sep}")
        print("📧  SIMULACIÓN DE ENVÍO DE CORREO")
        print(sep)
        print(f"  De:          manager-crew@ia-curso.cl")
        print(f"  Para:        {destinatario}")
        print(f"  Asunto:      {asunto}")
        print(f"  Fecha/Hora:  {timestamp}")
        print(sep)
        print(cuerpo)
        print(f"{sep}\n")

        return (
            f"✅ Correo enviado exitosamente a '{destinatario}' "
            f"con asunto '{asunto}' a las {timestamp}."
        )


# Instancia compartida (permite acceder a sent_emails al final)
email_tool = SimulatedEmailTool()


# ---------------------------------------------------------------------------
# Agentes trabajadores (workers)
# En modo jerárquico el Manager les delega tareas; ellos NO coordinan entre sí.
# ---------------------------------------------------------------------------

investigador = Agent(
    role="Investigador Senior",
    goal=(
        "Recopilar hechos concretos, precisos y bien estructurados sobre el "
        "tema solicitado. Entregar los datos en formato de lista numerada."
    ),
    backstory=(
        "Eres un investigador académico con 15 años de experiencia. "
        "Tu especialidad es sintetizar información compleja en hechos claros "
        "y verificables. No emites opiniones, solo presentas datos."
    ),
    llm=llm_worker,
    max_iter=3,
    verbose=True,
)

analista = Agent(
    role="Analista de Datos",
    goal=(
        "Identificar patrones, tendencias y conclusiones prácticas a partir "
        "de los hechos entregados por el investigador."
    ),
    backstory=(
        "Eres un analista estratégico con experiencia en transformar datos "
        "crudos en insights accionables. Estructuras tu análisis en tres secciones: "
        "Patrones identificados, Puntos clave y Conclusiones prácticas."
    ),
    llm=llm_worker,
    max_iter=3,
    verbose=True,
)

comunicador = Agent(
    role="Comunicador Ejecutivo",
    goal=(
        "Redactar un correo profesional con el análisis generado y enviarlo "
        "al destinatario indicado usando la herramienta de correo disponible."
    ),
    backstory=(
        "Eres un comunicador corporativo experto en transformar análisis "
        "técnicos en mensajes ejecutivos claros y accionables. "
        "Siempre usas un tono profesional y conciso."
    ),
    llm=llm_comm,
    tools=[email_tool],
    max_iter=3,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Tareas
# ---------------------------------------------------------------------------
# En modo jerárquico NO se usan dependencias explícitas context=[...].
# El Manager decide el orden de ejecución y pasa el contexto acumulado.
# Las variables {topic}, {fecha_actual}, {recipient_email} se inyectan
# vía crew.kickoff(inputs=...) igual que en el modo serial.
# ---------------------------------------------------------------------------

tarea_investigar = Task(
    description=(
        "Fecha de hoy: {fecha_actual}.\n"
        "Investiga el siguiente tema y recopila todos los hechos relevantes: "
        "{topic}\n\n"
        "Incluye la fecha en el encabezado del reporte. "
        "Entrega los datos en formato de lista numerada con al menos 5 hechos."
    ),
    expected_output=(
        "Fecha, título del tema y lista de 5 a 8 hechos concretos y breves "
        "(máximo 2 líneas por hecho)."
    ),
    agent=investigador,
)

tarea_analizar = Task(
    description=(
        "Analiza los datos recopilados sobre '{topic}'. "
        "Identifica patrones, tendencias y genera conclusiones prácticas. "
        "Estructura el análisis en tres secciones: "
        "1) Patrones identificados, 2) Puntos clave, 3) Conclusiones y recomendaciones."
    ),
    expected_output=(
        "Análisis en tres secciones (Patrones, Puntos clave, Conclusiones). "
        "Máximo 3 puntos por sección, cada uno en 1-2 líneas."
    ),
    agent=analista,
)

tarea_enviar_email = Task(
    description=(
        "Redacta un correo profesional con el análisis generado sobre '{topic}' "
        "y envíalo a '{recipient_email}' usando la herramienta de correo. "
        "El correo debe incluir: saludo, resumen ejecutivo del análisis (3-4 párrafos) "
        "y cierre profesional. "
        "Usa como asunto: 'Análisis: {topic} — {fecha_actual}'."
    ),
    expected_output=(
        "Confirmación del envío del correo con destinatario, asunto y timestamp."
    ),
    agent=comunicador,
)


# ---------------------------------------------------------------------------
# Crew JERÁRQUICO
# ---------------------------------------------------------------------------
# Diferencias clave con el Crew serial:
#   process=Process.hierarchical   → activa el modo jerárquico
#   manager_llm=llm_manager        → LLM que usa el Manager interno de CrewAI
#                                    para planificar y delegar tareas
#
# El Manager:
#   1. Recibe el objetivo global (las tareas definidas arriba)
#   2. Decide a qué agente delegar cada sub-tarea y en qué orden
#   3. Revisa los resultados parciales y puede pedir re-trabajo
#   4. Consolida el resultado final
#
# NO se añade el manager a la lista `agents` — el framework lo crea internamente.
# ---------------------------------------------------------------------------

crew = Crew(
    agents=[investigador, analista, comunicador],
    tasks=[tarea_investigar, tarea_analizar, tarea_enviar_email],
    process=Process.hierarchical,
    manager_llm=llm_manager,           # ← LLM del Manager jerárquico
    verbose=True,
)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inputs = {
        "topic":           "tendencias en inteligencia artificial 2025",
        "recipient_email": "director@duoc.cl",
        "fecha_actual":    datetime.now().strftime("%d/%m/%Y"),
    }

    print(f"\n{'='*65}")
    print(f"🏛️   Pipeline CrewAI — ORQUESTACIÓN JERÁRQUICA")
    print(f"{'='*65}")
    print(f"  Modo:         Process.hierarchical (Manager delega y revisa)")
    print(f"  Agentes:      Investigador → Analista → Comunicador")
    print(f"  Manager LLM:  {os.getenv('LLM_MODEL', 'qwen/qwen3.8-27b')}")
    print(f"  Tema:         {inputs['topic']}")
    print(f"  Destinatario: {inputs['recipient_email']}")
    print(f"  Fecha:        {inputs['fecha_actual']}")
    print(f"{'='*65}")
    print()
    print("  🔑  Diferencia con el pipeline serial (3-multiagent_crewai_full.py):")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │  Serial:    A → B → C  (orden fijo, context manual)    │")
    print("  │  Jerárquico: Manager decide quién, cuándo y si rehacer │")
    print("  └─────────────────────────────────────────────────────────┘")
    print()

    result = crew.kickoff(inputs=inputs)

    print(f"\n{'='*65}")
    print("✅  RESULTADO FINAL DEL CREW JERÁRQUICO")
    print(f"{'='*65}")
    print(result)

    # Log de correos enviados
    if email_tool.sent_emails:
        print(f"\n📬  Log de correos enviados esta sesión: {len(email_tool.sent_emails)}")
        for i, e in enumerate(email_tool.sent_emails, 1):
            print(f"  [{i}] {e['timestamp']} → {e['destinatario']} | {e['asunto']}")
