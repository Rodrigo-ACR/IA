"""
IL2.3: Orquestación Multi-Agente Completa con CrewAI
=====================================================
Evolución de 2-multiagent_orchestration_llm.py usando todas las capacidades
de CrewAI 1.x: agentes especializados, herramientas personalizadas, tareas
con dependencias explícitas (context), proceso secuencial y un tercer agente
que simula el envío del análisis por correo electrónico.

Pipeline:
    Investigador  →  Analista  →  Comunicador (simula envío por email)

Capacidades de CrewAI utilizadas:
    ✅  Agent          — rol, goal, backstory, llm, tools, max_iter
    ✅  Task           — description, expected_output, agent, context
    ✅  Crew           — agents, tasks, Process.sequential, verbose
    ✅  BaseTool       — herramienta de clase con estado (registro de emails)
    ✅  Process        — orquestación secuencial explícita
    ✅  inputs         — inyección dinámica de variables en las tareas

Nota: Herramientas sin parámetros (p.ej. "obtener fecha") generan un JSON
schema inválido al pasar por la cadena CrewAI → LangChain → LiteLLM → API.
La práctica correcta es inyectar esos valores como variables en `inputs`.

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
# CrewAI usa LiteLLM internamente. El prefijo "openai/" activa el protocolo
# compatible con OpenAI para cualquier proveedor (Groq, Mistral, etc.).
#
# Se usan dos instancias con distinto max_tokens:
#   llm       → agentes sin tools (Investigador, Analista): 500 tokens
#   llm_comm  → Comunicador (recibe contexto acumulado de 2 tareas): 950 tokens
#
# El free tier de Groq tiene límite de 1000 OTPM para qwen3.8-27b.
# ---------------------------------------------------------------------------
_llm_base = dict(
    model="openai/" + os.getenv("LLM_MODEL", "qwen/qwen3.8-27b"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    temperature=0.3,
)

llm      = LLM(**_llm_base, max_tokens=500)   # investigador y analista
llm_comm = LLM(**_llm_base, max_tokens=950)   # comunicador (más contexto)


# ---------------------------------------------------------------------------
# Herramientas (Tools)
# Solo se define la herramienta de correo, que sí tiene parámetros válidos.
# La fecha se inyecta directamente como variable en `inputs` (ver __main__).
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

        # Registrar en el log interno
        self.sent_emails.append({
            "timestamp":    timestamp,
            "destinatario": destinatario,
            "asunto":       asunto,
            "cuerpo":       cuerpo,
        })

        # Simulación visual en consola
        sep = "─" * 60
        print(f"\n{sep}")
        print("📧  SIMULACIÓN DE ENVÍO DE CORREO")
        print(sep)
        print(f"  De:          agente-comunicador@ia-curso.cl")
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


# Instancia compartida (permite acceder a sent_emails al final de la ejecución)
email_tool = SimulatedEmailTool()


# ---------------------------------------------------------------------------
# Agentes
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
    llm=llm,
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
        "crudos en insights accionables. Estructuras tu análisis con secciones "
        "claras: patrones identificados, puntos clave y conclusiones prácticas."
    ),
    llm=llm,
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
# Tareas (con dependencias explícitas mediante context=[...])
# Las variables entre llaves {variable} se reemplazan por los valores de
# `inputs` al llamar a crew.kickoff(inputs=...).
# ---------------------------------------------------------------------------

tarea_investigar = Task(
    description=(
        "Fecha de hoy: {fecha_actual}.\n"
        "Investiga el siguiente tema y recopila todos los hechos relevantes: "
        "{topic}\n\n"
        "Incluye la fecha en el encabezado del reporte. "
        "Entrega los datos en formato de lista numerada con al menos 8 hechos."
    ),
    expected_output=(
        "Fecha, título del tema y lista de 5 a 8 hechos concretos y breves "
        "(máximo 2 líneas por hecho)."
    ),
    agent=investigador,
)

tarea_analizar = Task(
    description=(
        "Analiza los datos recopilados por el investigador sobre '{topic}'. "
        "Identifica patrones, tendencias y genera conclusiones prácticas. "
        "Estructura el análisis en tres secciones: "
        "1) Patrones identificados, 2) Puntos clave, 3) Conclusiones y recomendaciones."
    ),
    expected_output=(
        "Análisis en tres secciones (Patrones, Puntos clave, Conclusiones). "
        "Máximo 3 puntos por sección, cada uno en 1-2 líneas."
    ),
    agent=analista,
    context=[tarea_investigar],
)

tarea_enviar_email = Task(
    description=(
        "Redacta un correo profesional con el análisis generado sobre '{topic}' "
        "y envíalo a '{recipient_email}' usando la herramienta de correo. "
        "El correo debe incluir: saludo, resumen ejecutivo del análisis (3-4 párrafos "
        "basados en el análisis del analista), y cierre profesional. "
        "Usa como asunto: 'Análisis: {topic} — {fecha_actual}'."
    ),
    expected_output=(
        "Confirmación del envío del correo con destinatario, asunto y timestamp."
    ),
    agent=comunicador,
    context=[tarea_investigar, tarea_analizar],
)


# ---------------------------------------------------------------------------
# Crew: equipo completo con proceso secuencial
# ---------------------------------------------------------------------------

crew = Crew(
    agents=[investigador, analista, comunicador],
    tasks=[tarea_investigar, tarea_analizar, tarea_enviar_email],
    process=Process.sequential,
    verbose=True,
)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inputs = {
        "topic":           "recetas de café",
        "recipient_email": "estudiante@duoc.cl",
        "fecha_actual":    datetime.now().strftime("%d/%m/%Y"),  # ← fecha inyectada como variable
    }

    print(f"\n{'='*60}")
    print(f"🚀  Pipeline CrewAI — 3 Agentes Especializados")
    print(f"    Tema:         {inputs['topic']}")
    print(f"    Destinatario: {inputs['recipient_email']}")
    print(f"    Fecha:        {inputs['fecha_actual']}")
    print(f"{'='*60}\n")

    result = crew.kickoff(inputs=inputs)

    print(f"\n{'='*60}")
    print("✅  RESULTADO FINAL DEL CREW")
    print(f"{'='*60}")
    print(result)

    # Log de correos enviados (estado interno de SimulatedEmailTool)
    if email_tool.sent_emails:
        print(f"\n📬  Log de correos enviados esta sesión: {len(email_tool.sent_emails)}")
        for i, e in enumerate(email_tool.sent_emails, 1):
            print(f"  [{i}] {e['timestamp']} → {e['destinatario']} | {e['asunto']}")
