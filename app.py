import gradio as gr
from src.agent import run_agent

EJEMPLOS = [
    ["Canasta vegetariana para 2 personas, presupuesto $80.000 COP"],
    ["Desayuno saludable bajo en azúcar para una persona"],
    ["Opciones veganas, sin lácteos ni huevo, máximo $60.000 COP"],
    ["Canasta alta en proteína para deportista, $100.000 COP"],
    ["Productos sin gluten y bajos en sodio"],
]

# ── Función que conecta Gradio con el agente ──────────────────────────────────

def chat(user_request: str, history_state: list) -> tuple[str, str, list]:
    """
    history_state: lista de dicts guardada en gr.State (memoria entre turnos)
    Retorna: (canasta, debug, history_state_actualizado)
    """
    if not user_request.strip():
        return "Por favor escribe tu solicitud.", "", history_state

    basket, updated_history, debug = run_agent(user_request, history_state)
    return basket, debug, updated_history


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="🛒 Shopper Assistant — retail",
    theme=gr.themes.Soft(primary_hue="green"),
) as demo:

    # Estado de memoria (persiste entre mensajes en la misma sesión)
    history_state = gr.State([])

    gr.Markdown(
        """
        # 🛒 Shopper Assistant 
        Cuéntame qué necesitas en lenguaje natural y armo tu canasta personalizada
        según tu presupuesto, preferencias dietéticas y restricciones nutricionales.

        > 💬 **Tip:** Puedes hacer preguntas de seguimiento. El agente recuerda tu conversación.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            txt_input = gr.Textbox(
                label="¿Qué necesitas?",
                placeholder="Ej: Canasta vegetariana para 2 personas, presupuesto $80.000 COP...",
                lines=3,
            )
            btn = gr.Button("🧺 Generar canasta", variant="primary")
            gr.Examples(
                examples=EJEMPLOS,
                inputs=txt_input,
                label="Ejemplos",
            )

        with gr.Column(scale=4):
            out_canasta = gr.Markdown(label="Canasta sugerida")
            with gr.Accordion("🔍 Detalle del proceso (LangGraph)", open=False):
                out_debug = gr.Markdown()

    btn.click(
        fn=chat,
        inputs=[txt_input, history_state],
        outputs=[out_canasta, out_debug, history_state],
    )
    txt_input.submit(
        fn=chat,
        inputs=[txt_input, history_state],
        outputs=[out_canasta, out_debug, history_state],
    )

if __name__ == "__main__":
    demo.launch(share=False)