import gradio as gr
import pandas as pd

from app.services.chat_service import ChatService

from app.visualization.vector_visualizer import (
    VectorVisualizer,
)

from app.evaluation.evaluator import (
    evaluate_all_retrieval,
)

from app.evaluation.evaluator import (
    evaluate_all_answers,
)

from app.vectorstore.chroma_store import (
    ChromaStore,
)

from app.utils.logger import get_logger

logger = get_logger(__name__)


chat_service = ChatService()
visualizer = VectorVisualizer()
store = ChromaStore()


# =====================================================
# CHAT FUNCTIONS
# =====================================================

def chat_response(message, history):

    history = history or []

    answer, docs = chat_service.answer(
        question=message,
        history=history
    )

    context_text = ""

    for doc in docs:

        context_text += (
            f"📄 Source: "
            f"{doc.metadata.get('source')}\n\n"
        )

        context_text += (
            doc.page_content[:500]
            + "\n"
            + "=" * 80
            + "\n\n"
        )

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    return (
        history,
        history,
        context_text
    )


# =====================================================
# VECTOR VISUALIZATION
# =====================================================

def load_2d_plot():

    return visualizer.create_2d_plot()


def load_3d_plot():

    return visualizer.create_3d_plot()


# =====================================================
# RETRIEVAL EVALUATION
# =====================================================

def run_retrieval_eval():

    results = evaluate_all_retrieval()

    df = pd.DataFrame(results)

    return df


# =====================================================
# ANSWER EVALUATION
# =====================================================

def run_answer_eval():

    results = evaluate_all_answers()

    df = pd.DataFrame(results)

    return df


# =====================================================
# SYSTEM STATS
# =====================================================

def load_system_stats():

    data = store.get_collection_data()

    total_vectors = len(data["documents"])

    metadata = data["metadatas"]

    company_count = len(
        [
            x
            for x in metadata
            if x["doc_type"] == "company"
        ]
    )

    employee_count = len(
        [
            x
            for x in metadata
            if x["doc_type"] == "employees"
        ]
    )

    product_count = len(
        [
            x
            for x in metadata
            if x["doc_type"] == "products"
        ]
    )

    contract_count = len(
        [
            x
            for x in metadata
            if x["doc_type"] == "contracts"
        ]
    )

    return pd.DataFrame(
        {
            "Metric": [
                "Total Vectors",
                "Company Chunks",
                "Employee Chunks",
                "Product Chunks",
                "Contract Chunks"
            ],
            "Value": [
                total_vectors,
                company_count,
                employee_count,
                product_count,
                contract_count
            ]
        }
    )


# =====================================================
# CUSTOM CSS
# =====================================================

CUSTOM_CSS = """

.gradio-container {
    max-width: 100% !important;
}

#title {
    text-align:center;
    font-size:34px;
    font-weight:bold;
    margin-bottom:10px;
}

#subtitle {
    text-align:center;
    font-size:18px;
    margin-bottom:25px;
}

"""


# =====================================================
# UI
# =====================================================

with gr.Blocks(
    title="Enterprise RAG Assistant",
    # css=CUSTOM_CSS,
    # theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        <div id='title'>
        Enterprise RAG Assistant
        </div>

        <div id='subtitle'>
        Semantic Search • ChromaDB • LangChain • Ollama • Evaluation Dashboard
        </div>
        """
    )

    with gr.Tabs():

        # =============================================
        # TAB 1 CHATBOT
        # =============================================

        with gr.Tab("💬 AI Assistant"):

            with gr.Row():

                with gr.Column(scale=3):

                    chatbot = gr.Chatbot(
                        height=600,
                        label="Assistant"
                    )

                    message = gr.Textbox(
                        placeholder="Ask a question...",
                        label="Question"
                    )

                    send_btn = gr.Button(
                        "Send",
                        variant="primary"
                    )

                with gr.Column(scale=2):

                    retrieved_context = gr.Textbox(
                        label="Retrieved Context",
                        lines=30
                    )

            state = gr.State([])

            send_btn.click(
                fn=chat_response,
                inputs=[
                    message,
                    state
                ],
                outputs=[
                    chatbot,
                    state,
                    retrieved_context
                ]
            )

        # =============================================
        # TAB 2 VECTOR VISUALIZATION
        # =============================================

        with gr.Tab("📊 Vector Visualization"):

            gr.Markdown(
                """
                Visualize embeddings reduced
                from 384 dimensions using t-SNE.
                """
            )

            with gr.Row():

                plot2d = gr.Plot(
                    label="2D Visualization"
                )

                plot3d = gr.Plot(
                    label="3D Visualization"
                )

            load_vectors_btn = gr.Button(
                "Generate Visualization"
            )

            load_vectors_btn.click(
                fn=load_2d_plot,
                outputs=plot2d
            )

            load_vectors_btn.click(
                fn=load_3d_plot,
                outputs=plot3d
            )

        # =============================================
        # TAB 3 RETRIEVAL EVALUATION
        # =============================================

        with gr.Tab("🔍 Retrieval Evaluation"):

            retrieval_btn = gr.Button(
                "Run Retrieval Evaluation"
            )

            retrieval_df = gr.Dataframe()

            retrieval_btn.click(
                fn=run_retrieval_eval,
                outputs=retrieval_df
            )

        # =============================================
        # TAB 4 ANSWER EVALUATION
        # =============================================

        with gr.Tab("📝 Answer Evaluation"):

            answer_btn = gr.Button(
                "Run Answer Evaluation"
            )

            answer_df = gr.Dataframe()

            answer_btn.click(
                fn=run_answer_eval,
                outputs=answer_df
            )

        # =============================================
        # TAB 5 SYSTEM ANALYTICS
        # =============================================

        with gr.Tab("📈 Analytics"):

            stats_btn = gr.Button(
                "Load Statistics"
            )

            stats_table = gr.Dataframe()

            stats_btn.click(
                fn=load_system_stats,
                outputs=stats_table
            )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )