import gradio as gr
from sa_facade import SentimentAnalysisFacade

facade = SentimentAnalysisFacade()


def analyze(text, correction=""):
    sentiment = facade.analyze_sentiment(text)

    if correction:
        correction_result = facade.correct_sentiment(text, correction)
        return correction_result

    return sentiment


iface = gr.Interface(fn=analyze,
                     inputs=[gr.Textbox(label="Feedback do produto:", placeholder="Digite o feedback do produto aqui:"),
                             gr.Textbox(label="Correção da análise:", placeholder="Positivo ou Negativo")],
                     outputs="text",
                     title="SENTIX")

iface.launch(share=True)
