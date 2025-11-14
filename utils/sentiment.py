from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

SMT_MODEL = "tabularisai/multilingual-sentiment-analysis"

def get_sentiment_pipeline() -> pipeline:
    sentiment_pipeline = pipeline(
        task="text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(SMT_MODEL),
        tokenizer=AutoTokenizer.from_pretrained(SMT_MODEL),
        truncation=True,
        max_length=512
    )
    return sentiment_pipeline

class Sentiment:
    def __init__(self):
        self.smt_pipe = get_sentiment_pipeline()

    def get_feedback_sentiment(self, feedback: str):
        if not feedback.strip():
            return {
                "label": "NEUTRAL",
                "score": 0.0
            }
        result = self.smt_pipe(feedback)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }