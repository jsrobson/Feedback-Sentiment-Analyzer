# == Standard library imports ==
import os

import pandas as pd
# == Third party imports ==
from dotenv import load_dotenv
from torch import bfloat16
from transformers import pipeline

load_dotenv()

ACCESS_TOKEN = os.getenv("HF_API_KEY")
THEME_MODEL = "google/gemma-3-4b-it"

def get_topic_pipeline() -> pipeline:
    topic_pipeline = pipeline(
        task="text-generation",
        model=THEME_MODEL,
        device="cpu",
        dtype=bfloat16,
        token=ACCESS_TOKEN
    )
    return topic_pipeline

def _bundle_messages(prompt: str):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

class Summary:
    def __init__(self):
        self.t_pipe = get_topic_pipeline()

    def get_output(self, name: str, prompt: str):
        try:
            output = self.t_pipe(_bundle_messages(prompt))
            gen_text = output[0]["generated_text"][-1]["content"]
            summary = gen_text.strip()
        except Exception as e:
            print(f"Summary generation failed for '{name}': {e}")
            summary = "Error generating summary"
        return summary