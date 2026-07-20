import os
from openai import OpenAI

class LLMConfigurationError(Exception):
    pass

def generate_answer(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError("OPENAI_API_KEY is not configured.")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()

