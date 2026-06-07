from app.llm.ollama_llm import OllamaLLM

llm = OllamaLLM()

response = llm.generate([
    {
        "role": "user",
        "content": "Who founded Microsoft?"
    }
])

print(response)