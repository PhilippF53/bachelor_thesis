import ollama

class OllamaService:
    def __init__(self, model_name: str, system_prompt: str) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        
    def get_response(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        response = ollama.chat(model=self.model_name, messages=messages)
        
        return response.message.content