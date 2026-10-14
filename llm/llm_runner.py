class MockLLMRunner:
    backend="MOCK"
    def generate(self,prompt, **kwargs):
        return "Verified protocol retrieved. Follow established command procedure and report the observed direction."
class LlamaCppRunner:
    def __init__(self, model_path, **kwargs):
        try:
            from llama_cpp import Llama; self.model=Llama(model_path=model_path, **kwargs)
        except ImportError as e: raise RuntimeError("Install llama-cpp-python to use a GGUF model") from e
    def generate(self,prompt,**kwargs): return self.model.create_completion(prompt=prompt, max_tokens=kwargs.get("max_tokens",128))["choices"][0]["text"].strip()
