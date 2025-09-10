from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class EmbedderFactory:
    @staticmethod
    def create_embedder(provider, model_name, device="cuda"):
        if provider == "huggingface":
            if not model_name:
                raise ValueError("Please specify the HuggingFace model.")
            
            return HuggingFaceEmbedding(model_name=model_name, device=device, embed_batch_size=16)
        
        else:
            raise ValueError(f"Currently support HuggingFace only.")