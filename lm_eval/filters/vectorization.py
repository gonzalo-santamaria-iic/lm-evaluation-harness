from lm_eval.api.filter import Filter
from sentence_transformers import SentenceTransformer

@register_filter("embedding")
class EmbeddingFilter(Filter):
    def __init__(
            self,
            model_name_or_path: SentenceTransformer = "sentence-transformers/all-roberta-large-v1",
    ) -> None:
        """
        Initializes the EmbeddingFilter with a given sentence transformer model.

        Args:
        - model_name_or_path (SentenceTransformer): The name of the encoder that transform the texts into embeddings.

        Example:
        embeddings = EmbeddingFilter(["this is a sentence", "this is a sencond sentence", "this is the last sentence"])
        """

        self.model = SentenceTransformer(model_name_or_path)

    def apply(self, resps, docs):
        embeddings = self.model.encode(resps)
        return embeddings.numpy()
