
from utils.etc import hit2docdict
import torch
# Modify
class ModelRAG():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def search(self, queries, qids, k=5):
        # Use the retriever to get relevant documents
        list_passages = []
        list_scores = []

        # fill here
        ######
        
        for query, qid in zip(queries, qids):
            hits = self.retriever.search(query, k)
            docs = []
            scores = []
            for hit in hits:
                doc, score = hit2docdict(hit)
                docs.append(doc)
                scores.append(score)
            list_passages.append(docs)
            list_scores.append(scores)
        
        ######

        return list_passages, list_scores

    # Modify
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # Get the relevant documents for each query
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        # fill here
        ######
        
        for query, passages in zip(queries, list_passages):
            context = ""
            for passage in passages:
                context += f"Title: \nPassage: {passage}\n"
            input_text = f"{context}Question: {query}\nAnswer:"
            list_input_text_without_answer.append(input_text)
        
        ######
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######
        
        input_texts = self.make_augmented_inputs_for_generate(queries, qids, k=k)
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        ######

        # # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs
