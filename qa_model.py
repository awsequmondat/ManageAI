from transformers import LongformerForQuestionAnswering, LongformerTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor

# Model ve tokenizer'ı yükleyelim
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")

def get_answer_from_chunks(question, context):
    """
    Longformer modeli ile verilen soru ve bağlamdan cevap alır.
    """
    # Metni parçalara ayıralım (örn: 1000 kelimelik parçalar)
    chunk_size = 1000  # Parça büyüklüğünü belirleyelim (isteğe göre ayarlayabilirsiniz)
    chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]

    def process_chunk(chunk):
        inputs = tokenizer.encode_plus(
            question, 
            chunk, 
            return_tensors="pt", 
            max_length=4096, 
            truncation=True, 
            padding="max_length"
        )
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        return answer

    with ThreadPoolExecutor() as executor:
        answers = list(executor.map(process_chunk, chunks))

    return " ".join(answers)