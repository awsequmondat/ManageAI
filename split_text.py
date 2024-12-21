# split_text.py
from transformers import AutoTokenizer

def split_text_into_chunks(text, tokenizer, max_length=512):
    # Metni token'lara dönüştür
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=True)
    
    # Eğer metin çok uzun ise, 512 token'lık parçalara ayır
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
    
    return chunks

def get_answer_from_chunks(chunks, model, question, tokenizer, device):
    answers = []
    
    for chunk in chunks:
        # Her parça için inputları al
        inputs = tokenizer(question, chunk, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Modeli çalıştır (GPU üzerinde işlem yapacak şekilde)
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Modelin cevabını al
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            # Cevap başlangıç ve bitiş index'lerini bulma
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)

            # Cevap tokenlarını çözümleyerek cevabı al
            answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
            answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            answers.append(answer)
    
    # Cevapları birleştir
    return " ".join(answers)
