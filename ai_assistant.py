from transformers import pipeline

# Özetleme işlemi için model
summarizer = pipeline("summarization")

# Metin özetleme fonksiyonu
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']
