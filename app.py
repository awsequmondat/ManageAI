import streamlit as st
from pdf_reader import read_pdf
from ai_assistant import summarize_text
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from split_text import split_text_into_chunks, get_answer_from_chunks

# CUDA cihazını kontrol et (GPU var mı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Türkçe modelini yükle ve cihazı (GPU ya da CPU) kullanmak için ayarla
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModelForQuestionAnswering.from_pretrained("dbmdz/bert-base-turkish-cased")
model = model.to(device)  # Modeli CUDA (GPU) ya da CPU'ya taşır

# Uygulama başlığı
st.title("Yerel Yapay Zeka Asistanı")
st.write("PDF'lerinizi yükleyin ve çalıştığınız proje hakkında bilgi alın!")

# PDF dosyasını yükleme
uploaded_file = st.file_uploader("Bir PDF dosyası yükleyin", type="pdf")

if uploaded_file:
    st.write("PDF İşleniyor...")
    pdf_text = read_pdf(uploaded_file)
    
    # Hata kontrolü
    if pdf_text.startswith("Hata"):
        st.error(pdf_text)
    else:
        st.write("PDF Metni:")
        st.write(pdf_text[:1000])  # İlk 1000 karakteri göster
        
        # Özet çıkartma
        if st.button("Özet Çıkart"):
            summary = summarize_text(pdf_text)
            st.write("Özet:")
            st.write(summary)
        
        # Soru sorma alanı
        st.write("Soru Sorun:")
        question = st.text_input("Soru:")
        
        if question:
            # Metni parçalara böl
            chunks = split_text_into_chunks(pdf_text, tokenizer)

            # Cevapları her bir parça için al
            answer = get_answer_from_chunks(chunks, model, question, tokenizer, device)

            # Cevapları göster
            st.write("Cevap:")
            st.write(answer)
