import streamlit as st
from pdf_reader import read_pdf, save_pdf_text_to_file, read_pdf_text_from_file
from qa_model import get_answer_from_chunks

# Uygulama başlığı
st.title("Yerel Yapay Zeka Asistanı")
st.write("PDF'lerinizi yükleyin ve çalıştığınız proje hakkında bilgi alın!")

# PDF dosyasını yükleme
uploaded_file = st.file_uploader("Bir PDF dosyası yükleyin", type="pdf")

pdf_text_file = "pdf_text.txt"  # Kaydedilen metin dosyasının adı

# İlerleme çubuğu
progress_bar = st.progress(0)  # Başlangıçta %0

# PDF'yi yükledikten sonra işlemler
if uploaded_file:
    # PDF okuma işlemi
    st.write("PDF işleniyor...")
    
    def progress_callback(progress):
        # İlerleme çubuğunun değerini güncelle
        progress_bar.progress(progress)

    # PDF'yi oku ve metni çıkart
    pdf_text = read_pdf(uploaded_file, progress_callback=progress_callback)
    save_pdf_text_to_file(pdf_text, pdf_text_file)
    
    st.write("PDF işlenmesi tamamlandı!")
    st.write("PDF Metni:")
    st.write(pdf_text[:1000])  # İlk 1000 karakteri göster
else:
    # PDF varsa, dosyadan okuma
    pdf_text = read_pdf_text_from_file(pdf_text_file)
    if pdf_text:
        st.write("Daha önce yüklenen PDF metni:")
        st.write(pdf_text[:1000])  # İlk 1000 karakteri göster

# Soru sorma alanı
st.write("Soru Sorun:")
question = st.text_input("Soru:")

if question and pdf_text:
    # Soruya cevap al
    answer = get_answer_from_chunks(question, pdf_text)

    # Cevap ver
    st.write("Cevap:")
    st.write(answer)