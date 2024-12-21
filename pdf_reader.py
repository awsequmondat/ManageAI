import PyPDF2

def read_pdf(file, progress_callback=None):
    """
    PDF dosyasını okur ve içeriğini çıkarır.
    :param file: Yüklenen PDF dosyası
    :param progress_callback: İlerleme çubuğunun güncellenmesi için geri çağırma fonksiyonu
    :return: PDF içeriği (metin)
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    total_pages = len(pdf_reader.pages)
    
    for page_num, page in enumerate(pdf_reader.pages):
        text += page.extract_text()
        
        # İlerleme çubuğunu güncelle
        if progress_callback:
            progress_callback(int(((page_num + 1) / total_pages) * 100))
    
    return text

def save_pdf_text_to_file(text, file_name):
    """
    PDF metnini dosyaya kaydeder.
    :param text: PDF metni
    :param file_name: Kaydedilecek dosyanın adı
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)

def read_pdf_text_from_file(file_name):
    """
    Kaydedilen metni dosyadan okur.
    :param file_name: Metnin kaydedildiği dosya
    :return: Dosyadaki metin
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None
