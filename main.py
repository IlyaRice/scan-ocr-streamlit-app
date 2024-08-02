import streamlit as st
from PIL import Image
import pytesseract
import pyperclip

# Настройка пути к Tesseract, если необходимо
# pytesseract.pytesseract.tesseract_cmd = "/path/to/tesseract"

st.title("OCR Приложение с использованием Tesseract")

# Форма для загрузки файла
uploaded_file = st.file_uploader("Загрузите JPEG изображение", type="jpeg")

if uploaded_file:
    # Показ загруженного изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    # Кнопка для запуска OCR
    if st.button('Оцифровать текст'):
        # OCR с использованием Tesseract
        text = pytesseract.image_to_string(image, lang='rus', config='--psm 6')
        
        # Вывод оцифрованного текста с увеличенной высотой
        st.text_area("Оцифрованный текст", text, height=400)

        # Кнопка для копирования текста в буфер обмена
        if st.button("Скопировать"):
            pyperclip.copy(text)
            st.success("Текст скопирован в буфер обмена.")
