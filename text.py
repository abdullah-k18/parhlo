from pdf2image import convert_from_path
import pytesseract

# Convert PDF pages to images
pages = convert_from_path("Physics 12 watermark Chapter 1 - 4.pdf")

for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang="eng")
    print(f"--- Page {i+1} ---")
    print(text)
    print("\n")
