import groq
import base64
import os
import fitz
import io
import subprocess
import openpyxl 

from PIL import Image
from pptx import Presentation
from docx import Document
from dotenv import load_dotenv

from .processing import make_chunks, embed_chunks
from .database import create_collection, insert_to_qdrant
from .utilities import clean_text, is_garbage_text, setup_qdrant, load_model, verify_insert

load_dotenv()

groq_client = groq.Groq(api_key=os.getenv("groq_api_key"))

def analyze_image_with_groq(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    {"type": "text", "text": "Extract all text from this image. If no text, describe what you see in two or more sentences"}
                ]
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"groq error: {e}")
        return ""


def concatenate_images_vertically(images: list[Image.Image], max_pixels=30000000) -> Image.Image:
    if not images:
        return None
    
    max_width = max(img.width for img in images)
    resized = []
    for img in images:
        if img.width != max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        resized.append(img)
    
    total_height = sum(img.height for img in resized)
    combined = Image.new('RGB', (max_width, total_height), 'white')
    y = 0
    for img in resized:
        combined.paste(img, (0, y))
        y += img.height
    
    if combined.width * combined.height > max_pixels:
        scale = (max_pixels / (combined.width * combined.height)) ** 0.5
        combined = combined.resize((int(combined.width * scale), int(combined.height * scale)), Image.LANCZOS)
    
    return combined
    
def load_ppt(file_path):
    original_path = file_path
    temp_converted = False
    if file_path.lower().endswith('.ppt'):
        file_dir = os.path.dirname(file_path) or "."
        cmd = ["soffice", "--headless", "--convert-to", "pptx", "--outdir", file_dir, file_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        file_path = file_path.replace('.ppt', '.pptx')
        temp_converted = True
    
    filename = os.path.basename(original_path)
    prs = Presentation(file_path)
    
    all_slides = []
    for slide_num, slide in enumerate(prs.slides):
        slide_text = []
        slide_images = []
        
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        slide_text.append(line)
            elif shape.shape_type == 13:
                try:
                    slide_images.append(Image.open(io.BytesIO(shape.image.blob)))
                except:
                    pass
        
        all_slides.append({'num': slide_num, 'text': slide_text, 'images': slide_images})
    
    pages = []
    for i in range(0, len(all_slides), 3):
        batch = all_slides[i:i+3]
        batch_images = [img for s in batch for img in s['images']]
        
        image_text = ""
        if batch_images:
            combined = concatenate_images_vertically(batch_images)
            image_text = analyze_image_with_groq(combined)
        
        for s in batch:
            full_text = " ".join(s['text']) + " " + image_text
            cleaned = clean_text(" ".join(full_text.split()))
            
            if len(cleaned.strip()) >= 3:
                pages.append({
                    "text": cleaned,
                    "page_num": s['num'] + 1,
                    "source": filename
                })
    
    if temp_converted and os.path.exists(file_path):
        os.remove(file_path)
    return pages
def extract_images_text_from_pdf_page(doc, page):
    image_texts = []
    image_list = page.get_images(full=True)
    
    for img in image_list:
        xref = img[0]#unique id for imag
        try:
            base_image  = doc.extract_image(xref) #extract raw image byte from pdf
            image_bytes = base_image["image"]
            image    = Image.open(io.BytesIO(image_bytes))
            ocr_text = analyze_image_with_groq(image)
            
            if ocr_text.strip():
                image_texts.append(ocr_text.strip())
                
        except Exception as e:
            print(f"Skipping pdf image error: {e}")
            continue
            
    return " ".join(image_texts)

def load_pdf(file_path):
    filename = os.path.basename(file_path)
    doc      = fitz.open(file_path)
    pages    = []
    skipped  = 0

    for page_num in range(len(doc)):
        page = doc[page_num] 
        raw_text = page.get_text()
        # image_text = extract_images_text_from_pdf_page(doc, page)
        full_text = raw_text + " " 
        if is_garbage_text(full_text):
            skipped += 1
            continue
        cleaned = clean_text(full_text)
        if len(cleaned) < 50:
            skipped += 1
            continue

        pages.append({
            "text"    : cleaned,
            "page_num": page_num + 1,
            "source"  : filename
        })
    doc.close()
    return pages

def load_image(file_path):
    filename = os.path.basename(file_path)
    image    = Image.open(file_path)
    text     = analyze_image_with_groq(image)
    cleaned  = clean_text(text)
    if len(cleaned) < 5:
        return []
    return [{"text": cleaned, "page_num": 1, "source": filename}]

def load_docx(file_path):
    filename = os.path.basename(file_path)
    doc      = Document(file_path)
    pages    = []
    para_batch = []
    page_num   = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            para_batch.append(text)

        # every 20 paragraphs treat as one page
        if len(para_batch) >= 20:
            full_text = " ".join(para_batch)
            cleaned   = clean_text(full_text)

            if not is_garbage_text(full_text) and len(cleaned) >= 50:
                pages.append({
                    "text"    : cleaned,
                    "page_num": page_num,
                    "source"  : filename
                })
                page_num += 1

            para_batch = []

    #remaining paragraphs
    if para_batch:
        full_text = " ".join(para_batch)
        cleaned   = clean_text(full_text)

        if not is_garbage_text(full_text) and len(cleaned) >= 50:
            pages.append({
                "text"    : cleaned,
                "page_num": page_num,
                "source"  : filename
            })
    return pages

def load_xlsx(file_path):
    filename = os.path.basename(file_path)
    wb       = openpyxl.load_workbook(file_path, data_only=True)
    pages    = []

    for sheet_num, sheet_name in enumerate(wb.sheetnames):
        ws         = wb[sheet_name]
        sheet_rows = []

        for row in ws.iter_rows(values_only=True):
            row_text = " | ".join(
                str(cell) for cell in row
                if cell is not None and str(cell).strip()
            )
            if row_text.strip():
                sheet_rows.append(row_text)

        full_text = " ".join(sheet_rows)
        if len(full_text.strip()) < 10:
            continue
        cleaned = clean_text(full_text)
        pages.append({
            "text"    : cleaned,
            "page_num": sheet_num + 1,
            "source"  : filename
        })
    return pages

def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"]:
        return load_image(file_path)
    elif ext in [".pptx", ".ppt"]:
        return load_ppt(file_path)
    elif ext in [".xlsx", ".xls"]:
        return load_xlsx(file_path)
    elif ext in [".docx", ".doc"]: 
        return load_docx(file_path)
    else:
        print(f"Unsupported file type: {ext}")
        return []

def main_pipeline(file_paths):
    client = setup_qdrant()
    create_collection(client)
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_pages  = []

    for file_path in file_paths:
        print(f"processing: {file_path}")
        pages = load_file(file_path)
        all_pages.extend(pages)

    chunks  = make_chunks(all_pages)
    model   = load_model()
    vectors = embed_chunks(chunks, model)
    insert_to_qdrant(chunks, vectors, client)
    verify_insert(client)

    return client, model

if __name__ == "__main__":
    files = [
        #add files to process
    ]
    client, model = main_pipeline(files)