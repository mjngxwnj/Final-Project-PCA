# Required imports
import os
import json
import pandas as pd
import numpy as np
# from PIL import Image
import docx
import pdfplumber
from bs4 import BeautifulSoup
import librosa 
import cv2


class DataExpander:
    def __init__(self):
        pass
    
    def expand(self, file_path, modality: str = None):
        try:
            file_extension = file_path.lower().split('.')[-1]
                
            if file_extension in ('jpg', 'jpeg', 'png', 'jpe'):
                return self._read_image(file_path)
            
            elif file_extension == 'docx':
                return self._read_docx(file_path)
            
            elif file_extension == 'csv':
                return self._read_csv(file_path)
            
            elif file_extension == 'pdf':
                return self._read_pdf(file_path)
            
            elif file_extension == 'txt':
                return self._read_txt(file_path)
            
            elif file_extension == 'html':
                return self._read_html(file_path)
            
            elif file_extension == 'json':
                return self._read_json(file_path)
            
            elif file_extension in ('xls', 'xlsx'):
                return self._read_xlsx(file_path)
            
            elif file_extension == 'mp3':
                return self._read_mp3(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension} \n Files must be in : PDF, DOCX, XLSX, IMAGE, TXT, HTML, CSV, JSON, MP3")
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error processing file: {str(e)}")
        
    def _read_pdf(self, file_path):
        text_content = ""
        num_pages = 0
        try:
            with pdfplumber.open(file_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content += text + "\n"
            return {
                "type": "text",
                "content": text_content.strip(),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "pages": num_pages,
                    "char_count": len(text_content.strip())
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            content = "\n".join(full_text)
            return {
                "type": "text",
                "content": content,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "num_paragraphs": len(doc.paragraphs),
                    "word_count": len(content.split())
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_xlsx(self, file_path, sheet_name=0):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            xls = pd.ExcelFile(file_path)
            
            active_sheet_name_resolved = ""
            if isinstance(sheet_name, int):
                if 0 <= sheet_name < len(xls.sheet_names):
                    active_sheet_name_resolved = xls.sheet_names[sheet_name]
                else:
                    raise ValueError(f"Sheet index {sheet_name} is out of bounds for file {os.path.basename(file_path)} with sheets: {xls.sheet_names}")
            elif isinstance(sheet_name, str):
                if sheet_name in xls.sheet_names:
                    active_sheet_name_resolved = sheet_name
                else:
                    raise ValueError(f"Sheet name '{sheet_name}' not found in file {os.path.basename(file_path)}. Available sheets: {xls.sheet_names}")
            else:
                raise TypeError(f"sheet_name must be an int or str, not {type(sheet_name)}")


            return {
                "type": "table",
                "content": pd.DataFrame(df), 
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "sheet_names": xls.sheet_names,
                    "active_sheet_name": active_sheet_name_resolved
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_image(self, file_path):
        try:
            img = cv2.imread(file_path)
            # BGR -> RGB
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return {
                "type": "image",
                "content": img_array,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "size_pixels": img_array.shape
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: 
                content = f.read()
            return {
                "type": "text",
                "content": content,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "lines": len(content.splitlines()),
                    "char_count": len(content)
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_html(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml') # Using lxml parser
            
            text_content = soup.get_text(separator='\n', strip=True)
            
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag and title_tag.string else "N/A"
            
            links = soup.find_all('a')
            
            return {
                "type": "text", 
                "content": text_content,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "title": title,
                    "num_links": len(links),
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_csv(self, file_path, delimiter=','):
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            return {
                "type": "table",
                "content": df,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "delimiter": delimiter
                }
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            meta_info = {
                "original_filename": os.path.basename(file_path),
                "original_extension": os.path.splitext(file_path)[1].lower(),
            }
            if isinstance(data, list):
                meta_info["is_list"] = True
                meta_info["num_elements"] = len(data)
            elif isinstance(data, dict):
                meta_info["is_list"] = False
                meta_info["num_top_level_keys"] = len(data.keys())

            return {
                "type": "table",
                "content": pd.DataFrame(data),
                "meta": meta_info
            }
        except Exception as e:
            return {
                "type": "error",
                "content": str(e),
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }

    def _read_mp3(self, file_path):
        try:
            y, sr = librosa.load(file_path)
            
            return {
                "type": "audio",
                "content": y, 
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower(),
                }
            }
        except Exception as e:
            error_content = str(e)
            return {
                "type": "error",
                "content": error_content,
                "meta": {
                    "original_filename": os.path.basename(file_path),
                    "original_extension": os.path.splitext(file_path)[1].lower()
                }
            }