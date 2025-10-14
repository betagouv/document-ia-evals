import pandas as pd
from typing import List, Optional
import requests
from io import BytesIO
from PIL import Image
import fitz
import os
from pydantic import BaseModel, Field
import numpy as np
import cv2
from pylibdmtx.pylibdmtx import decode
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc
import dotenv
from tdd.doc import TwoDDoc

dotenv.load_dotenv('../../.env')

def load_tax_notice_dataset(csv_path: str = "../../datasets/2d-doc/tax-notices.csv") -> List[str]:
    df = pd.read_csv(csv_path, usecols=["file_id"])
    ids = df["file_id"].dropna().astype(str).str.strip()
    return [s for s in ids.tolist() if s]

def get_tax_notice_image(file_id):
    """
    Fetch a file from DossierFacile and return it as a PIL Image.
    
    Args:
        file_id: The file identifier (e.g., "xx xxx xxx")
    
    Returns:
        PIL.Image: The loaded image
    """
    cookies = {'JSESSIONID': os.getenv("DOSSIER_FACILE_JSESSIONID")}
    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    
    url = f'https://bo.dossierfacile.fr/files/{file_id}'
    response = requests.get(url, cookies=cookies, headers=headers)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '').lower()
    
    if 'pdf' in content_type:
        doc = fitz.open(stream=response.content, filetype='pdf')
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        doc.close()
        del pix, page, doc
        gc.collect()
        return img
    
    img = Image.open(BytesIO(response.content))
    img.load()
    return img

class TaxNoticeData(BaseModel):
    """Tax notice data extracted from 2D-DOC"""
    doc_type: Optional[str] = None
    emitter_type: Optional[str] = None
    
    # Dataset fields with English names
    number_of_shares: Optional[str] = Field(None, description="Nombre de parts")
    tax_notice_reference: Optional[str] = Field(None, description="Référence d'avis d'impôt")
    income_year: Optional[str] = Field(None, description="Année des revenus")
    declarant_1: Optional[str] = Field(None, description="Déclarant 1")
    collection_date: Optional[str] = Field(None, description="Date de mise en recouvrement")
    tax_number_declarant_1: Optional[str] = Field(None, description="Numéro fiscal du déclarant 1")
    reference_tax_income: Optional[str] = Field(None, description="Revenu fiscal de référence")

def extract_tax_notice(pil_image: Image.Image, timeout=5000) -> Optional[TaxNoticeData]:
    """
    Extract tax notice information from a PIL Image containing a DataMatrix code.
    
    Args:
        pil_image: PIL Image object containing a DataMatrix code
        
    Returns:
        TaxNoticeData object with extracted information, or None if no code found
    """
    # Convert PIL Image to numpy array
    img_array = np.array(pil_image)
    
    # Convert to BGR if needed (OpenCV uses BGR)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    del img_array
    gc.collect()
    
    # Decode DataMatrix
    res = decode(gray, max_count=1, timeout=timeout)
    
    del gray
    gc.collect()
    
    if not res:
        return None
    
    doc = None
    # Parse 2D-DOC
    for potential_doc in res:
        try:
            raw_data = potential_doc.data.decode('latin-1')
            doc = TwoDDoc.from_code(raw_data)
        except Exception:
            # not a 2d-doc
            continue

    del res
    gc.collect()
    
    if doc is None:
        return None

    # Mapping from French field names to English attributes
    field_mapping = {
        "Nombre de parts": "number_of_shares",
        "Référence d'avis d'impôt": "tax_notice_reference",
        "Année des revenus": "income_year",
        "Déclarant 1": "declarant_1",
        "Date de mise en recouvrement": "collection_date",
        "Numéro fiscal du déclarant 1": "tax_number_declarant_1",
        "Revenu fiscal de référence": "reference_tax_income",
    }
    
    # Extract data
    data = {
        "doc_type": doc.header.doc_type().user_type if hasattr(doc.header.doc_type(), 'user_type') else None,
        "emitter_type": doc.header.doc_type().emitter_type if hasattr(doc.header.doc_type(), 'emitter_type') else None,
    }
    
    # Extract fields from dataset
    for item in doc.message.dataset:
        field_name = item.definition.name
        if field_name in field_mapping:
            data[field_mapping[field_name]] = str(item.value)
    
    del doc
    gc.collect()
    
    return TaxNoticeData(**data)

def _worker(file_id: str):
    """
    Worker executed in a separate process.
    Returns (file_id, ok:bool, tax_notice_dict or None)
    """
    try:
        img = get_tax_notice_image(file_id)
        tax_notice = extract_tax_notice(img, timeout=5000)
        
        # Free memory immediately
        del img
        gc.collect()
        
        if tax_notice:
            return (file_id, True, tax_notice.model_dump())
        return (file_id, False, None)
    except Exception:
        return (file_id, False, None)

def save_tax_results_batch(results_list, output_path="tax_results.csv"):
    """Save a batch of results directly from a list"""
    if not results_list:
        return
    
    df = pd.DataFrame(results_list)
    mode = "a" if os.path.exists(output_path) else "w"
    header = not os.path.exists(output_path)
    df.to_csv(output_path, mode=mode, header=header, index=False)

# Main processing
dataset = load_tax_notice_dataset()
output_path = "../../datasets/2d-doc/tax-notices-extracted-2d-doc.csv"
chunk_size = 100  # Process and save in chunks to reset processes and release memory
batch_size_save = 100  # But since chunk_size=100, it aligns
pending_batch = []
success_count = 0
total_count = len(dataset)

print(f"Processing {total_count} files in chunks of {chunk_size} for memory efficiency.")

with tqdm(total=total_count, desc="Processing tax notices", unit="file") as pbar:
    for start_idx in range(0, total_count, chunk_size):
        chunk = dataset[start_idx:start_idx + chunk_size]
        
        with ProcessPoolExecutor(max_workers=10) as exe:
            futures = {exe.submit(_worker, fid): fid for fid in chunk}
            
            for fut in as_completed(futures):
                file_id = futures[fut]
                try:
                    file_id, ok, tax_dict = fut.result()
                except Exception:
                    ok, tax_dict = False, None

                pbar.set_postfix({"success": "✓" if ok else "✗"})
                pbar.update(1)

                if ok and tax_dict:
                    tax_dict["file_id"] = file_id
                    pending_batch.append(tax_dict)
                    success_count += 1

                # Save if batch ready (though with chunk_size=100, saves per chunk)
                if len(pending_batch) >= batch_size_save:
                    save_tax_results_batch(pending_batch, output_path)
                    pending_batch.clear()
        
        # Force cleanup after each chunk
        gc.collect()

# Final flush
if pending_batch:
    save_tax_results_batch(pending_batch, output_path)
    pending_batch.clear()

print(f"\n✓ Successfully processed: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")