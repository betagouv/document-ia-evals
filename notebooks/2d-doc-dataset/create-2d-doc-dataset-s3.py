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
import boto3
import json
from botocore.exceptions import ClientError
import logging
import time

from tdd.doc import TwoDDoc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv('../../.env')

def load_tax_notice_dataset(csv_path: str = "../../datasets/2d-doc/tax-notices.csv") -> List[str]:
    df = pd.read_csv(csv_path, usecols=["file_id"])
    ids = df["file_id"].dropna().astype(str).str.strip()
    return [s for s in ids.tolist() if s]

def get_tax_notice_image(file_id):
    """
    Fetch a file from DossierFacile and return it as a PIL Image and raw content.
    
    Args:
        file_id: The file identifier (e.g., "xx xxx xxx")
    
    Returns:
        Tuple[PIL.Image, bytes, str]: (Image object, raw content, content type)
    """
    cookies = {'JSESSIONID': os.getenv("DOSSIER_FACILE_JSESSIONID")}
    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    
    url = f'https://bo.dossierfacile.fr/files/{file_id}'
    response = requests.get(url, cookies=cookies, headers=headers)
    response.raise_for_status()
    
    content_type = response.headers.get('content-type', '').lower()
    raw_content = response.content
    
    if 'pdf' in content_type:
        doc = fitz.open(stream=raw_content, filetype='pdf')
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        doc.close()
        del pix, page, doc
        gc.collect()
        return img, raw_content, 'application/pdf'
    
    img = Image.open(BytesIO(raw_content))
    img.load()
    return img, raw_content, content_type

class AvisImpositionExtract(BaseModel):
    annee_revenus: str
    revenu_fiscal_reference: float
    declarant_1_nom: str
    declarant_1_prenom: Optional[str] = None
    declarant_1_numero_fiscal: Optional[str] = None
    reference_avis: Optional[str] = None
    nombre_parts: Optional[float] = None
    date_mise_en_recouvrement: Optional[str] = None
    revenu_brut_global: Optional[float] = None
    revenu_imposable: Optional[float] = None
    impot_revenu_net_avant_corrections: Optional[float] = None
    montant_impot: Optional[float] = None

def extract_tax_notice(pil_image: Image.Image, timeout=5000) -> Optional[AvisImpositionExtract]:
    """
    Extract tax notice information from a PIL Image containing a DataMatrix code.
    
    Args:
        pil_image: PIL Image object containing a DataMatrix code
        
    Returns:
        AvisImpositionExtract object with extracted information, or None if no code found
    """
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    
    del img_array
    gc.collect()
    
    res = decode(gray, max_count=1, timeout=timeout)
    
    del gray
    gc.collect()
    
    if not res:
        return None
    
    doc = None
    for potential_doc in res:
        try:
            raw_data = potential_doc.data.decode('latin-1')
            doc = TwoDDoc.from_code(raw_data)
        except Exception:
            continue

    del res
    gc.collect()
    
    if doc is None:
        return None

    # Mapping des champs 2D-DOC vers AvisImpositionExtract
    field_mapping = {
        "Année des revenus": "annee_revenus",
        "Revenu fiscal de référence": "revenu_fiscal_reference",
        "Déclarant 1": "declarant_1_nom",  # Contient nom et prénom
        "Numéro fiscal du déclarant 1": "declarant_1_numero_fiscal",
        "Référence d'avis d'impôt": "reference_avis",
        "Nombre de parts": "nombre_parts",
        "Date de mise en recouvrement": "date_mise_en_recouvrement",
        "Revenu brut global": "revenu_brut_global",
        "Revenu imposable": "revenu_imposable",
        "Impôt sur le revenu net avant corrections": "impot_revenu_net_avant_corrections",
        "Montant de l'impôt": "montant_impot"
    }
    
    data = {}
    
    for item in doc.message.dataset:
        field_name = item.definition.name
        if field_name in field_mapping:
            pydantic_field = field_mapping[field_name]
            value = str(item.value)
            
            # Traitement spécifique pour séparer nom/prénom du déclarant
            if pydantic_field == "declarant_1_nom":
                parts = value.split(maxsplit=1)
                data["declarant_1_nom"] = parts[0] if parts else value
                data["declarant_1_prenom"] = parts[1] if len(parts) > 1 else None
            # Conversion en float pour les champs numériques
            elif pydantic_field in ["revenu_fiscal_reference", "nombre_parts", "revenu_brut_global", 
                                   "revenu_imposable", "impot_revenu_net_avant_corrections", "montant_impot"]:
                try:
                    data[pydantic_field] = float(value.replace(" ", "").replace(",", "."))
                except ValueError:
                    data[pydantic_field] = None
            else:
                data[pydantic_field] = value
    
    del doc
    gc.collect()
    
    # Vérifier que les champs obligatoires sont présents
    required_fields = ["annee_revenus", "revenu_fiscal_reference", "declarant_1_nom"]
    if not all(field in data for field in required_fields):
        return None
    
    return AvisImpositionExtract(**data)

def pydantic_to_annotation_result(model_instance: BaseModel) -> list:
    """Convertit une instance Pydantic en structure d'annotation Label Studio"""
    results = []
    
    for field_name, value in model_instance.model_dump().items():
        if value is not None:  # Ne pas inclure les valeurs None
            results.append({
                'value': {'text': [str(value)]},
                'from_name': field_name,
                'to_name': 'image',
                'type': 'textarea',
                'readonly': False
            })
    
    # Ajout du JSON brut comme une annotation supplémentaire
    results.append({
        'value': {'text': [json.dumps(model_instance.model_dump())]},
        'from_name': 'raw_api_response',
        'to_name': 'image',
        'type': 'textarea'
    })
    
    return results

def create_task(image_url, ground_truth: Optional[BaseModel] = None, pipelines: Optional[list] = None):
    """
    Crée une task Label Studio complète à partir de modèles Pydantic
    
    Args:
        image_url: URL de l'image à annoter
        ground_truth: Instance Pydantic de la ground truth (optionnel)
        pipelines: Liste de dicts avec 'name' et 'data' (instance Pydantic) (optionnel)
    """
    task = {
        'data': {
            'image': image_url
        }
    }
    
    # Ground truth en tant qu'annotation
    if ground_truth:
        task['annotations'] = [{
            'result': pydantic_to_annotation_result(ground_truth),
            'ground_truth': True
        }]
    
    # Prédictions des pipelines
    if pipelines:
        task['predictions'] = []
        for pipeline in pipelines:
            task['predictions'].append({
                'result': pydantic_to_annotation_result(pipeline['data']),
                'model_version': pipeline['name']
            })
    
    return task

def upload_to_s3(s3_client, bucket: str, prefix: str, file_id: str, tax_notice: AvisImpositionExtract, 
                raw_content: bytes, content_type: str, retries=3, delay=1):
    """
    Upload raw file and tax notice data as JSON to S3 in Label Studio task format.
    
    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix for JSON (e.g., 'tax-notices-extracted/')
        file_id: File identifier for naming
        tax_notice: AvisImpositionExtract instance
        raw_content: Raw file content (PDF or image)
        content_type: Content type of raw file
        retries: Number of retry attempts
        delay: Delay between retries in seconds
    """
    try:
        # Determine file extension
        ext = '.pdf' if 'pdf' in content_type else '.jpg' if 'jpeg' in content_type else '.png'
        
        # Upload raw file
        raw_key = f"source/{file_id}{ext}"
        s3_client.put_object(
            Bucket=bucket,
            Key=raw_key,
            Body=raw_content,
            ContentType=content_type
        )
        logger.info(f"Successfully uploaded raw file {file_id} to s3://{bucket}/{raw_key}")
        
        # Create Label Studio task using the provided function
        image_url = f"s3://{bucket}/{raw_key}"
        task_data = create_task(
            image_url=image_url,
            ground_truth=tax_notice
        )
        
        # Upload JSON task
        json_key = f"{prefix}{file_id}.json"
        json_data = json.dumps(task_data, ensure_ascii=False, indent=2)
        s3_client.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json_data.encode('utf-8'),
            ContentType='application/json'
        )
        logger.info(f"Successfully uploaded JSON task {file_id} to s3://{bucket}/{json_key}")
        
    except ClientError as e:
        logger.error(f"Failed to upload {file_id} to S3 (attempt 1/{retries}): {e}")
        if retries > 1:
            time.sleep(delay)
            upload_to_s3(s3_client, bucket, prefix, file_id, tax_notice, raw_content, content_type, retries-1, delay*2)
        else:
            raise

def _worker(file_id: str, bucket: str, prefix: str):
    """
    Worker executed in a separate process.
    Returns (file_id, ok:bool, tax_notice_dict or None)
    """
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT'),
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            region_name=os.getenv('S3_REGION')
        )
        
        img, raw_content, content_type = get_tax_notice_image(file_id)
        tax_notice = extract_tax_notice(img, timeout=5000)
        
        del img
        gc.collect()
        
        if tax_notice:
            upload_to_s3(s3_client, bucket, prefix, file_id, tax_notice, raw_content, content_type)
            return (file_id, True, tax_notice.model_dump())
        return (file_id, False, None)
    except Exception as e:
        logger.error(f"Error processing {file_id}: {e}")
        return (file_id, False, None)

# Main processing
dataset = load_tax_notice_dataset()[0:1600]

bucket_name = os.getenv('S3_BUCKET_NAME')
s3_prefix = "tax-notices-extracted/"
chunk_size = 100
success_count = 0
total_count = len(dataset)

logger.info(f"Processing {total_count} files in chunks of {chunk_size} for memory efficiency.")

with tqdm(total=total_count, desc="Processing tax notices", unit="file") as pbar:
    for start_idx in range(0, total_count, chunk_size):
        chunk = dataset[start_idx:start_idx + chunk_size]
        
        with ProcessPoolExecutor(max_workers=10) as exe:
            futures = {exe.submit(_worker, fid, bucket_name, s3_prefix): fid for fid in chunk}
            
            for fut in as_completed(futures):
                file_id = futures[fut]
                try:
                    file_id, ok, tax_dict = fut.result()
                except Exception as e:
                    logger.error(f"Error in future for {file_id}: {e}")
                    ok, tax_dict = False, None

                pbar.set_postfix({"success": "✓" if ok else "✗"})
                pbar.update(1)

                if ok and tax_dict:
                    success_count += 1
        
        gc.collect()

logger.info(f"✓ Successfully processed: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")