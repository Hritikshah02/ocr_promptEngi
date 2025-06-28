import os
from typing import Callable
import glob
from tesseract_ocr import ocr_file_tesseract
from parsers import parse_driving_license, parse_shop_receipt, parse_resume

PARSERS = {
    'driving_license': parse_driving_license,
    'shop_receipt': parse_shop_receipt,
    'resume': parse_resume
}

def process_documents(input_dir: str, doc_type: str, output_dir: str):
    """
    Process all files in input_dir of given doc_type and write JSON outputs to output_dir.
    """
    if doc_type not in PARSERS:
        raise ValueError(f"Unknown doc_type: {doc_type}")
    os.makedirs(output_dir, exist_ok=True)
    parser = PARSERS[doc_type]
    print(f"Processing {doc_type} documents from {input_dir}")
    print(f"Using Tesseract OCR with document scanning")
    
    for fname in os.listdir(input_dir):
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            continue
            
        # Skip non-image files
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf')):
            print(f"Skipping non-image file: {fname}")
            continue
            
        print(f"\n=== Processing {fname} ===")
        # Use Tesseract OCR with document type for better preprocessing
        text = ocr_file_tesseract(path, doc_type=doc_type)
        
        if not text or len(text.strip()) < 10:
            print(f"WARNING: Very little text extracted from {fname}. OCR may have failed.")
        
        # Parse the extracted text
        parsed = parser(text)
        # write JSON - compatible with Pydantic v2
        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            import json
            from datetime import date
            
            # Custom JSON encoder for handling dates
            class DateEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, date):
                        return obj.isoformat()
                    return super().default(obj)
            
            # Convert to dict first, then use json.dump with custom encoder
            f.write(json.dumps(parsed.model_dump(), indent=2, cls=DateEncoder))
