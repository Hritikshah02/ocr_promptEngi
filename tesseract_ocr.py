import os
import io
from PIL import Image
import cv2
import numpy as np
import pytesseract
from document_scanner import scan_document, scan_receipt

def preprocess_image_for_ocr(image, doc_type=None):
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image or numpy array
        doc_type: Type of document ('shop_receipt', 'driving_license', etc.)
        
    Returns:
        Preprocessed PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert PIL to numpy array if needed
    if not isinstance(image, np.ndarray):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Handle different color formats
    if len(img_array.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = img_array
    
    # Apply document-specific preprocessing
    if doc_type == 'shop_receipt':
        try:
            print("Applying receipt-specific document scanning...")
            print(f"Image shape before scanning: {img_cv.shape}")
            print(f"Scanning receipt: {img_cv}")
            
            # Try to scan and crop the receipt
            scanned = scan_receipt(img_cv)
            
            # Convert to grayscale
            gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY) if len(scanned.shape) == 3 else scanned
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Convert back to PIL
            return Image.fromarray(denoised)
        except Exception as e:
            print(f"WARNING: Could not scan receipt: {str(e)}. Using basic preprocessing.")
            # Fall back to basic preprocessing
    
    # Basic preprocessing for all document types
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Convert back to PIL
    return Image.fromarray(denoised)

def tesseract_ocr_extract(image, doc_type=None, lang=None):
    """
    Extract text from image using Tesseract OCR
    
    Args:
        image: PIL Image
        doc_type: Type of document
        lang: Language for OCR (e.g., 'eng')
        
    Returns:
        Extracted text string
    """
    try:
        # Set language if provided, otherwise use English
        lang_param = lang if lang else 'eng'
        
        # Set Tesseract configuration based on document type
        config = ''
        if doc_type == 'shop_receipt':
            # For receipts, optimize for single column text
            config = '--oem 1 --psm 4 -c preserve_interword_spaces=1'
        elif doc_type == 'driving_license':
            # For IDs, use sparse text mode
            config = '--oem 1 --psm 11 -c preserve_interword_spaces=1'
        else:
            # Default configuration
            config = '--oem 1 --psm 3'
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(image, lang=lang_param, config=config)
        return text
        
    except Exception as e:
        print(f"Error in Tesseract OCR: {str(e)}")
        return ""

def ocr_file_tesseract(file_path, doc_type=None, lang=None):
    """
    Extract text from an image or PDF file using Tesseract OCR with preprocessing
    
    Args:
        file_path: Path to the image or PDF file
        doc_type: Type of document ('driving_license', 'shop_receipt', or 'resume')
        lang: Language for OCR (e.g., 'eng')
        
    Returns:
        Extracted text from the document
    """
    print(f"Processing file with Tesseract OCR: {file_path}")
    
    # Infer document type from file path if not provided
    if doc_type is None:
        if 'shop_receipt' in file_path.lower() or 'receipt' in file_path.lower():
            doc_type = 'shop_receipt'
        elif 'license' in file_path.lower() or 'licence' in file_path.lower() or 'driving' in file_path.lower():
            doc_type = 'driving_license'
        elif 'resume' in file_path.lower() or 'cv' in file_path.lower():
            doc_type = 'resume'
    
    print(f"Document type detected: {doc_type}")
    
    try:
        # Check if file is PDF
        if file_path.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300)
            text = ""
            
            # Process each page
            for i, img in enumerate(images):
                print(f"Processing PDF page {i+1}")
                
                # Try multiple preprocessing approaches
                results = []
                
                # Approach 1: Document scanning + preprocessing + OCR
                try:
                    preprocessed_img = preprocess_image_for_ocr(img, doc_type)
                    text1 = tesseract_ocr_extract(preprocessed_img, doc_type, lang)
                    results.append((len(text1), text1, "Document Scanning"))
                except Exception as e:
                    print(f"Approach 1 failed: {str(e)}")
                    results.append((0, "", "Document Scanning"))
                
                # Approach 2: Basic preprocessing + OCR
                try:
                    # Convert to grayscale and enhance contrast
                    img_array = np.array(img)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray)
                    enhanced_pil = Image.fromarray(enhanced)
                    
                    text2 = tesseract_ocr_extract(enhanced_pil, doc_type, lang)
                    results.append((len(text2), text2, "Basic Enhancement"))
                except Exception as e:
                    print(f"Approach 2 failed: {str(e)}")
                    results.append((0, "", "Basic Enhancement"))
                
                # Approach 3: Original image + OCR
                try:
                    text3 = tesseract_ocr_extract(img, doc_type, lang)
                    results.append((len(text3), text3, "Original Image"))
                except Exception as e:
                    print(f"Approach 3 failed: {str(e)}")
                    results.append((0, "", "Original Image"))
                
                # Select the best result
                if results:
                    best_result = max(results, key=lambda x: x[0])
                    page_text = best_result[1]
                    print(f"Selected best result for page {i+1}: {best_result[2]} with {best_result[0]} characters")
                else:
                    page_text = ""
                
                text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
            
            return text
        else:
            # Process image file
            print(f"Processing image: {file_path}")
            
            # Try multiple preprocessing approaches
            results = []
            
            # Approach 1: Document scanning + preprocessing + OCR
            try:
                image = Image.open(file_path)
                preprocessed_img = preprocess_image_for_ocr(image, doc_type)
                text1 = tesseract_ocr_extract(preprocessed_img, doc_type, lang)
                results.append((len(text1), text1, "Document Scanning"))
            except Exception as e:
                print(f"Approach 1 failed: {str(e)}")
                results.append((0, "", "Document Scanning"))
            
            # Approach 2: Basic preprocessing + OCR
            try:
                image = Image.open(file_path)
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                enhanced_pil = Image.fromarray(enhanced)
                
                text2 = tesseract_ocr_extract(enhanced_pil, doc_type, lang)
                results.append((len(text2), text2, "Basic Enhancement"))
            except Exception as e:
                print(f"Approach 2 failed: {str(e)}")
                results.append((0, "", "Basic Enhancement"))
            
            # Approach 3: Original image + OCR
            try:
                image = Image.open(file_path)
                text3 = tesseract_ocr_extract(image, doc_type, lang)
                results.append((len(text3), text3, "Original Image"))
            except Exception as e:
                print(f"Approach 3 failed: {str(e)}")
                results.append((0, "", "Original Image"))
            
            # Print results and select best one
            print("\nTesseract OCR Results:")
            for i, (length, text, method) in enumerate(results, 1):
                print(f"Approach {i} ({method}): {length} chars")
                if text:
                    print(f"  Sample: {text[:50]}...")
            
            # Select the result with the most text
            if results:
                best_result = max(results, key=lambda x: x[0])
                print(f"\nSelected best result: {best_result[2]} with {best_result[0]} characters")
                text = best_result[1]
            else:
                text = ""
            
            # If text is too short, it might indicate OCR issues
            if len(text) < 20:
                print(f"WARNING: Extracted very little text ({len(text)} chars). This might indicate OCR issues.")
            
            return text.strip()
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tesseract_ocr.py <image_path> [document_type]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    text = ocr_file_tesseract(image_path, doc_type)
    
    print("\n=== EXTRACTED TEXT ===")
    print(text)
    print("=== END OF TEXT ===")
