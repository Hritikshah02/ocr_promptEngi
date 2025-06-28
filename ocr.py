import os
import pytesseract
# allow overriding the tesseract executable path via TESSERACT_CMD env var (default install path)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

def detect_receipt_area(img_cv):
    """
    Detect and extract just the receipt area from an image
    
    Args:
        img_cv: OpenCV image
        
    Returns:
        Cropped image containing just the receipt or original if no clear receipt is detected
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return original image
    if not contours:
        return img_cv
    
    # Find the largest contour by area - likely to be the receipt
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    # If no significant contour found, return original image
    if largest_contour is None or cv2.contourArea(largest_contour) < 1000:  # Minimum area threshold
        return img_cv
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Check if the detected area is reasonable (at least 30% of the image)
    img_area = img_cv.shape[0] * img_cv.shape[1]
    contour_area = w * h
    
    if contour_area < 0.3 * img_area:
        # Try another approach - find all white/light areas
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Combine all contours to get the receipt area
            all_contours = np.vstack([contour for contour in contours])
            x, y, w, h = cv2.boundingRect(all_contours)
    
    # Add a small margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img_cv.shape[1] - x, w + 2*margin)
    h = min(img_cv.shape[0] - y, h + 2*margin)
    
    # Crop the image to the receipt area
    cropped = img_cv[y:y+h, x:x+w]
    
    # If cropped area is too small, return original
    if cropped.size == 0 or cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return img_cv
    
    return cropped

def preprocess_receipt(img):
    """
    Special preprocessing for shop receipts
    
    Args:
        img: PIL Image of receipt
        
    Returns:
        Preprocessed PIL Image optimized for receipt OCR
    """
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Detect and crop to just the receipt area
    img_cv = detect_receipt_area(img_cv)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while removing noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 15, 2)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(morph)
    
    # Convert back to PIL Image
    return Image.fromarray(enhanced)

def preprocess_image(img, technique='adaptive'):
    """
    Apply various preprocessing techniques to improve OCR accuracy
    
    Args:
        img: PIL Image to preprocess
        technique: Preprocessing technique to apply
        
    Returns:
        Preprocessed PIL Image
    """
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    if technique == 'adaptive':
        # Adaptive thresholding
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_processed = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif technique == 'otsu':
        # Otsu's thresholding
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif technique == 'canny':
        # Canny edge detection
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_processed = cv2.Canny(img_gray, 100, 200)
    elif technique == 'dilate_erode':
        # Dilation followed by erosion (closing operation)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        img_processed = cv2.dilate(img_gray, kernel, iterations=1)
        img_processed = cv2.erode(img_processed, kernel, iterations=1)
    elif technique == 'noise_removal':
        # Noise removal with bilateral filter
        img_processed = cv2.bilateralFilter(img_cv, 9, 75, 75)
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    elif technique == 'deskew':
        # Deskew the image
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(img_gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img_gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_processed = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    elif technique == 'id_card_enhance':
        # Special processing for ID cards/licenses
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_contrast = clahe.apply(img_gray)
        # Denoise
        img_denoised = cv2.fastNlMeansDenoising(img_contrast, None, 10, 7, 21)
        # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_processed = cv2.filter2D(img_denoised, -1, kernel)
    else:
        # Default: just grayscale
        img_processed = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_cv = cv2.erode(img_cv, kernel, iterations=1)
        # Invert back for OCR
        img_cv = 255 - img_cv
    
    # Convert back to PIL Image
    enhanced_img = Image.fromarray(img_cv)
    return enhanced_img

def try_multiple_ocr_approaches(img, lang='eng', doc_type=None):
    """
    Try multiple preprocessing and OCR approaches to get the best result
    
    Args:
        img: PIL Image to process
        lang: OCR language(s)
        doc_type: Type of document ('driving_license', 'shop_receipt', or 'resume')
        
    Returns:
        Best OCR text result
    """
    results = []
    
    # Approach 1: Original image
    text1 = pytesseract.image_to_string(img, lang=lang)
    results.append((len(text1), text1))
    
    # Document-specific preprocessing
    if doc_type == 'shop_receipt':
        # Special receipt preprocessing
        img_receipt = preprocess_receipt(img)
        text_receipt = pytesseract.image_to_string(img_receipt, lang=lang)
        results.append((len(text_receipt), text_receipt))
        
        # Try receipt with different PSM modes optimized for receipts
        for psm in [4, 6]:  # 4=single column, 6=single block of text
            config = f'--oem 1 --psm {psm}'
            text_receipt_psm = pytesseract.image_to_string(img_receipt, lang=lang, config=config)
            results.append((len(text_receipt_psm), text_receipt_psm))
    
    # Approach 2: Enhanced contrast
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(2.0)  # Increase contrast
    text2 = pytesseract.image_to_string(img_contrast, lang=lang)
    results.append((len(text2), text2))
    
    # Approach 3: Sharpened image
    img_sharp = img.filter(ImageFilter.SHARPEN)
    text3 = pytesseract.image_to_string(img_sharp, lang=lang)
    results.append((len(text3), text3))
    
    # Try all preprocessing techniques
    techniques = ['adaptive', 'otsu', 'dilate_erode']
    if doc_type != 'shop_receipt':  # These techniques work better for IDs than receipts
        techniques.extend(['canny', 'id_card_enhance'])
        
    for technique in techniques:
        img_preprocessed = preprocess_image(img, technique=technique)
        text_preprocessed = pytesseract.image_to_string(img_preprocessed, lang=lang)
        results.append((len(text_preprocessed), text_preprocessed))
    
    # Try with multiple languages
    text_multi_lang = pytesseract.image_to_string(img, lang='eng+fra+deu')
    results.append((len(text_multi_lang), text_multi_lang))
    
    # Try with different OCR configurations
    custom_config = r'--oem 1 --psm 3'
    text_custom = pytesseract.image_to_string(img, lang=lang, config=custom_config)
    results.append((len(text_custom), text_custom))
    
    # Try with different PSM modes for problematic images
    psm_modes = [6, 11, 12]  # Single block, single line, sparse text
    if doc_type == 'shop_receipt':
        psm_modes = [3, 4, 6]  # Full page, single column, single block
        
    for psm in psm_modes:
        config = f'--oem 1 --psm {psm}'
        text_psm = pytesseract.image_to_string(img, lang=lang, config=config)
        results.append((len(text_psm), text_psm))
    
    # Sort by text length and return the best result
    results.sort(reverse=True)
    
    # Debug: Print top 3 results
    for i, (length, text) in enumerate(results[:3]):
        print(f"OCR approach #{i+1}: {length} chars, sample: {text[:30]}...")
    
    return results[0][1]

def ocr_file(file_path, lang='eng', doc_type=None):
    """
    Extract text from an image or PDF file using OCR
    
    Args:
        file_path: Path to the image or PDF file
        lang: OCR language(s)
        doc_type: Type of document ('driving_license', 'shop_receipt', or 'resume')
        
    Returns:
        Extracted text from the document
    """
    print(f"Processing file: {file_path}")
    
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
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300, poppler_path=r'C:\Program Files\poppler-23.11.0\Library\bin')
            text = ""
            
            # Process each page
            for i, img in enumerate(images):
                print(f"Processing PDF page {i+1}")
                page_text = try_multiple_ocr_approaches(img, lang=lang, doc_type=doc_type)
                text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
            
            return text
        else:
            # Process image file
            print(f"Processing image: {file_path}")
            img = Image.open(file_path)
            text = try_multiple_ocr_approaches(img, lang=lang, doc_type=doc_type)
            print(f"Image extracted {len(text)} characters")
            print(f"OCR Text Sample (first 100 chars): {text[:100]}...")
            
            # If text is too short, it might indicate OCR issues
            if len(text) < 20:
                print(f"WARNING: OCR extracted very little text ({len(text)} chars). This might indicate OCR issues.")
                
            return text.strip()
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return ""
