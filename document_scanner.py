"""
Document Scanner Module using docscan library for automatic document cropping
"""
import cv2
import numpy as np
from PIL import Image
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os


def order_points(pts):
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    # Sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    # Grab the left-most and right-most points from the sorted x-coordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # Sort the left-most coordinates according to their y-coordinates
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    # Compute the distance between all points in the right-most coordinates
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    
    # Return the coordinates in top-left, top-right, bottom-right, bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    """
    Apply perspective transformation to get a top-down view of the document
    """
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def detect_document_contour(image):
    """
    Detect the document contour in the image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # If our approximated contour has four points, then we can assume
        # that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        # If no 4-point contour is found, use the largest contour
        if cnts:
            screenCnt = cnts[0]
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(screenCnt)
            screenCnt = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        else:
            # If no contours found, return None
            return None
    
    return screenCnt.reshape(4, 2)


def scan_document(image_path, output_path=None, enhance=True):
    """
    Scan and crop a document from an image
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the scanned document (optional)
        enhance: Whether to apply image enhancement after scanning
        
    Returns:
        PIL Image of the scanned document
    """
    print(f"Scanning document: {image_path}")
    
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        # Assume it's already a numpy array
        image = image_path
    
    # Store original dimensions
    orig = image.copy()
    
    # Resize image for better processing (keep aspect ratio)
    height, width = image.shape[:2]
    if width > 1000:
        ratio = 1000.0 / width
        image = imutils.resize(image, width=1000)
    else:
        ratio = 1.0
    
    # Detect document contour
    document_contour = detect_document_contour(image)
    
    if document_contour is None:
        print("WARNING: Could not detect document contour. Using original image.")
        # Convert to PIL and return
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(orig_rgb)
        if output_path:
            pil_image.save(output_path)
        return pil_image
    
    # Scale the contour back to original image size
    document_contour = document_contour / ratio
    
    # Apply perspective transformation
    warped = four_point_transform(orig, document_contour)
    
    if enhance:
        # Convert to grayscale for enhancement
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text clarity
        warped_enhanced = cv2.adaptiveThreshold(
            warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
        )
        
        # Convert back to RGB for PIL
        warped_rgb = cv2.cvtColor(warped_enhanced, cv2.COLOR_GRAY2RGB)
    else:
        # Convert BGR to RGB for PIL
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(warped_rgb)
    
    # Save if output path is provided
    if output_path:
        pil_image.save(output_path)
        print(f"Scanned document saved to: {output_path}")
    
    return pil_image


def scan_receipt(image_path, output_path=None):
    """
    Specialized function for scanning receipts with receipt-specific optimizations
    
    Args:
        image_path: Path to the input receipt image
        output_path: Path to save the scanned receipt (optional)
        
    Returns:
        PIL Image of the scanned receipt
    """
    print(f"Scanning receipt: {image_path}")
    
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    else:
        image = image_path
    
    orig = image.copy()
    
    # Resize for processing
    height, width = image.shape[:2]
    if width > 800:
        ratio = 800.0 / width
        image = imutils.resize(image, width=800)
    else:
        ratio = 1.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply edge detection with lower thresholds for receipts
    edged = cv2.Canny(filtered, 50, 150)
    
    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Filter contours by area and aspect ratio (receipts are usually tall and narrow)
    receipt_contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:  # Minimum area threshold
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = h / float(w)
            
            # Receipts typically have aspect ratio > 1.5 (taller than wide)
            if aspect_ratio > 1.2:
                receipt_contours.append((area, c))
    
    if not receipt_contours:
        print("WARNING: Could not detect receipt contour. Using original image.")
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(orig_rgb)
        if output_path:
            pil_image.save(output_path)
        return pil_image
    
    # Sort by area and take the largest
    receipt_contours.sort(key=lambda x: x[0], reverse=True)
    largest_contour = receipt_contours[0][1]
    
    # Try to approximate to 4 points
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    if len(approx) == 4:
        document_contour = approx.reshape(4, 2)
    else:
        # Use bounding rectangle if we can't get 4 points
        x, y, w, h = cv2.boundingRect(largest_contour)
        document_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    
    # Scale back to original size
    document_contour = document_contour / ratio
    
    # Apply perspective transformation
    warped = four_point_transform(orig, document_contour)
    
    # Apply receipt-specific enhancement
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    warped_enhanced = clahe.apply(warped_gray)
    
    # Apply slight Gaussian blur to reduce noise
    warped_enhanced = cv2.GaussianBlur(warped_enhanced, (1, 1), 0)
    
    # Convert to RGB
    warped_rgb = cv2.cvtColor(warped_enhanced, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(warped_rgb)
    
    # Save if output path is provided
    if output_path:
        pil_image.save(output_path)
        print(f"Scanned receipt saved to: {output_path}")
    
    return pil_image


if __name__ == "__main__":
    # Test the document scanner
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "scanned_output.jpg"
        
        try:
            scanned = scan_document(input_path, output_path)
            print(f"Document scanned successfully! Output saved to: {output_path}")
        except Exception as e:
            print(f"Error scanning document: {str(e)}")
    else:
        print("Usage: python document_scanner.py <input_image_path> [output_image_path]")
