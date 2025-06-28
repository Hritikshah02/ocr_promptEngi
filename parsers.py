import os
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from models import DrivingLicense, ShopReceipt, Resume
from llm import call_gemini

# Template prompts
DRIVING_LICENSE_PROMPT = '''Extract the following fields from the document text and return them in a valid JSON format with these exact field names:
{
  "name": "Full Name",
  "date_of_birth": "YYYY-MM-DD",
  "license_number": "License Number",
  "issuing_state": "State Name",
  "expiry_date": "YYYY-MM-DD"
}

IMPORTANT: For Irish driving licenses, the fields are numbered as follows:
1. Surname (last name)
2. First name(s)
3. Date of birth (DD.MM.YY format)
4a. Date of issue
4b. Date of expiry
4c. Issuing authority
4d. or 5. License number
8. Address
9. Categories of vehicles

The full name should be constructed as "First name(s) Surname".

Ensure all dates are in ISO format (YYYY-MM-DD) and all field names match exactly as shown above.
If a field is not found, use null or empty string as appropriate.
''' 

SHOP_RECEIPT_PROMPT = '''Extract the following fields from the receipt and return them in a valid JSON format with these exact field names:
{
  "merchant_name": "Store Name",
  "total_amount": 0.00,
  "date_of_purchase": "YYYY-MM-DD",
  "payment_method": "Method",
  "items": [
    {
      "name": "Item Name",
      "quantity": 0,
      "price": 0.00
    }
  ]
}

Ensure all dates are in ISO format (YYYY-MM-DD), all numbers are properly formatted, and all field names match exactly as shown above.
If a field is not found, use null or empty values as appropriate.
''' 

RESUME_PROMPT = '''Extract the following fields from the resume and return them in a valid JSON format with these exact field names:
{
  "full_name": "Person's Name",
  "email": "email@example.com",
  "phone_number": "Phone Number",
  "skills": ["Skill 1", "Skill 2"],
  "work_experience": [
    {
      "company": "Company Name",
      "role": "Job Title",
      "start_date": "YYYY-MM-DD",
      "end_date": "YYYY-MM-DD"
    }
  ],
  "education": [
    {
      "institution": "School Name",
      "degree": "Degree Name",
      "graduation_year": 2020
    }
  ]
}

Ensure all dates are in ISO format (YYYY-MM-DD) and all field names match exactly as shown above.
If a field is not found, use null or empty values as appropriate.
''' 

def parse_with_gemini(text: str, doc_type: str) -> dict:
    """
    Parse document text using Gemini API based on document type.
    
    Args:
        text: OCR text from the document
        doc_type: Type of document ('driving_license', 'shop_receipt', or 'resume')
        
    Returns:
        Dictionary with extracted fields
    """
    # Select the appropriate prompt based on document type
    if doc_type == "driving_license":
        prompt_template = DRIVING_LICENSE_PROMPT
    elif doc_type == "shop_receipt":
        prompt_template = SHOP_RECEIPT_PROMPT
    elif doc_type == "resume":
        prompt_template = RESUME_PROMPT
    else:
        raise ValueError(f"Unknown document type: {doc_type}")
    
    # Create a full prompt with the OCR text
    full_prompt = f"{prompt_template}\n\nOCR TEXT FROM DOCUMENT:\n{text}\n\nPlease extract the information from the above OCR text."
    
    try:
        # Call Gemini API
        response = call_gemini(full_prompt)
        print(f"Raw Gemini response: {response}")
        return response
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        # Return a fallback response with empty values
        if doc_type == "driving_license":
            return {
                "name": "",
                "date_of_birth": None,
                "license_number": "",
                "issuing_state": "",
                "expiry_date": None
            }
        elif doc_type == "shop_receipt":
            return {
                "merchant_name": "",
                "total_amount": 0.0,
                "date_of_purchase": None,
                "payment_method": "",
                "items": []
            }
        elif doc_type == "resume":
            return {
                "full_name": "",
                "email": "unknown@example.com",
                "phone_number": "",
                "skills": [],
                "work_experience": [],
                "education": []
            }
        return {}


def parse_irish_license_fields(text: str) -> dict:
    """Parse Irish driving license fields using regex patterns for numbered fields."""
    import re
    result = {
        "name": "",
        "date_of_birth": None,
        "license_number": "",
        "issuing_state": "IRELAND",  # Default for Irish licenses
        "expiry_date": None
    }
    
    # Extract name (field 1 and 2) - typically after '1.' or '1' and '2.' or '2'
    firstname = None
    lastname = None
    
    # Very precise pattern for Irish license format - limit to just the first word after the number
    # This prevents capturing extra text that might appear on the same line
    lastname_match = re.search(r'1[.,]\s*([A-Za-z]+)', text)
    firstname_match = re.search(r'2[.,]\s*([A-Za-z]+)', text)
    
    if firstname_match:
        firstname = firstname_match.group(1).strip()
        # Clean up common OCR errors in names
        firstname = re.sub(r'[^A-Za-z\s\-\']+', '', firstname)
        
    if lastname_match:
        lastname = lastname_match.group(1).strip()
        # Clean up common OCR errors in names
        lastname = re.sub(r'[^A-Za-z\s\-\']+', '', lastname)
    
    # Try alternative patterns if the standard ones didn't work
    if not firstname:
        # Try to find multi-word first names but with strict boundaries
        patterns = [
            # Look for 2-3 consecutive words after '2.'
            r'2[.,]\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,2})(?:\s*\d|\s*$|\s*[,.])',
            # Look for words between '2.' and next number/field
            r'2[.,]\s*([^\d\n]{2,20})(?:\s*\d|\s*$)'
        ]
        
        for pattern in patterns:
            alt_firstname_match = re.search(pattern, text)
            if alt_firstname_match:
                firstname = alt_firstname_match.group(1).strip()
                firstname = re.sub(r'[^A-Za-z\s\-\']+', '', firstname)
                # Limit to first 2-3 words to avoid capturing OCR artifacts
                firstname_words = firstname.split()
                if len(firstname_words) > 3:
                    firstname = ' '.join(firstname_words[:2])
                break
    
    if not lastname:
        # Try to find multi-word last names but with strict boundaries
        patterns = [
            # Look for 2-3 consecutive words after '1.'
            r'1[.,]\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,2})(?:\s*\d|\s*$|\s*[,.])',
            # Look for words between '1.' and next number/field
            r'1[.,]\s*([^\d\n]{2,20})(?:\s*\d|\s*$)'
        ]
        
        for pattern in patterns:
            alt_lastname_match = re.search(pattern, text)
            if alt_lastname_match:
                lastname = alt_lastname_match.group(1).strip()
                lastname = re.sub(r'[^A-Za-z\s\-\']+', '', lastname)
                # Limit to first 2-3 words to avoid capturing OCR artifacts
                lastname_words = lastname.split()
                if len(lastname_words) > 3:
                    lastname = ' '.join(lastname_words[:2])
                break
    
    # Construct the full name
    if firstname and lastname:
        result["name"] = f"{firstname} {lastname}"
    elif lastname:
        result["name"] = lastname
    elif firstname:
        result["name"] = firstname
    
    # If we still don't have a name, try a generic name pattern but with limits
    if not result.get("name"):
        full_name_match = re.search(r'name[:\s]+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})', text, re.IGNORECASE)
        if full_name_match:
            result["name"] = full_name_match.group(1).strip()
            result["name"] = re.sub(r'[^A-Za-z\s\-\']+', '', result["name"])
    
    # Extract date of birth (field 3) with multiple patterns
    dob_patterns = [
        # Standard field format DD.MM.YY
        r'3[,.]\s*(\d{2}[.,]\d{2}[.,]\d{2,4})',
        # Labeled format
        r'(?:birth|born|dob)[:\s]+(\d{2}[.,/\s]\d{2}[.,/\s]\d{2,4})',
        # ISO format YYYY-MM-DD
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # Any date-like pattern
        r'\b(\d{1,2}[.,/\s]\d{1,2}[.,/\s]\d{2,4})\b'
    ]
    
    for pattern in dob_patterns:
        dob_match = re.search(pattern, text, re.IGNORECASE)
        if dob_match:
            dob_str = dob_match.group(1).replace(',', '.').replace(' ', '')
            try:
                # Handle different date formats
                if '-' in dob_str:  # ISO format
                    year, month, day = dob_str.split('-')
                    result["date_of_birth"] = f"{year}-{month}-{day}"
                elif '/' in dob_str:  # MM/DD/YYYY or DD/MM/YYYY
                    parts = dob_str.split('/')
                    if len(parts) == 3:
                        if len(parts[2]) == 4:  # Assume it's DD/MM/YYYY for Irish licenses
                            day, month, year = parts
                        else:  # Assume DD/MM/YY
                            day, month, year = parts
                            if len(year) == 2:
                                year = '19' + year if int(year) > 50 else '20' + year
                        result["date_of_birth"] = f"{year}-{month}-{day}"
                else:  # Assume DD.MM.YY
                    parts = dob_str.split('.')
                    if len(parts) == 3:
                        day, month, year = parts
                        if len(year) == 2:
                            year = '19' + year if int(year) > 50 else '20' + year
                        result["date_of_birth"] = f"{year}-{month}-{day}"
                print(f"Parsed date of birth: {result['date_of_birth']} from {dob_str}")
                break
            except Exception as e:
                print(f"Could not parse date of birth: {dob_str}, error: {str(e)}")
                continue
    
    # Extract license number (field 4d or 5) with multiple patterns
    license_patterns = [
        r'(?:4d|5)[,.:]?\s*([A-Z0-9]{6,12})',  # Field 4d or 5 followed by alphanumeric
        r'(?:license|licence)\s*(?:no|number|#)?[:.\s]*([A-Z0-9]{6,12})',  # Labeled license number
        r'\b([A-Z0-9]{8})\b',  # Any 8-character alphanumeric string (common format)
        r'\b([A-Z]{1,3}[0-9]{5,6})\b',  # Common Irish format: 1-3 letters followed by 5-6 digits
        r'\b([0-9]{6}[A-Z]{1,2})\b'   # Alternative format: 6 digits followed by 1-2 letters
    ]
    
    for pattern in license_patterns:
        license_matches = re.findall(pattern, text, re.IGNORECASE)
        if license_matches:
            # Filter out matches that are likely not license numbers (too short, etc)
            valid_matches = [match for match in license_matches if len(match) >= 6]
            if valid_matches:
                result["license_number"] = valid_matches[0].strip()
                print(f"Found license number '{result['license_number']}' using pattern: {pattern}")
                break
            
    # Extract expiry date (field 4b) with multiple patterns
    expiry_patterns = [
        # Standard field format DD.MM.YY
        r'4b[,.]\s*(\d{2}[.,]\d{2}[.,]\d{2,4})',
        # Labeled format
        r'(?:expiry|expiration|valid until)[:\s]+(\d{2}[.,/\s]\d{2}[.,/\s]\d{2,4})',
        # ISO format YYYY-MM-DD
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # Any date-like pattern
        r'\b(\d{1,2}[.,/\s]\d{1,2}[.,/\s]\d{2,4})\b'
    ]
    
    expiry_date = None
    for pattern in expiry_patterns:
        expiry_match = re.search(pattern, text, re.IGNORECASE)
        if expiry_match:
            expiry_str = expiry_match.group(1).replace(',', '.').replace(' ', '')
            try:
                # Handle different date formats
                if '-' in expiry_str:  # ISO format
                    year, month, day = expiry_str.split('-')
                    result["expiry_date"] = f"{year}-{month}-{day}"
                elif '/' in expiry_str:  # MM/DD/YYYY or DD/MM/YYYY
                    parts = expiry_str.split('/')
                    if len(parts[2]) == 4:  # Assume it's MM/DD/YYYY or DD/MM/YYYY
                        # For Irish licenses, assume DD/MM/YYYY
                        day, month, year = parts
                        result["expiry_date"] = f"{year}-{month}-{day}"
                    else:  # Assume DD/MM/YY
                        day, month, year = parts
                        if len(year) == 2:
                            year = '20' + year  # Assume 20xx for 2-digit years
                        result["expiry_date"] = f"{year}-{month}-{day}"
                else:  # Assume DD.MM.YY
                    parts = expiry_str.split('.')
                    if len(parts) == 3:
                        day, month, year = parts
                        if len(year) == 2:
                            year = '20' + year  # Assume 20xx for 2-digit years
                        result["expiry_date"] = f"{year}-{month}-{day}"
                print(f"Parsed expiry date: {result['expiry_date']} from {expiry_str}")
                break
            except Exception as e:
                print(f"Could not parse expiry date: {expiry_str}, error: {str(e)}")
                continue
    
    return result

def parse_driving_license(text: str) -> DrivingLicense:
    """
    Parse driving license information from OCR text.
    """
    # Check if text is too short or empty
    if not text or len(text) < 20:
        print(f"WARNING: OCR text too short or empty: '{text}'")
        return DrivingLicense()
    
    # Check if this is an Irish driving license
    irish_keywords = ["CEADUNAS TIOMANA", "DRIVING LICENCE", "IRELAND", "EIRE"]
    is_irish = any(keyword in text.upper() for keyword in irish_keywords)
    
    if is_irish:
        print("Detected Irish driving license format with numbered fields")
        try:
            # Try specialized Irish license parser first
            result = parse_irish_license_fields(text)
            print(f"Irish license parser extracted: {result}")
            
            # Validate the extracted data
            missing_fields = []
            if not result.get("name"):
                missing_fields.append("name")
            if not result.get("date_of_birth"):
                missing_fields.append("date_of_birth")
            if not result.get("license_number"):
                missing_fields.append("license_number")
            if not result.get("expiry_date"):
                missing_fields.append("expiry_date")
                
            # If we're missing critical fields, try to fill them with Gemini if possible
            if missing_fields:
                print(f"Irish license parser missing fields: {', '.join(missing_fields)}, attempting to fill with Gemini")
                try:
                    gemini_result = parse_with_gemini(text, "driving_license")
                    
                    # Fill in missing fields from Gemini result
                    for field in missing_fields:
                        if gemini_result.get(field):
                            result[field] = gemini_result[field]
                            print(f"Filled missing field '{field}' with Gemini data")
                except Exception as gemini_error:
                    print(f"Gemini API error: {str(gemini_error)}")
                    # Continue with what we have - don't let Gemini errors stop us
            
            # Convert to DrivingLicense model with fallbacks for missing fields
            return DrivingLicense(
                name=result.get("name", ""),
                date_of_birth=result.get("date_of_birth"),
                license_number=result.get("license_number", ""),
                issuing_state=result.get("issuing_state", "IRELAND"),
                expiry_date=result.get("expiry_date")
            )
        except Exception as e:
            print(f"Error in Irish license parser: {str(e)}")
            # Return what we can extract with default values
            return DrivingLicense(
                name="",
                date_of_birth=None,
                license_number="",
                issuing_state="IRELAND",
                expiry_date=None
            )
    
    # Create a more detailed prompt with the OCR text
    full_prompt = f"{DRIVING_LICENSE_PROMPT}\n\nOCR TEXT FROM DRIVING LICENSE:\n{text}\n\nPlease extract the information from the above OCR text."
    
    try:
        resp = call_gemini(full_prompt)
        # Print response for debugging
        print(f"Raw Gemini response: {resp}")

        # Handle missing fields with defaults
        if 'name' not in resp:
            resp['name'] = ""
        if 'date_of_birth' not in resp:
            resp['date_of_birth'] = None
        if 'license_number' not in resp:
            resp['license_number'] = ""
        if 'issuing_state' not in resp:
            resp['issuing_state'] = ""
        if 'expiry_date' not in resp:
            resp['expiry_date'] = None

        # Process dates safely
        try:
            if resp.get('date_of_birth'):
                resp['date_of_birth'] = datetime.fromisoformat(resp['date_of_birth']).date()
        except (ValueError, TypeError):
            print(f"Invalid date_of_birth format: {resp.get('date_of_birth')}")
            resp['date_of_birth'] = None

        try:
            if resp.get('expiry_date'):
                resp['expiry_date'] = datetime.fromisoformat(resp['expiry_date']).date()
        except (ValueError, TypeError):
            print(f"Invalid expiry_date format: {resp.get('expiry_date')}")
            resp['expiry_date'] = None

        return DrivingLicense(**resp)
    except Exception as e:
        print(f"Error parsing driving license with Gemini API: {str(e)}")
        # Return empty object with proper default values, not error messages
        return DrivingLicense(
            name="",
            date_of_birth=None,
            license_number="",
            issuing_state="",
            expiry_date=None
        )


def parse_shop_receipt_with_regex(text: str) -> dict:
    """
    Parse shop receipt using regex patterns to extract key information.
    This is used as a fallback when Gemini API fails.
    
    Args:
        text: OCR text from receipt
        
    Returns:
        Dictionary with extracted receipt information
    """
    result = {
        "merchant_name": "",
        "total_amount": 0.0,
        "date_of_purchase": None,
        "payment_method": "",
        "items": []
    }
    
    # Clean up text - remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('$', ' $ ').replace(':', ' : ')
    
    # Extract merchant name - usually at the top of receipt
    # Look for common patterns in receipt headers
    merchant_patterns = [
        # Look for store name at the beginning of receipt
        r'^[\s\n]*([A-Z][A-Za-z0-9\s&\.,]{2,30})\s',
        # Look for common receipt header patterns
        r'(?:WELCOME TO|RECEIPT|STORE:)\s+([A-Za-z0-9\s&\.,]{2,30})\b',
        # Look for capitalized words at the beginning
        r'^[\s\n]*([A-Z][A-Z\s&]{2,30})\b',
        # Look for name followed by address pattern
        r'([A-Za-z0-9\s&\.,]{3,30})\n[0-9]+\s+[A-Za-z\s]+\s+(?:ST|AVE|BLVD|RD|ROAD|STREET)',
        # Walmart specific pattern
        r'(?:Walmart|WALMART)'
    ]
    
    for pattern in merchant_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result["merchant_name"] = match.group(1).strip() if len(match.groups()) > 0 else "Walmart"
            break
    
    # Extract total amount
    total_patterns = [
        r'(?:TOTAL|Total|total|AMOUNT|Amount|DUE|Due|BALANCE|Balance)[\s:]*[$]?\s*([0-9]+\.[0-9]{2})',
        r'(?:TOTAL|Total|total|AMOUNT|Amount|DUE|Due|BALANCE|Balance)[\s:]*[$]?\s*([0-9]+,[0-9]{3}\.[0-9]{2})',
        r'[$]\s*([0-9]+\.[0-9]{2})\s*(?:USD)?$'
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                amount_str = match.group(1).replace(',', '')
                result["total_amount"] = float(amount_str)
                break
            except (ValueError, TypeError):
                continue
    
    # Extract date
    date_patterns = [
        # MM/DD/YYYY
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        # DD/MM/YYYY
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        # YYYY/MM/DD
        r'(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})',
        # Text date format
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        # Date with time
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})\s+\d{1,2}:\d{2}'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(0)
            try:
                # Try multiple date formats
                for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',
                           '%m.%d.%Y', '%d.%m.%Y', '%Y.%m.%d', '%b %d, %Y', '%B %d, %Y']:
                    try:
                        date_obj = datetime.strptime(date_str, fmt).date()
                        result["date_of_purchase"] = date_obj.isoformat()
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
            if result["date_of_purchase"]:
                break
    
    # Extract payment method
    payment_patterns = [
        r'(?:PAYMENT|Payment|PAID|Paid|TENDER|Tender)[\s:]*(?:BY|Via|via|With|with)?[\s:]*([A-Za-z\s]{2,20})',
        r'(?:CARD|Card|CREDIT|Credit|DEBIT|Debit|CASH|Cash|CHECK|Check)[\s:]*([A-Za-z\s]{2,20})',
        r'(?:VISA|MASTERCARD|AMEX|DISCOVER|CASH|CHECK)'
    ]
    
    for pattern in payment_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["payment_method"] = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
            break
    
    # Try to extract items - this is more complex and may not always work well with regex
    # Look for patterns like "1 x Item $10.99" or "Item Name 2 $5.99"
    item_patterns = [
        r'(\d+)\s*[xX]\s*([A-Za-z0-9\s&\.,\-\(\)]{3,30})\s*[$]?\s*(\d+\.\d{2})',
        r'([A-Za-z0-9\s&\.,\-\(\)]{3,30})\s+(\d+)\s+[$]?\s*(\d+\.\d{2})'
    ]
    
    for pattern in item_patterns:
        for match in re.finditer(pattern, text):
            try:
                if len(match.groups()) == 3:
                    # First pattern: quantity, name, price
                    item = {
                        "name": match.group(2).strip(),
                        "quantity": int(match.group(1)),
                        "price": float(match.group(3))
                    }
                    result["items"].append(item)
                else:
                    # Second pattern: name, quantity, price
                    item = {
                        "name": match.group(1).strip(),
                        "quantity": int(match.group(2)),
                        "price": float(match.group(3))
                    }
                    result["items"].append(item)
            except (ValueError, IndexError):
                continue
    
    return result

def parse_shop_receipt(text: str) -> ShopReceipt:
    """
    Parse shop receipt information from OCR text.
    Uses a combination of Gemini API and regex fallback.
    
    Args:
        text: OCR text from receipt
        
    Returns:
        ShopReceipt object with extracted information
    """
    # Validate OCR text
    if not text or len(text) < 20:
        print(f"WARNING: OCR text too short or empty: '{text}'")
        # Return empty object with default values
        return ShopReceipt(
            merchant_name="OCR_FAILED",
            total_amount=0.0
        )
    
    # First try regex parsing as it's faster and doesn't use API quota
    regex_result = parse_shop_receipt_with_regex(text)
    has_valid_regex_result = regex_result["merchant_name"] and regex_result["total_amount"] > 0
    
    if has_valid_regex_result:
        print("Successfully parsed receipt with regex")
        try:
            # Convert date string to date object if it exists
            if regex_result["date_of_purchase"] and isinstance(regex_result["date_of_purchase"], str):
                regex_result["date_of_purchase"] = datetime.fromisoformat(regex_result["date_of_purchase"]).date()
            return ShopReceipt(**regex_result)
        except Exception as e:
            print(f"Error converting regex result to ShopReceipt: {str(e)}")
            # Continue to Gemini API as fallback
    
    # Create a more detailed prompt with the OCR text
    full_prompt = f"{SHOP_RECEIPT_PROMPT}\n\nOCR TEXT FROM RECEIPT:\n{text}\n\nPlease extract the information from the above OCR text."
    
    try:
        resp = call_gemini(full_prompt)
        # Print response for debugging
        print(f"Raw Gemini response: {resp}")
        
        # Handle missing fields with defaults
        if 'merchant_name' not in resp:
            # Use regex result if available
            resp['merchant_name'] = regex_result["merchant_name"] if regex_result["merchant_name"] else ""
            
        if 'payment_method' not in resp:
            resp['payment_method'] = regex_result["payment_method"] if regex_result["payment_method"] else ""
            
        if 'items' not in resp or not isinstance(resp['items'], list):
            resp['items'] = regex_result["items"] if regex_result["items"] else []
            
        # Handle date safely
        try:
            if 'date_of_purchase' in resp and resp['date_of_purchase']:
                resp['date_of_purchase'] = datetime.fromisoformat(resp['date_of_purchase']).date()
            elif regex_result["date_of_purchase"]:
                # Use regex result if Gemini didn't provide a date
                if isinstance(regex_result["date_of_purchase"], str):
                    resp['date_of_purchase'] = datetime.fromisoformat(regex_result["date_of_purchase"]).date()
                else:
                    resp['date_of_purchase'] = regex_result["date_of_purchase"]
            else:
                resp['date_of_purchase'] = None
        except (ValueError, TypeError):
            print(f"Invalid date_of_purchase format: {resp.get('date_of_purchase')}")
            resp['date_of_purchase'] = None
        
        # Handle numeric fields safely
        try:
            resp['total_amount'] = float(resp.get('total_amount', 0))
            # If Gemini returned 0 but regex found a value, use the regex value
            if resp['total_amount'] == 0 and regex_result["total_amount"] > 0:
                resp['total_amount'] = regex_result["total_amount"]
        except (ValueError, TypeError):
            resp['total_amount'] = regex_result["total_amount"] if regex_result["total_amount"] > 0 else 0.0
        
        # Process items safely
        for item in resp['items']:
            try:
                item['quantity'] = int(item.get('quantity', 1))
            except (ValueError, TypeError):
                item['quantity'] = 1
                
            try:
                item['price'] = float(item.get('price', 0))
            except (ValueError, TypeError):
                item['price'] = 0.0
                
            if 'name' not in item:
                item['name'] = "Unknown Item"
        
        return ShopReceipt(**resp)
    except Exception as e:
        print(f"Error parsing shop receipt with Gemini: {str(e)}")
        
        # If regex parsing was successful, use that instead
        if has_valid_regex_result:
            print("Falling back to regex parsing result")
            try:
                if regex_result["date_of_purchase"] and isinstance(regex_result["date_of_purchase"], str):
                    regex_result["date_of_purchase"] = datetime.fromisoformat(regex_result["date_of_purchase"]).date()
                return ShopReceipt(**regex_result)
            except Exception as e2:
                print(f"Error using regex fallback: {str(e2)}")
        
        # Return empty object with default values and error message
        return ShopReceipt(
            merchant_name=f"ERROR: {str(e)[:30]}...",
            total_amount=0.0
        )


def parse_resume(text: str) -> Resume:
    # Validate OCR text
    if not text or len(text) < 20:
        print(f"WARNING: OCR text too short or empty: '{text}'")
        # Return empty object with default values
        return Resume(
            full_name="OCR_FAILED",
            email="ocr.failed@example.com",
            phone_number="",
            skills=[],
            work_experience=[],
            education=[]
        )
    
    # Create a more detailed prompt with the OCR text
    full_prompt = f"{RESUME_PROMPT}\n\nOCR TEXT FROM RESUME:\n{text}\n\nPlease extract the information from the above OCR text."
    
    try:
        resp = call_gemini(full_prompt)
        print(f"Raw Gemini response: {resp}")
        # Robustly clean and default all fields
        cleaned = {}
        cleaned['full_name'] = resp.get('full_name', '') or ''
        email = resp.get('email', 'unknown@example.com')
        # Patch invalid email
        if not isinstance(email, str) or '@' not in email:
            email = 'unknown@example.com'
        cleaned['email'] = email
        cleaned['phone_number'] = resp.get('phone_number', '') or ''
        skills = resp.get('skills', [])
        if not isinstance(skills, list):
            skills = []
        cleaned['skills'] = [str(s) for s in skills if isinstance(s, str)]
        # Work experience
        work_experience = resp.get('work_experience', [])
        if not isinstance(work_experience, list):
            work_experience = []
        cleaned['work_experience'] = []
        for we in work_experience:
            if not isinstance(we, dict):
                continue
            entry = {
                'company': we.get('company', '') or '',
                'role': we.get('role', '') or '',
                'start_date': None,
                'end_date': None
            }
            try:
                if we.get('start_date'):
                    entry['start_date'] = datetime.fromisoformat(we['start_date']).date()
            except Exception as ex:
                print(f"Invalid work_experience.start_date: {we.get('start_date')}")
            try:
                if we.get('end_date'):
                    entry['end_date'] = datetime.fromisoformat(we['end_date']).date()
            except Exception as ex:
                print(f"Invalid work_experience.end_date: {we.get('end_date')}")
            cleaned['work_experience'].append(entry)
        # Education
        education = resp.get('education', [])
        if not isinstance(education, list):
            education = []
        cleaned['education'] = []
        for edu in education:
            if not isinstance(edu, dict):
                continue
            entry = {
                'institution': edu.get('institution', '') or '',
                'degree': edu.get('degree', '') or '',
                'graduation_year': None
            }
            try:
                if edu.get('graduation_year'):
                    entry['graduation_year'] = int(edu['graduation_year'])
            except Exception as ex:
                print(f"Invalid education.graduation_year: {edu.get('graduation_year')}")
            cleaned['education'].append(entry)
        try:
            return Resume(**cleaned)
        except Exception as e:
            import traceback
            print("Validation error in Resume model:")
            print(traceback.format_exc())
            print(f"Cleaned dict: {cleaned}")
            return Resume(
                full_name=f"ERROR: {str(e)[:100]}",
                email="error@example.com",
                phone_number="",
                skills=[],
                work_experience=[],
                education=[]
            )
    except Exception as e:
        import traceback
        print("Exception in parse_resume:")
        print(traceback.format_exc())
        return Resume(
            full_name=f"ERROR: {str(e)[:100]}",
            email="error@example.com",
            phone_number="",
            skills=[],
            work_experience=[],
            education=[]
        )
