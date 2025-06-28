# Document Understanding Prototype

This project builds a document understanding pipeline using enhanced Tesseract OCR with document scanning and Google Gemini (free-tier) to extract structured data from various document types:

- **Driving Licenses**
- **Shop Receipts**
- **Resumes (CVs)**

## Features

- Enhanced OCR via Tesseract with document scanning preprocessing
- Multiple OCR approaches with automatic selection of best results
- Document-specific preprocessing optimizations
- Parsing prompts via Google Gemini API
- Field validators for quality checks
- Modular: separate modules for OCR, LLM, parsers, validators, and utilities
- CLI interface for batch processing

## Setup

1. Install Tesseract-OCR on your system and ensure `tesseract` is in your PATH.
2. Clone the repo.
3. Create a Python virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
4. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Set up your `.env` file with the necessary environment variables, such as your Google API key for Gemini parsing.

## Execution

To process documents, use the CLI interface:

```bash
python main.py --input_dir "path/to/documents" --doc_type "document_type" --output_dir "path/to/outputs"
```

Replace `document_type` with `driving_license`, `shop_receipt`, or `resume` as needed.

## Usage

```bash
python main.py --input_dir data/driving_license --doc_type driving_license --output_dir outputs/driving_license
```  
Replace `doc_type` with `shop_receipt` or `resume` as needed.

Outputs will be JSON files per input document in the specified output directory.

## Sample Output

Here is an example of a JSON output for a processed document:

```json
{
  "full_name": "John Doe",
  "email": "john.doe@example.com",
  "phone_number": "123-456-7890",
  "skills": ["Python", "Machine Learning", "Data Analysis"],
  "work_experience": [
    {
      "company": "Tech Solutions Inc.",
      "role": "Software Engineer",
      "start_date": "2018-06-01",
      "end_date": "2021-08-31"
    }
  ],
  "education": [
    {
      "institution": "University of Technology",
      "degree": "Bachelor of Science in Computer Science",
      "graduation_year": 2018
    }
  ]
}
```
