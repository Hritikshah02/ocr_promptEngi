import argparse
from config import GOOGLE_API_KEY
from utils import process_documents


def main():
    parser = argparse.ArgumentParser(description='Document Understanding CLI')
    parser.add_argument('--input_dir', required=True, help='Directory of documents')
    parser.add_argument('--doc_type', required=True, choices=['driving_license', 'shop_receipt', 'resume'])
    parser.add_argument('--output_dir', required=True, help='Where to write JSON outputs')
    args = parser.parse_args()
    process_documents(args.input_dir, args.doc_type, args.output_dir)

if __name__ == '__main__':
    main()
