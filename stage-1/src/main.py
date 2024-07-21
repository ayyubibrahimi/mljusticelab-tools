import os
import json
import logging
from classify import process_pdf as classify_process_pdf
from ocr_headers_optimized import DocClient, update_page_keys_in_json, reformat_json_structure, getcreds

def filter_pages(classification_json, ocr_results, excluded_types):
    excluded_pages = {classification['page_number'] for classification in classification_json['messages'] if classification['type'] in excluded_types}
    filtered_ocr_results = {
        "messages": [
            {**page, 'page_content': ''} if page['page_number'] in excluded_pages else page
            for page in ocr_results['messages']
        ]
    }
    return filtered_ocr_results

def main():
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    input_directory = "../data/input/bias-2010-2014"
    output_directory_classify = "../data/output/classify/bias-2010-2014"
    output_directory_ocr = "../data/output/ocr"

    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    excluded_types = ['picture','0']

    for root, dirs, files in os.walk(input_directory):
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]
        for file in pdf_files:
            file_path = os.path.join(root, file)
            logging.info(f"Processing {file_path}")

            relative_path_classify = os.path.relpath(root, input_directory)
            output_subdir_classify = os.path.join(output_directory_classify, relative_path_classify)
            output_subdir_ocr = os.path.join(output_directory_ocr, relative_path_classify)
            os.makedirs(output_subdir_classify, exist_ok=True)
            os.makedirs(output_subdir_ocr, exist_ok=True)
            
            classify_output_path = os.path.join(output_subdir_classify, os.path.splitext(file)[0] + "_classify.json")
            ocr_output_path = os.path.join(output_subdir_ocr, os.path.splitext(file)[0] + "_ocr.json")

            # Check if both classification and OCR have been completed
            if os.path.exists(classify_output_path) and os.path.exists(ocr_output_path):
                logging.info(f"Skipping already processed file: {file_path}")
                continue

            # Check if classification has already been done
            if os.path.exists(classify_output_path):
                logging.info(f"Classification already done for {file_path}. Loading existing results.")
                with open(classify_output_path, 'r') as f:
                    classification_json = json.load(f)
            else:
                # Run classification using classify.py
                classification_json = classify_process_pdf(file_path)

                if classification_json is None:
                    logging.info(f"Skipped {file_path} due to high percentage of excluded pages")
                    continue

                # Save classification results to JSON
                with open(classify_output_path, 'w') as f:
                    json.dump(classification_json, f, indent=4)

            # Check if OCR has already been done
            if os.path.exists(ocr_output_path):
                logging.info(f"OCR already done for {file_path}. Loading existing results.")
                with open(ocr_output_path, 'r') as f:
                    ocr_results = json.load(f)
            else:
                # Perform OCR using ocr.py on all pages
                json_file_path = client.process(file_path, output_subdir_ocr)

                # Load OCR results
                with open(json_file_path, 'r') as f:
                    ocr_results = json.load(f)

            # Filter out unwanted pages from OCR results based on classification
            filtered_ocr_results = filter_pages(classification_json, ocr_results, excluded_types)

            # Save filtered OCR results to a new JSON file
            with open(ocr_output_path, 'w') as f:
                json.dump(filtered_ocr_results, f, indent=4)

            update_page_keys_in_json(ocr_output_path)
            reformat_json_structure(ocr_output_path)

    client.close()

if __name__ == "__main__":
    main()