import os
import logging
import pandas as pd
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import concurrent.futures

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
INPUT_DIR = "../data/input/archive"
OUTPUT_DIR = "../data/output_baseline"
LARGE_IMAGE_THRESHOLD = 3.7 * 1024 * 1024  # 5 megapixels

def is_large_image(pdf_path, page_number):
    """Check if a page in the PDF is a large image"""
    try:
        images = convert_from_path(pdf_path, first_page=page_number + 1, last_page=page_number + 1)
        image = images[0]
        width, height = image.size
        return (width * height) > LARGE_IMAGE_THRESHOLD
    except Exception as e:
        logging.warning(f"Error processing page {page_number+1} of {pdf_path}: {str(e)}")
        return False

def process_pdf(pdf_path):
    """Process a PDF file page by page in parallel and classify based on image size"""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        page_count = len(reader.pages)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_page = {executor.submit(is_large_image, pdf_path, page_number): page_number for page_number in range(page_count)}
        results = {}
        for future in concurrent.futures.as_completed(future_to_page):
            page_number = future_to_page[future]
            is_large = future.result()
            results[page_number + 1] = int(is_large)  # 1 if large image, 0 otherwise
    
    return results

def process_dataset(pdf_path, csv_path, output_path):
    """Process a single dataset and save results"""
    # Read original CSV
    df = pd.read_csv(csv_path)

    # Process PDF
    model_predictions = process_pdf(pdf_path)

    # Add model predictions to dataframe
    df["prediction"] = df["page_no"].map(model_predictions)

    # Sort the dataframe by page_no
    df = df.sort_values("page_no")

    # Save results
    df.to_csv(output_path, index=False)

def calculate_metrics(output_dir):
    """Calculate comprehensive metrics for each processed CSV and overall metrics"""
    results = []
    overall_metrics = {
        'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0
    }

    for filename in os.listdir(output_dir):
        if filename.endswith("_baseline_output.csv"):
            filepath = os.path.join(output_dir, filename)
            df = pd.read_csv(filepath)

            if 'label' not in df.columns or 'prediction' not in df.columns:
                logging.error(f"Required columns not found in {filename}. Skipping this file.")
                continue

            TP = ((df['label'] == 1) & (df['prediction'] == 1)).sum()
            FP = ((df['label'] == 0) & (df['prediction'] == 1)).sum()
            TN = ((df['label'] == 0) & (df['prediction'] == 0)).sum()
            FN = ((df['label'] == 1) & (df['prediction'] == 0)).sum()

            total = TP + FP + TN + FN
            accuracy = (TP + TN) / total if total > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'dataset': filename,
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })

            # Accumulate overall metrics
            for key in ['TP', 'FP', 'TN', 'FN']:
                overall_metrics[key] += results[-1][key]

    # Calculate overall metrics
    total_samples = sum(overall_metrics.values())
    overall_accuracy = (overall_metrics['TP'] + overall_metrics['TN']) / total_samples if total_samples > 0 else 0
    overall_precision = overall_metrics['TP'] / (overall_metrics['TP'] + overall_metrics['FP']) if (overall_metrics['TP'] + overall_metrics['FP']) > 0 else 0
    overall_recall = overall_metrics['TP'] / (overall_metrics['TP'] + overall_metrics['FN']) if (overall_metrics['TP'] + overall_metrics['FN']) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    # Add overall metrics to results
    results.append({
        'dataset': 'Overall',
        'TP': overall_metrics['TP'],
        'FP': overall_metrics['FP'],
        'TN': overall_metrics['TN'],
        'FN': overall_metrics['FN'],
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1
    })

    # Create and save results.csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)

    return results_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, pdf_file)

        csv_file = os.path.splitext(pdf_file)[0] + ".csv"
        csv_path = os.path.join(INPUT_DIR, csv_file)

        if not os.path.exists(csv_path):
            logging.warning(f"Missing CSV file for {pdf_file}")
            continue

        output_filename = f"{os.path.splitext(pdf_file)[0]}_baseline_output.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Check if output file already exists
        if os.path.exists(output_path):
            logging.info(f"Output file {output_filename} already exists. Skipping processing.")
            continue

        process_dataset(pdf_path, csv_path, output_path)

        logging.info(f"Processed {pdf_file} and saved results to {output_filename}")

    # Calculate comprehensive metrics after processing all datasets
    results_df = calculate_metrics(OUTPUT_DIR)

    # Log overall metrics
    overall_metrics = results_df.iloc[-1]
    logging.info(f"Overall Metrics:")
    logging.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logging.info(f"Precision: {overall_metrics['precision']:.4f}")
    logging.info(f"Recall: {overall_metrics['recall']:.4f}")
    logging.info(f"F1 Score: {overall_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()