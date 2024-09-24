import pandas as pd
import PyPDF2
import os

def concatenate_csvs(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    csv_files.sort()  # Ensure files are processed in order

    dfs = []
    current_page = 0
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(directory, csv_file))
        df['page_no'] += current_page  # Adjust page numbers
        
        # Count pages in the corresponding PDF
        pdf_file = csv_file.replace('.csv', '.pdf')
        pdf_path = os.path.join(directory, pdf_file)
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            current_page += len(pdf_reader.pages)
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def concatenate_pdfs(directory, output_file):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    pdf_files.sort()  # Ensure files are processed in order

    pdf_writer = PyPDF2.PdfWriter()

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)

    with open(output_file, 'wb') as out_file:
        pdf_writer.write(out_file)

def main():
    directory = '../data/archive' 
    output_csv = '../wrangle/combined_dataset.csv'
    output_pdf = '../wrangle/combined_dataset.pdf'

    combined_df = concatenate_csvs(directory)
    combined_df.to_csv(output_csv, index=False)
    print(f"Combined CSV saved as {output_csv}")

    concatenate_pdfs(directory, output_pdf)
    print(f"Combined PDF saved as {output_pdf}")

if __name__ == "__main__":
    main()