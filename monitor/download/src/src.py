import pandas as pd
import os
import re
import logging
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import io
import time
import hashlib
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_downloader.log'),
        logging.StreamHandler()
    ]
)

class ProcessingHistory:
    def __init__(self, history_file='processed_urls.txt'):
        self.history_file = history_file
        self.processed_urls = self._load_history()

    def _load_history(self):
        """Load previously processed URLs from history file."""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def is_processed(self, url):
        """Check if URL has been processed before."""
        return url in self.processed_urls if url else False

    def mark_processed(self, url):
        """Mark URL as processed and save to history file."""
        if url:
            self.processed_urls.add(url)
            with open(self.history_file, 'a') as f:
                f.write(f"{url}\n")

def get_drive_service():
    """Create a Drive API service using OAuth 2.0."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None

    # Load credentials from token.pickle if it exists
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json',
                SCOPES
            )
            creds = flow.run_local_server(port=0)
            
        # Save credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('drive', 'v3', credentials=creds)
        logging.info("Successfully created Drive API service")
        return service
    except Exception as e:
        logging.error(f"Error creating Drive API service: {str(e)}")
        raise

def extract_folder_id(url):
    """Extract Google Drive folder ID from URL."""
    if pd.isna(url):
        return None
    match = re.search(r'folders/([a-zA-Z0-9-_]+)', url)
    return match.group(1) if match else None

def get_safe_directory_name(officer_name, star_no):
    """Create a safe directory name from officer name and optional star number."""
    safe_name = re.sub(r'[<>:"/\\|?*]', '', str(officer_name).strip())
    safe_name = safe_name.replace(' ', '_')
    
    if pd.notna(star_no):
        return f"{safe_name}_{star_no}"
    return safe_name

def list_files_in_folder(service, folder_id):
    """List all PDF files in a public Google Drive folder."""
    try:
        logging.info(f"Listing files in folder: {folder_id}")
        
        query = f"'{folder_id}' in parents and mimeType='application/pdf'"
        results = service.files().list(
            q=query,
            fields="files(id, name, md5Checksum)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        logging.info(f"Found {len(files)} PDF files in folder")
        return files
        
    except Exception as e:
        logging.error(f"Error listing files in folder {folder_id}: {str(e)}")
        return []

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_file_already_downloaded(file_info, output_dir):
    """Check if file already exists and matches the MD5 checksum."""
    if 'md5Checksum' not in file_info:
        return False
    
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            if get_file_hash(filepath) == file_info['md5Checksum']:
                return True
    return False

def download_file(service, file_id, output_path, max_retries=5):
    """Download a file from Google Drive with retry logic."""
    for attempt in range(max_retries):
        try:
            logging.info(f"Downloading file {file_id} to {output_path}")
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logging.info(f"Download {int(status.progress() * 100)}%")
                
            fh.seek(0)
            with open(output_path, 'wb') as f:
                f.write(fh.read())
                
            logging.info(f"Successfully downloaded file to {output_path}")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                logging.warning(f"Download failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            logging.error(f"Error downloading file {file_id}: {str(e)}")
            return False


def download_pdfs_from_folder(service, folder_id, output_dir, incident_date=None):
    """Download all PDFs from a public Google Drive folder with parallel downloads."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")
    
    results = []
    files = list_files_in_folder(service, folder_id)
    
    def download_single_file(file):
        file_id = file['id']
        file_name = file['name']
        
        # If incident date exists, prefix it to the filename
        if incident_date and pd.notna(incident_date):
            incident_date_safe = str(incident_date).replace('/', '-')
            file_name = f"{incident_date_safe}_{file_name}"
        
        file_path = os.path.join(output_dir, file_name)
        
        # Check if file already exists with matching checksum
        if is_file_already_downloaded(file, output_dir):
            logging.info(f"File {file_name} already exists with matching checksum, skipping")
            return file_path
        
        # If file exists but checksum doesn't match, create new filename
        counter = 1
        base_name, ext = os.path.splitext(file_path)
        while os.path.exists(file_path):
            file_path = f"{base_name}_{counter}{ext}"
            counter += 1
        
        # Download file
        if download_file(service, file_id, file_path):
            return file_path
        return None

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_file = {executor.submit(download_single_file, file): file for file in files}
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                path = future.result()
                if path:
                    results.append(path)
                    time.sleep(1)  # Small delay between completions to avoid rate limiting
            except Exception as e:
                logging.error(f"Error downloading file: {str(e)}")

    return results




def split_incidents_and_urls(row):
    """Split multiple incidents and their corresponding URLs."""
    incidents = str(row.get('1421 Incident', '')).split(',')
    url_text = str(row.get('1421 files (google link)', ''))
    
    # Extract URLs using regex pattern
    urls = re.findall(r'https://drive\.google\.com/drive/folders/[^?\s]+', url_text)
    
    # Clean up the incidents and urls
    incidents = [i.strip() for i in incidents if i.strip()]
    urls = [u.strip() for u in urls if u.strip()]
    
    # Pair them together, handling cases where counts might not match
    return list(zip(incidents, urls)) if incidents and urls else []



def process_csv():
    """Process the CSV file and download PDFs."""
    logging.info("Starting CSV processing")
    
    try:
        # Initialize Drive API service and processing history
        service = get_drive_service()
        history = ProcessingHistory()
        
        # Check if output CSV exists and load it
        output_csv_path = '../data/output/processed_index.csv'
        if os.path.exists(output_csv_path):
            existing_df = pd.read_csv(output_csv_path)
            logging.info(f"Loaded existing output CSV with {len(existing_df)} rows")
        else:
            existing_df = pd.DataFrame()
        
        # Read the input CSV
        df = pd.read_csv('../data/input/index.csv')
        logging.info(f"Successfully read input CSV with {len(df)} rows")
        
        # Create output directory
        output_base_dir = '../data/output/pdfs'
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        
        # Initialize list to store new rows
        new_rows = []
        
        # Process each row
        for idx, row in df.iterrows():
            logging.info(f"Processing row {idx}: {row['Officer Name']}")
            
            # Get incident-URL pairs
            incident_url_pairs = split_incidents_and_urls(row)
            
            # Skip if no valid pairs found
            if not incident_url_pairs:
                logging.info(f"No valid incident-URL pairs found for row {idx}, skipping")
                continue
            
            # Process each incident-URL pair
            for incident, url in incident_url_pairs:
                # Skip if URL has been processed before
                if history.is_processed(url):
                    logging.info(f"URL already processed, skipping: {url}")
                    # Add existing entries to new_rows
                    if not existing_df.empty:
                        matching_rows = existing_df[existing_df['1421 files (google link)'] == url]
                        new_rows.extend(matching_rows.to_dict('records'))
                    continue
                
                folder_id = extract_folder_id(url)
                if not folder_id:
                    logging.info(f"No folder ID found for URL: {url}")
                    continue
                
                # Create a modified row with single incident
                incident_row = row.copy()
                incident_row['1421 Incident'] = incident
                
                officer_dir = os.path.join(
                    output_base_dir,
                    get_safe_directory_name(row['Officer Name'], row.get('Star No'))
                )
                
                # Download PDFs in parallel
                pdf_paths = download_pdfs_from_folder(
                    service,
                    folder_id,
                    officer_dir,
                    incident_date=incident
                )
                
                # Create new rows for each PDF
                for pdf_path in pdf_paths:
                    new_row = incident_row.copy()
                    new_row['local_pdf_path'] = pdf_path
                    new_rows.append(new_row)
                
                # Mark URL as processed
                history.mark_processed(url)
        
        # Create new DataFrame and save
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(output_csv_path, index=False)
            logging.info(f"Successfully saved processed CSV with {len(new_df)} rows")
        else:
            logging.info("No new rows to save")
        
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        raise


if __name__ == "__main__":
    logging.info("Starting script")
    process_csv()  # Removed API_KEY parameter
    logging.info("Script completed")