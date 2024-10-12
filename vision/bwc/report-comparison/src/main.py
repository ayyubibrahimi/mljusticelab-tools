import os
import json
import logging
from moviepy.editor import VideoFileClip
from dotenv import find_dotenv, load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_helper import setup_llm, process_video_segment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

def load_json_data(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)

def process_file(input_path, json_path, output_path, llm):
    logging.info(f"Processing file: {input_path}")

    try:
        # Load JSON data
        audio_chunks = load_json_data(json_path)

        # Process video based on audio chunks
        video = VideoFileClip(input_path)
        results = []

        for chunk in audio_chunks:
            start_time, end_time = chunk['start_timestamp'], chunk['end_timestamp']
            transcription = chunk['text'].strip()
            video_description = process_video_segment(video, start_time, end_time, llm)
            results.append((start_time, end_time, transcription, video_description))

        # Write results to file
        write_results_to_file(results, output_path)

        video.close()
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")

def write_results_to_file(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for start_time, end_time, transcription, description in results:
            f.write(f"[{start_time:.2f} - {end_time:.2f}]\n")
            f.write(f"Transcription: {transcription}\n")
            f.write(f"Video Description: {description}\n\n")

def main():
    config = load_config()
    input_dir = config.get("input_dir", "../data/input")
    output_dir = config.get("output_dir", "../data/output/")

    os.makedirs(output_dir, exist_ok=True)

    llm = setup_llm()

    mp4_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for filename in mp4_files:
        input_path = os.path.join(input_dir, filename)
        json_filename = f"{os.path.splitext(filename)[0]}.json"
        json_path = os.path.join(input_dir, json_filename)
        
        if not os.path.exists(json_path):
            logging.warning(f"JSON file not found for {filename}. Skipping this file.")
            continue

        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_analysis.txt")
        process_file(input_path, json_path, output_path, llm)

if __name__ == "__main__":
    main()