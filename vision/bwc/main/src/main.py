import os
import json
import re
import time
from datetime import datetime
from read_audio import process_audio
from analyze_video import analyze_video
import ocr
from analyze_report import load_and_split, create_memory_log, process_file, final_summarization, organize_final_summary, save_summaries_to_json

INPUT_AUDIO_DIR = "../../audio/data/output"
OUTPUT_DIR = "../data/output"
INPUT_VIDEO_DIR = "../../audio/data/input"  

INPUT_OCR_DIR = "../../ocr/data/input"
OUTPUT_OCR_DIR = "../../ocr/data/output"

OUTPUT_REPORT_DIR = "../../report/data/output"

def parse_timestamp(timestamp_str):
    match = re.search(r'start_time=(\d+\.\d+), end_time=(\d+\.\d+)', timestamp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def count_true_matches(video_results):
    true_count = sum(1 for result in video_results if result['match_classification'].strip().split('\n')[1] == 'TRUE')
    return true_count, len(video_results)

def process_pdf_files(input_dir, output_dir):
    endpoint, key = ocr.getcreds()
    client = ocr.create_client(endpoint, key)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            json_file_path = ocr.process_pdf(client, pdf_path, output_dir)
            
            if json_file_path:
                ocr.update_page_keys_in_json(json_file_path)
                ocr.reformat_json_structure(json_file_path)
                print(f"Processed PDF: {filename}")

def process_report_data(input_dir, output_dir):
    output_data = []
    custom_template = ""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                output_filename = os.path.basename(filename)
                output_file_path = os.path.join(output_dir, output_filename)
                
                if os.path.exists(output_file_path):
                    print(f"Skipping file: {filename} (output already exists)")
                    continue
                
                print(f"Processing file: {filename}")
                json_path = os.path.join(input_dir, filename)
                
                docs = load_and_split(json_path)
                memory_log = create_memory_log(docs)
                
                file_summaries = process_file(filename, input_dir, output_dir, memory_log)
                final_summary = final_summarization(file_summaries, memory_log)
                condensed_summary = organize_final_summary(final_summary)
                
                start_page = docs[0].metadata["seq_num"]
                end_page = docs[-1].metadata["seq_num"]
                
                output_data.append(save_summaries_to_json(condensed_summary, filename, start_page, end_page))
        
                # Write to a file with the original filename in the output directory
                with open(output_file_path, "w") as output_file:
                    json.dump(output_data, output_file, indent=4)
                
                print(f"Processed and saved: {output_filename}")
                output_data = []  # Reset output_data for the next file
        
        end_time = time.time()
        total_time = end_time - start_time
        print("All files processed successfully.")
        print(f"Total execution time: {total_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error processing JSON: {str(e)}")
        print(json.dumps({"success": False, "message": "Failed to process JSON"}))
        raise

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OCR_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

    # Process PDF files
    print("Processing PDF files...")
    process_pdf_files(INPUT_OCR_DIR, OUTPUT_OCR_DIR)

    # Process report data
    print("Processing report data...")
    process_report_data(OUTPUT_OCR_DIR, OUTPUT_REPORT_DIR)

    # Read the processed report data
    report_files = [f for f in os.listdir(OUTPUT_REPORT_DIR) if f.endswith('.json')]
    if report_files:
        first_report_file = report_files[0]
        report_file_path = os.path.join(OUTPUT_REPORT_DIR, first_report_file)
        
        with open(report_file_path, 'r') as f:
            report_data = json.load(f)
        
        report_text = report_data[0]['files'][0]['sentence']
        print(report_text)
        print(f"Processing report file: {first_report_file}")
    else:
        print("No report files found in the output directory.")
        return


    for filename in os.listdir(INPUT_AUDIO_DIR):
        if filename.endswith('.json'):
            input_path = os.path.join(INPUT_AUDIO_DIR, filename)
            
            print(f"Processing audio file: {filename}")
            
            # Process the audio transcription
            audio_result = process_audio(input_path)
            
            print(f"Audio analysis result for {filename}: {audio_result}")
            
            # Parse the timestamp from the audio result
            start_time, end_time = parse_timestamp(audio_result)
            
            if start_time is not None and end_time is not None:
                video_filename = os.path.splitext(filename)[0] + '.mp4' 
                video_path = os.path.join(INPUT_VIDEO_DIR, video_filename)
                
                if os.path.exists(video_path):
                    print(f"Analyzing video segment: {video_filename}")
                    
                    iteration = 0
                    max_iterations = 5
                    time_step = 1  # Initial time step
                    
                    while iteration < max_iterations:
                        video_results = analyze_video(video_path, start_time, end_time, frames_per_context=1, report_text=report_text, time_step=time_step)
                        
                        true_matches, total_matches = count_true_matches(video_results)
                        match_percentage = (true_matches / total_matches) * 100 if total_matches > 0 else 0
                        
                        print(f"Iteration {iteration + 1}: {true_matches}/{total_matches} ({match_percentage:.2f}%) TRUE matches")
                        
                        if match_percentage >= 10:
                            print("Reached 10% or more TRUE matches. Stopping iterations.")
                            break
                        
                        # Adjust time window
                        start_time -= 10
                        end_time += 10
                        time_step = max(0.01, time_step - 0.25)  
                        print(f"Adjusting time window: start_time={start_time}, end_time={end_time}, time_step={time_step}")

                        iteration += 1
                    
                    output_filename = os.path.splitext(filename)[0] + '.json'
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    with open(output_path, 'w') as outfile:
                        json.dump({
                            'audio_result': audio_result,
                            'video_results': video_results,
                            'final_start_time': start_time,
                            'final_end_time': end_time,
                            'iterations': iteration,
                            'match_percentage': match_percentage
                        }, outfile, indent=2)
                    
                    print(f"Results saved to: {output_path}")
                else:
                    print(f"Video file not found: {video_path}")
            else:
                print(f"Failed to parse timestamp from audio result: {audio_result}")
            
            print()  

if __name__ == "__main__":
    main()