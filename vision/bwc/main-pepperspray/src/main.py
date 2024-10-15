import os
import json
import re
from read_audio import process_audio
from analyze_video import analyze_video

INPUT_DIR = "../../audio/data/output"
OUTPUT_DIR = "../data/output"
VIDEO_DIR = "../../audio/data/input"  

def parse_timestamp(timestamp_str):
    match = re.search(r'start_time=(\d+\.\d+), end_time=(\d+\.\d+)', timestamp_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.json'):
            input_path = os.path.join(INPUT_DIR, filename)
            
            print(f"Processing audio file: {filename}")
            
            # Process the audio transcription
            audio_result = process_audio(input_path)
            
            print(f"Audio analysis result for {filename}: {audio_result}")
            
            # Parse the timestamp from the audio result
            start_time, end_time = parse_timestamp(audio_result)
            
            if start_time is not None and end_time is not None:
                video_filename = os.path.splitext(filename)[0] + '.mp4' 
                video_path = os.path.join(VIDEO_DIR, video_filename)
                
                if os.path.exists(video_path):
                    print(f"Analyzing video segment: {video_filename}")
                    video_results = analyze_video(video_path, start_time, end_time,frames_per_context=1)
                    
                    output_filename = os.path.splitext(filename)[0] + '_result.json'
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    with open(output_path, 'w') as outfile:
                        json.dump({
                            'audio_result': audio_result,
                            'video_results': video_results
                        }, outfile, indent=2)
                    
                    print(f"Results saved to: {output_path}")
                else:
                    print(f"Video file not found: {video_path}")
            else:
                print(f"Failed to parse timestamp from audio result: {audio_result}")
            
            print()  

if __name__ == "__main__":
    main()