import os
import json
import logging
import tempfile
import numpy as np
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import librosa
import soundfile as sf
import scipy.signal

logging.basicConfig(level=logging.INFO)

def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

def extract_audio_from_mp4(video_file_path):
    """
    Extracts the audio from an MP4 file and saves it as a temporary WAV file.
    """
    try:
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with VideoFileClip(video_file_path) as video:
            audio = video.audio
            audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le")
        return temp_audio_file.name
    except Exception as e:
        logging.error(f"Error extracting audio from {video_file_path}: {e}")
        return None


def high_pass_filter(y, sr, cutoff=100):
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(5, normalized_cutoff, btype='high', analog=False)
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered

def enhance_speech_frequencies(y, sr):
    # Define speech frequency range (approx. 300-3000 Hz)
    low_freq, high_freq = 300, 3000
    
    # Create a bandpass filter
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    
    # Apply the filter
    y_filtered = scipy.signal.lfilter(b, a, y)
    
    # Boost the filtered signal
    boost_factor = 3
    y_enhanced = y + (y_filtered * boost_factor)
    
    return librosa.util.normalize(y_enhanced)

def preprocess_audio(audio_path):
    """
    Preprocesses the audio file to improve speech clarity for transcription.
    """
    try:
        logging.info(f"Starting preprocessing for {audio_path}")
        start_time = time.time()

        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        logging.info(f"Audio loaded. Duration: {len(y)/sr:.2f} seconds")

        # Check for non-finite values
        if not np.isfinite(y).all():
            logging.warning("Non-finite values detected in the audio. Replacing with zeros.")
            y = np.nan_to_num(y)

        # Apply a high-pass filter to remove low-frequency noise
        y_filtered = high_pass_filter(y, sr, cutoff=100)

        # Normalize audio
        y_normalized = librosa.util.normalize(y_filtered)

        # Enhance speech frequencies
        y_enhanced = enhance_speech_frequencies(y_normalized, sr)

        # Check for non-finite values again
        if not np.isfinite(y_enhanced).all():
            logging.warning("Non-finite values detected after processing. Replacing with zeros.")
            y_enhanced = np.nan_to_num(y_enhanced)

        # Create a temporary file for the processed audio
        temp_processed_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_processed_file.name, y_enhanced, sr)

        end_time = time.time()
        logging.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds")

        return temp_processed_file.name
    except Exception as e:
        logging.error(f"Error preprocessing audio {audio_path}: {e}")
        return audio_path

def write_transcript_to_file(transcript, output_path):
    """
    Writes the processed transcript to a text file with timestamps.
    Each chunk is written on a new line with formatted timestamps.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in transcript:
                start, end = chunk['timestamp']
                text = chunk['text'].strip()
                formatted_line = f"[{start:.2f} - {end:.2f}] {text}\n"
                f.write(formatted_line)
        print(f"Transcript successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing transcript to {output_path}: {e}")

def write_transcript_to_json(transcript, output_path):
    """
    Writes the processed transcript to a JSON file.
    Each chunk includes start_timestamp, end_timestamp, and text.
    """
    try:
        json_data = []
        for chunk in transcript:
            start, end = chunk['timestamp']
            json_data.append({
                "start_timestamp": start,
                "end_timestamp": end,
                "text": chunk['text'].strip()
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON transcript successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing JSON transcript to {output_path}: {e}")

def setup_whisper_pipeline():
    """
    Sets up and returns the Whisper pipeline for transcription.
    """
    device = "mps:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},
    )

    return pipe

def process_file(input_dir, output_dir, filename, pipe):
    """
    Processes a single file: extracts audio, preprocesses it, transcribes it, and saves the transcript.
    """
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
    output_json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")


    logging.info(f"Processing file: {filename}")

    audio_path = extract_audio_from_mp4(input_path)
    if not audio_path:
        return

    try:
        preprocessed_audio_path = preprocess_audio(audio_path)
        
        start_time = time.time()
        result = pipe(audio_path)
        end_time = time.time()

        logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        write_transcript_to_file(result["chunks"], output_path)
        write_transcript_to_json(result["chunks"], output_json_path)

        logging.info(f"Transcript saved to {output_path}")
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
    finally:
        os.unlink(audio_path)
        if preprocessed_audio_path != audio_path:
            os.unlink(preprocessed_audio_path)

def main():
    config = load_config()
    input_dir = config.get("input_dir", "../data/input")
    output_dir = config.get("output_dir", "../data/output/")

    os.makedirs(output_dir, exist_ok=True)

    pipe = setup_whisper_pipeline()

    mp4_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for filename in mp4_files:
        process_file(input_dir, output_dir, filename, pipe)

if __name__ == "__main__":
    main()