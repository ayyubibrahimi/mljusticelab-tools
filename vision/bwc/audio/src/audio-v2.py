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
import itertools

logging.basicConfig(level=logging.INFO)

# Define expanded parameter ranges
params = {
    "pipeline": {
        "max_new_tokens": [64, 128, 256, 512],
        "chunk_length_s": [10, 20, 30, 40, 50, 60],
        "batch_size": [8, 16, 32, 64],
    },
    "high_pass_filter": {
        "cutoff": [30, 50, 80, 100, 150, 200],
    },
    "enhance_speech_frequencies": {
        "low_freq": [100, 200, 300, 400, 500],
        "high_freq": [1500, 2000, 3000, 4000, 5000],
        "boost_factor": [1, 2, 4, 6, 8],
    },
}

def get_evenly_spaced_indices(num_total, num_select):
    return np.linspace(0, num_total - 1, num_select, dtype=int)

# Generate all combinations
param_combinations = list(itertools.product(
    *[[(k, v) for v in values] for k, values in params["pipeline"].items()],
    *[[(k, v) for v in values] for k, values in params["high_pass_filter"].items()],
    *[[(k, v) for v in values] for k, values in params["enhance_speech_frequencies"].items()]
))

# Sort combinations based on the sum of parameter values
sorted_combinations = sorted(param_combinations, key=lambda x: sum(v for k, v in x if isinstance(v, (int, float))))

# Select 10 evenly spaced combinations
num_combinations = len(sorted_combinations)
selected_indices = get_evenly_spaced_indices(num_combinations, 10)
selected_combinations = [dict(sorted_combinations[i]) for i in selected_indices]

def load_config():
    with open('config.json') as config_file:
        return json.load(config_file)

def extract_audio_from_mp4(video_file_path):
    try:
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with VideoFileClip(video_file_path) as video:
            audio = video.audio
            audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le")
        return temp_audio_file.name
    except Exception as e:
        logging.error(f"Error extracting audio from {video_file_path}: {e}")
        return None

def high_pass_filter(y, sr, cutoff):
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(5, normalized_cutoff, btype='high', analog=False)
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered

def enhance_speech_frequencies(y, sr, low_freq, high_freq, boost_factor):
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    
    y_filtered = scipy.signal.lfilter(b, a, y)
    y_enhanced = y + (y_filtered * boost_factor)
    
    return librosa.util.normalize(y_enhanced)

def preprocess_audio(audio_path, params):
    try:
        logging.info(f"Starting preprocessing for {audio_path}")
        start_time = time.time()

        y, sr = librosa.load(audio_path, sr=None)
        logging.info(f"Audio loaded. Duration: {len(y)/sr:.2f} seconds")

        if not np.isfinite(y).all():
            logging.warning("Non-finite values detected in the audio. Replacing with zeros.")
            y = np.nan_to_num(y)

        y_filtered = high_pass_filter(y, sr, cutoff=params["cutoff"])
        y_normalized = librosa.util.normalize(y_filtered)
        y_enhanced = enhance_speech_frequencies(
            y_normalized, sr, 
            low_freq=params["low_freq"], 
            high_freq=params["high_freq"], 
            boost_factor=params["boost_factor"]
        )

        if not np.isfinite(y_enhanced).all():
            logging.warning("Non-finite values detected after processing. Replacing with zeros.")
            y_enhanced = np.nan_to_num(y_enhanced)

        temp_processed_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_processed_file.name, y_enhanced, sr)

        end_time = time.time()
        logging.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds")

        return temp_processed_file.name
    except Exception as e:
        logging.error(f"Error preprocessing audio {audio_path}: {e}")
        return audio_path

def setup_whisper_pipeline(params):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        max_new_tokens=params["max_new_tokens"],
        chunk_length_s=params["chunk_length_s"],
        batch_size=params["batch_size"],
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},
    )

    return pipe

def write_results(output_path, transcript, params):
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- Parameters ---\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Transcript ---\n")
        chunk_count = 0
        for chunk in transcript:
            f.write(f"{chunk['text']} ")
            chunk_count += 1
            if chunk_count % 8 == 0:
                f.write('\n')

def process_file(input_dir, output_dir, filename, pipe, params):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_results.txt")

    logging.info(f"Processing file: {filename}")

    audio_path = extract_audio_from_mp4(input_path)
    if not audio_path:
        return

    try:
        preprocessed_audio_path = preprocess_audio(audio_path, params)
        
        start_time = time.time()
        result = pipe(preprocessed_audio_path)
        end_time = time.time()

        logging.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        write_results(output_path, result["chunks"], params)
        logging.info(f"Results appended to {output_path}")
        
        return result
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

    mp4_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for params in selected_combinations:
        pipe = setup_whisper_pipeline(params)
        
        for filename in mp4_files:
            process_file(input_dir, output_dir, filename, pipe, params)

if __name__ == "__main__":
    main()