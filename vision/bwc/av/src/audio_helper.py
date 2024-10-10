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
        chunk_length_s=35,
        batch_size=32,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},
    )

    return pipe
