import os
import json
import logging
import tempfile
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import cv2
from PIL import Image
import base64
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
import time
import librosa
import soundfile as sf
import scipy.signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

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

def high_pass_filter(y, sr, cutoff=100):
    nyquist = 0.5 * sr
    normalized_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(5, normalized_cutoff, btype='high', analog=False)
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered

def enhance_speech_frequencies(y, sr):
    low_freq, high_freq = 300, 3000
    nyquist = 0.5 * sr
    low, high = low_freq / nyquist, high_freq / nyquist
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    y_filtered = scipy.signal.lfilter(b, a, y)
    boost_factor = 3
    y_enhanced = y + (y_filtered * boost_factor)
    return librosa.util.normalize(y_enhanced)

def preprocess_audio(audio_path):
    try:
        logging.info(f"Starting preprocessing for {audio_path}")
        start_time = time.time()

        y, sr = librosa.load(audio_path, sr=None)
        logging.info(f"Audio loaded. Duration: {len(y)/sr:.2f} seconds")

        if not np.isfinite(y).all():
            logging.warning("Non-finite values detected in the audio. Replacing with zeros.")
            y = np.nan_to_num(y)

        y_filtered = high_pass_filter(y, sr, cutoff=100)
        y_normalized = librosa.util.normalize(y_filtered)
        y_enhanced = enhance_speech_frequencies(y_normalized, sr)

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

def setup_whisper_pipeline():
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
        max_new_tokens=128,
        chunk_length_s=35,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": "english"},
    )

    return pipe

def setup_openai_chat():
    return ChatOpenAI(model="gpt-4-vision-preview")

def process_audio(audio_file, pipe):
    logging.info(f"Processing audio: {audio_file}")
    result = pipe(audio_file)
    return result["chunks"]

def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)
    processed_image = Image.fromarray(smooth)
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.2)
    sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
    final_image = sharpness_enhancer.enhance(1.3)
    return final_image

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def process_video_segment(video, start_time, end_time, chat):
    logging.info(f"Processing video segment: {start_time:.2f} - {end_time:.2f}")
    frames = []
    for t in np.arange(start_time, end_time, 1.0):  # 1 frame per second
        frame = video.get_frame(t)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image = preprocess_image(pil_image)
        base64_frame = encode_frame(np.array(processed_image))
        frames.append(base64_frame)

    prompt = f"""
    Analyze the following sequence of frames from a video scene. 
    The scene starts at {start_time:.2f} seconds and ends at {end_time:.2f} seconds.
    
    Provide a technical description of action or changes occurring in this scene.
    Write your description inside <image_analysis> tags. 
    Begin your description with "The image depicts" or a similar phrase.
    """

    image_contents = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
        }
        for frame in frames
    ]

    try:
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        *image_contents
                    ]
                )
            ]
        )
        return msg.content
    except Exception as e:
        logging.error(f"Error in OpenAI API call for scene {start_time:.2f} - {end_time:.2f}: {str(e)}")
        raise

def process_file(input_path, output_path, audio_pipe, video_chat):
    logging.info(f"Processing file: {input_path}")

    # Extract audio
    audio_file = extract_audio_from_mp4(input_path)
    if not audio_file:
        return

    try:
        # Preprocess audio
        preprocessed_audio_path = preprocess_audio(audio_file)
        
        # Process audio
        start_time = time.time()
        audio_chunks = process_audio(preprocessed_audio_path, audio_pipe)
        end_time = time.time()
        logging.info(f"Audio transcription completed in {end_time - start_time:.2f} seconds")

        # Process video based on audio chunks
        video = VideoFileClip(input_path)
        results = []

        for chunk in audio_chunks:
            start_time, end_time = chunk['timestamp']
            transcription = chunk['text']
            video_description = process_video_segment(video, start_time, end_time, video_chat)
            results.append((start_time, end_time, transcription, video_description))

        # Write results to file
        write_results_to_file(results, output_path)

        video.close()
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
    finally:
        os.unlink(audio_file)
        if preprocessed_audio_path != audio_file:
            os.unlink(preprocessed_audio_path)

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

    audio_pipe = setup_whisper_pipeline()
    video_chat = setup_openai_chat()

    mp4_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    for filename in mp4_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_analysis.txt")
        process_file(input_path, output_path, audio_pipe, video_chat)

if __name__ == "__main__":
    main()