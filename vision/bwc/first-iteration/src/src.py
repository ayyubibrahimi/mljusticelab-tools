
import os
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

INPUT_DIR = "../data/input/"
OUTPUT_DIR = "../data/output"


chat = ChatOpenAI(
        model="gpt-4o-mini",
    )

def preprocess_image(image):
    """Apply advanced preprocessing techniques to enhance overall image quality"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if image is in RGBA mode
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    
    # Convert to LAB color space for more accurate color processing
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel to improve overall contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back and convert to RGB
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Apply bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # Adjust gamma to enhance details in darker regions
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(smooth, table)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(gamma_corrected)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = enhancer.enhance(1.2)
    
    # Enhance color
    color_enhancer = ImageEnhance.Color(contrast_enhanced)
    color_enhanced = color_enhancer.enhance(1.1)
    
    # Enhance sharpness one more time
    sharpness_enhancer = ImageEnhance.Sharpness(color_enhanced)
    final_image = sharpness_enhancer.enhance(1.3)
    
    return final_image

def encode_frame(frame):
    """Encode a video frame to base64"""
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    processed_image = preprocess_image(pil_image)
    
    # Convert back to OpenCV format
    processed_frame = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode(".jpg", processed_frame)
    return base64.b64encode(buffer).decode("utf-8")

def scene_summarize(frames_base64, start_time, end_time):
    """Summarize a video scene"""
    logging.info(f"Summarizing scene from {start_time:.2f} to {end_time:.2f} seconds")

    prompt = f"""
    Analyze the following sequence of frames from a video scene. 
    The scene starts at {start_time:.2f} seconds and ends at {end_time:.2f} seconds.
    
    Provide a technical description of action or changes occurring in this scene.

    Write your description inside <image_analysis> tags. Begin your description with "The image depicts" or a similar phrase.
    """

    image_contents = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
        }
        for frame in frames_base64
    ]

    logging.info(f"Sending request to OpenAI API for scene {start_time:.2f} - {end_time:.2f}")
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
        logging.info(f"Received response from OpenAI API for scene {start_time:.2f} - {end_time:.2f}")
        return msg.content
    except Exception as e:
        logging.error(f"Error in OpenAI API call for scene {start_time:.2f} - {end_time:.2f}: {str(e)}")
        raise

def process_scene(args):
    """Process a single scene"""
    frames, start_time, end_time = args
    logging.info(f"Processing scene from {start_time:.2f} to {end_time:.2f} seconds")
    base64_frames = [encode_frame(frame) for frame in frames]
    summary = scene_summarize(base64_frames, start_time, end_time)
    return start_time, end_time, summary

def generate_scene_summaries(video_path, scene_duration=7, frames_per_scene=1):
    """
    Generate summaries for scenes in a video using parallel processing.
    video_path: Path to the video file.
    scene_duration: Duration of each scene in seconds.
    frames_per_scene: Number of frames to analyze per scene.
    """
    logging.info(f"Starting to process video: {video_path}")
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {fps}, Total frames: {total_frames}")

    scenes_to_process = []
    frame_count = 0
    frames_per_interval = scene_duration * fps // frames_per_scene
    
    while frame_count < total_frames:
        scene_frames = []
        scene_start_time = frame_count / fps
        
        for _ in range(frames_per_scene):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            success, frame = video.read()
            if not success:
                break
            scene_frames.append(frame)
            frame_count += frames_per_interval

        if scene_frames:
            scene_end_time = min(frame_count / fps, total_frames / fps)
            scenes_to_process.append((scene_frames, scene_start_time, scene_end_time))

    video.release()
    logging.info(f"Video processing complete. Total scenes to process: {len(scenes_to_process)}")

    # Use ProcessPoolExecutor for parallel processing
    output_data = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_scene = {executor.submit(process_scene, args): args for args in scenes_to_process}
        for future in as_completed(future_to_scene):
            args = future_to_scene[future]
            try:
                start_time, end_time, summary = future.result()
                output_data.append([start_time, end_time, summary])
                logging.info(f"Processed scene from {start_time:.2f} to {end_time:.2f} seconds")
            except Exception as exc:
                logging.error(f'Scene processing generated an exception: {exc}')

    # Sort the output data by start time
    output_data.sort(key=lambda x: x[0])
    return output_data

def generate_police_report(summaries):
    """Generate a police report-style summary from all scene summaries"""
    logging.info("Generating research report")
    
    prompt = """
    <task_description>

    Generate a narrative-based paragraph summary of the input. 
    Do not reflect on the input with an eye toward using language that is grounded in value judgements.
    Use technical language. 
    Produce the output as if you were a robot describing a scene without emotions.
    If multiple scenes can be described briefly, prioritize brevity over repeating duplicate information.  
    Split your output into multiple paragraphs. 

    </task_description>
    """
    
    for start_time, end_time, summary in summaries:
        prompt += f"\nScene {start_time:.2f} - {end_time:.2f} seconds: {summary}\n"
    
    try:
        response = chat.invoke(
            [
                HumanMessage(content=prompt)
            ]
        )
        logging.info("Police report generated successfully")
        return response.content
    except Exception as e:
        logging.error(f"Error in generating police report: {str(e)}")
        raise

def save_to_csv(data, filename):
    """Save data to a CSV file"""
    logging.info(f"Saving data to CSV: {filename}")
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Start Time", "End Time", "Summary"])
        writer.writerows(data)

def save_to_text(data, filename):
    """Save data to a text file"""
    logging.info(f"Saving data to text file: {filename}")
    with open(filename, 'w', encoding='utf-8') as file:
        for start_time, end_time, summary in data:
            file.write(f"Scene: {start_time:.2f} - {end_time:.2f} seconds\n")
            file.write(f"{summary}\n\n")

if __name__ == "__main__":
    video_path = os.path.join(INPUT_DIR, "akers.mp4")
    logging.info(f"Starting script with video: {video_path}")
    
    try:
        output_data = generate_scene_summaries(video_path, scene_duration=7, frames_per_scene=7)

        csv_filename = os.path.join(OUTPUT_DIR, "video_scene_summaries.csv")
        save_to_csv(output_data, csv_filename)
        logging.info(f"Scene summaries saved to {csv_filename}")

        txt_filename = os.path.join(OUTPUT_DIR, "video_scene_summaries.txt")
        save_to_text(output_data, txt_filename)
        logging.info(f"Scene summaries saved to {txt_filename}")

        # Generate and save police report
        police_report = generate_police_report(output_data)
        report_filename = os.path.join(OUTPUT_DIR, "police_report.txt")
        with open(report_filename, 'w', encoding='utf-8') as file:
            file.write(police_report)
        logging.info(f"Police report saved to {report_filename}")

    except Exception as e:
        logging.error(f"An error occurred during script execution: {str(e)}")

    logging.info("Script execution completed")