import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import concurrent.futures
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def setup_llm():
#     return ChatOpenAI(model="gpt-4o-mini")


def setup_llm():
    return ChatAnthropic(model="claude-3-haiku-20240307")

llm = setup_llm()


def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Mild denoising to preserve details
    denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 5, 5, 7, 21)
    
    # Subtle contrast enhancement using CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Mild sharpening
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Subtle bilateral filtering to reduce noise while preserving edges
    smooth = cv2.bilateralFilter(sharpened, 9, 25, 25)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(smooth)
    
    # Slight contrast and sharpness boost
    contrast_enhancer = ImageEnhance.Contrast(processed_image)
    contrast_enhanced = contrast_enhancer.enhance(1.1)
    sharpness_enhancer = ImageEnhance.Sharpness(contrast_enhanced)
    final_image = sharpness_enhancer.enhance(1.2)
    
    return final_image

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

image_prompt = """
Is anyone being sprayed with a canister? 

Write your description inside <input_analysis> tags. Begin your description with "The input depicts" or a similar phrase.
"""



verifier_template = """
<task_description>
Based on the explanation in the materials to analyze, determine whether or not the input describes an incident where force was used. 
Return 'true', or 'false'. Do not return any additional explanation or commentary.
Lean toward returning true, rather than false, if the input possibly describes an incident where force was used.
</task_description>

<materials_to_analyze>
{input}
</materials_to_analyze>
"""

def seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"

def process_frames(frames_data):
    timestamps, frames = zip(*frames_data)
    
    processed_images = []
    base64_frames = []
    
    for frame in frames:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image = preprocess_image(pil_image)
        processed_image = pil_image
        processed_images.append(processed_image)
        base64_frame = encode_frame(np.array(processed_image))
        base64_frames.append(base64_frame)
    
    # Save the processed images
    output_dir = "../data/output"
    os.makedirs(output_dir, exist_ok=True)
    for t, img in zip(timestamps, processed_images):
        output_filename = f"{output_dir}/frame_{seconds_to_minutes_seconds(t).replace(':', '_')}.jpg"
        img.save(output_filename)
    
    image_contents = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
        }
        for frame in base64_frames
    ]
    
    try:
        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": image_prompt},
                        *image_contents
                    ]
                )
            ]
        )
        verifier_response = ChatPromptTemplate.from_template(verifier_template)
        verification = verifier_response.format(input=msg.content)
        final_chain = verifier_response | llm | StrOutputParser()
        verification_result = final_chain.invoke({"input": msg.content})
        return (timestamps, msg.content, verification_result)
    except Exception as e:
        logging.error(f"Error in OpenAI API call for frames at times {[seconds_to_minutes_seconds(t) for t in timestamps]}: {str(e)}")
        return (timestamps, None, None)

def process_video_segment(video_path, start_time, end_time, frames_per_context=2, max_workers=10):
    start_time = 113.00
    end_time = 115.00

    logging.info(f"Processing video segment: {seconds_to_minutes_seconds(start_time)} - {seconds_to_minutes_seconds(end_time)}")
    
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    for t in np.arange(start_time, end_time, 0.01):  # 0.05 second intervals for 20 frames per second
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((t, frame))
    
    cap.release()

    # Group frames into batches of size frames_per_context
    frame_batches = [frames[i:i+frames_per_context] for i in range(0, len(frames), frames_per_context)]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frames = {executor.submit(process_frames, frame_batch): frame_batch for frame_batch in frame_batches}
        for future in concurrent.futures.as_completed(future_to_frames):
            frame_batch = future_to_frames[future]
            try:
                result = future.result()
                if result[1] is not None:  # Check if processing was successful
                    results.append(result)
            except Exception as exc:
                logging.error(f"Frame batch processing generated an exception: {exc}")

    # Sort results by timestamp
    results.sort(key=lambda x: x[0][0])  # Sort by the first timestamp in each batch
    
    # Convert timestamps to readable format
    readable_results = [
        ([seconds_to_minutes_seconds(t) for t in timestamps], content, verification)
        for timestamps, content, verification in results
    ]
    
    return readable_results

def analyze_video(video_path, start_time, end_time, frames_per_context=1):
    return process_video_segment(video_path, start_time, end_time, frames_per_context)