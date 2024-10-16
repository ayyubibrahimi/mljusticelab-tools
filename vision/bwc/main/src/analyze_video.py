
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


def setup_llm():
    return ChatOpenAI(model="gpt-4o-mini")


# def setup_llm():
#     return ChatAnthropic(model="claude-3-haiku-20240307")

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



image_template = """
<task_description>
You are a specialized analyst tasked with identifying potential use of force incidents in the input data. 
Your goal is to provide a clear, concise, and objective description of the scene, focusing on elements that could indicate use of force.
</task_description>

<instructions>
1. Describe the scene objectively, focusing on:
   - Number of individuals present in the frame (You do not need to see the entire profile of a person in the frame to include them. Partial persons include just their hands, legs, feet, etc. For example, do you see a hand extending into view from the corner of the screen?)
   - Positions and actions of individuals
   - Any physical contact or interactions between individuals
   - Presence of law enforcement officers or security personnel
   - Visible equipment or tools (e.g., handcuffs, batons, firearms)
   - Any signs of physical struggle or restraint

2. Use precise, factual language. Avoid interpretations or subjective assessments.

3. If the scene is unclear or ambiguous, state this explicitly.

4. Prioritize information relevant to potential use of force scenarios.
</instructions>

<output_format>

Provide your analysis in the following format:
<scene_description>
[Your objective description of the scene here]
</scene_description>

<potential_force_indicators>
[List any elements that could potentially indicate use of force]
</potential_force_indicators>

</output_format>
"""

verifier_template = """
<task_description>
You are a specialized analyst tasked with determining whether a given scene description matches any use of force incidents described in an official report. 
Your goal is to identify matches and return a list of matching elements.
</task_description>

<instructions>
1. Carefully review the provided scene description and potential force indicators.
2. Compare these elements to the types of force and incidents described in the official report.
3. Determine if there is a match between the scene and any incident in the report. 
</instructions>

<scene_analysis>
{input}
</scene_analysis>

<official_report>
{report_text}
</official_report>

<output_format>

Provide your analysis in the following format:
<match_determination>
[TRUE if there's a match, FALSE if there isn't]
</match_determination>

<matching_elements>
- If there is a match, return a bulletpoint list of elements from the scene that match the report. 
Do not speculate. Only consider matches that are explicitly present.
Use precise, factual language. Avoid interpretations or subjective analysis. 

- If there is no match, return "No matching elements"
</matching_elementsn>

</output_format>
"""

def seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"

def process_frames(frames_data, report_text):
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
    
    # Save the processed images to manually review
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
                        {"type": "text", "text": image_template},
                        *image_contents
                    ]
                )
            ]
        )
        verifier_response = ChatPromptTemplate.from_template(verifier_template)
        final_chain = verifier_response | llm | StrOutputParser()
        verification_result = final_chain.invoke({"input": msg.content, "report_text": report_text})
        return (timestamps, msg.content, verification_result)
    except Exception as e:
        logging.error(f"Error in OpenAI API call for frames at times {[seconds_to_minutes_seconds(t) for t in timestamps]}: {str(e)}")
        return (timestamps, None, None)

def process_video_segment(video_path, start_time, end_time, report_text, frames_per_context=1, time_step=1, max_workers=10):
    logging.info(f"Processing video segment: {seconds_to_minutes_seconds(start_time)} - {seconds_to_minutes_seconds(end_time)}")
    
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    for t in np.arange(start_time, end_time, time_step):  # Use the new time_step parameter here
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
        future_to_frames = {executor.submit(process_frames, frame_batch, report_text): frame_batch for frame_batch in frame_batches}
        for future in concurrent.futures.as_completed(future_to_frames):
            frame_batch = future_to_frames[future]
            try:
                result = future.result()
                if result[1] is not None:  # Check if processing was successful
                    results.append(result)
            except Exception as exc:
                logging.error(f"Frame batch processing generated an exception: {exc}")

    # Sort results by timestamp
    results.sort(key=lambda x: x[0][0])  
    
  # Convert timestamps to readable format and structure the output
    results = [
        {
            "timestamps": [seconds_to_minutes_seconds(t) for t in timestamps],
            "scene_description": content,
            "match_classification": verification
        }
        for timestamps, content, verification in results
    ]
    
    return results

def analyze_video(video_path, start_time, end_time, frames_per_context, report_text, time_step=1):
    return process_video_segment(video_path, start_time, end_time, report_text, frames_per_context, time_step)
