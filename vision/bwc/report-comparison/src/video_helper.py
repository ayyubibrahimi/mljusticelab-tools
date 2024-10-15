import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_llm():
    return ChatOpenAI(model="gpt-4o-mini")

# def setup_llm():
#     return ChatAnthropic(model="claude-3-haiku-20240307")

# def setup_llm():
#     return ChatGoogleGenerativeAI(model="gemini-1.5-flash")


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
    fps = 100  # 10 frames per second
    duration = end_time - start_time
    
    for t in np.arange(0, duration, 1/fps):
        frame_time = start_time + t
        frame = video.get_frame(frame_time)
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)
        
        processed_image = preprocess_image(pil_image)
        base64_frame = encode_frame(np.array(processed_image))
        frames.append(base64_frame)

    print(f"Processed {len(frames)} frames for scene {start_time:.2f} - {end_time:.2f}")

    prompt = f"""
    Analyze the following sequence of frames from a video scene.

    Think step by step about how the scene changes from one frame to the next.

    Is someone being pepper sprayed? 

    Write your description inside <image_analysis> tags. Begin your description with "The image depicts" or a similar phrase.
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