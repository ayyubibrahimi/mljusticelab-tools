import os
import base64
import logging
from io import BytesIO
from dotenv import find_dotenv, load_dotenv
from PIL import Image
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm
from openai import BadRequestError
import textgrad as tg

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
TARGET_IMAGE_SIZE = (800, 800)
TARGET_ENCODED_SIZE = 100000 

INPUT_DIR = "../data/input"
OUTPUT_DIR = "../data/output"

def load_and_preprocess_data(csv_path, pdf_path):
    """Load and preprocess data from CSV and PDF"""
    df = pd.read_csv(csv_path)
    images = convert_from_path(pdf_path)

    processed_data = []
    for _, row in df.iterrows():
        image = images[row['page_no'] - 1]

        resized_image = resize_image(image)
        try:
            base64_image = encode_image_with_size_control(resized_image)
            processed_data.append((base64_image, int(row['label'])))
        except ValueError as e:
            logging.warning(f"Error processing image on page {row['page_no']}: {str(e)}")
            continue
    return processed_data

def resize_image(image):
    """Resize the image to fit within the target size"""
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        new_width = min(image.width, TARGET_IMAGE_SIZE[0])
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(image.height, TARGET_IMAGE_SIZE[1])
        new_width = int(new_height * aspect_ratio)
    
    return image.resize((new_width, new_height), Image.LANCZOS)

def encode_image_with_size_control(image, target_size=TARGET_ENCODED_SIZE, initial_quality=95):
    """Encode an image to base64 with size control"""
    quality = initial_quality
    while quality > 20:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality, optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        if len(img_str) <= target_size:
            return img_str.decode("utf-8")
        quality -= 5
    
    # If we can't meet the target size, resize the image further
    while len(img_str) > target_size:
        width, height = image.size
        image = image.resize((int(width*0.9), int(height*0.9)), Image.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=20, optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
    
    return img_str.decode("utf-8")

def check_encoded_image_size(encoded_image):
    size_bytes = len(encoded_image.encode('utf-8'))
    logging.info(f"Encoded image size: {size_bytes} bytes")
    return size_bytes

def eval_dataset(test_set, eval_fn, model, max_samples=10):
    accuracy_list = []
    for sample in test_set[:max_samples]:
        x, y = sample
        x = tg.Variable(x, requires_grad=False, role_description="Input: base64 encoded image. Task: binary classification of photo or non-photo. Return 1 for photo and 0 otherwise.")
        y = tg.Variable(y, requires_grad=False, role_description="Input: groundtruth label. Task: return 1 for photo, 0 otherwise.")
        response = model(x)
        accuracy = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        accuracy_list.append(int(accuracy.value))
    return accuracy_list

def run_validation_revert(system_prompt, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1]) if results["validation_acc"] else 0
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1] if results["prompt"] else system_prompt.value

    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


def initialize_components():
    llm_api_eval = tg.get_engine(engine_name="gpt-4o-mini")
    llm_api_test = tg.get_engine(engine_name="gpt-4o-mini")
    tg.set_backward_engine(llm_api_eval, override=True)

    STARTING_SYSTEM_PROMPT = """
    <task_description>
    Analyze the input image and determine whether it is a photograph or not. This is a binary classification task. Return True if the input is a photograph, and False otherwise. Do not provide any additional explanation or commentary.
    </task_description>

    <context>
    This classification will be used to quickly sort inputs into photographs and non-photographs. The majority of inputs may be documents or other non-photographic images, but the focus is on accurately identifying true photographs when they occur.
    </context>

    <classification_categories>
    1. Photograph: True
    2. Non-Photograph: False
    </classification_categories>

    <thinking_process>
    1. Observe the visual characteristics of the input.
    2. Identify key features that suggest whether it's a photograph or not.
    3. Consider any ambiguities or edge cases.
    4. Make a final determination based on the overall assessment.
    5. Return only True or False based on this determination.
    </thinking_process>

    <classification_guidelines>

    <photograph_indicators>
    - Realistic representation of real-world scenes or objects
    - Natural lighting, shadows, and textures
    - Depth of field and focus typical of camera lenses
    - Presence of photographic artifacts (e.g., lens flare, motion blur)
    </photograph_indicators>

    <non_photograph_indicators>
    - Stylized or abstract representations
    - Predominantly text-based content (e.g., documents, screenshots of text)
    - Graphical elements like charts, diagrams, or user interface components
    - Hand-drawn or digitally created illustrations
    - Computer-generated imagery or renderings
    </non_photograph_indicators>

    </classification_guidelines>

    <additional_instructions>
    - If the input is ambiguous, classify it based on the predominant characteristics.
    - Do not provide any explanation or reasoning in the output.
    - If no image is provided or there are technical issues preventing analysis, return False.
    </additional_instructions>

    <output_format>
    [True/False]
    </output_format>
    """

    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="System prompt for binary image classification [return 1 if the input is a photo, and 0 otherwise]")

    model = tg.BlackboxLLM(llm_api_test, system_prompt)
    model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

    OPTIMIZER_SYSTEM_PROMPT = """
        <system_instructions>
        You are an AI assistant specialized in optimizing prompts for image classification tasks. Your current task is to refine and improve a given prompt used for distinguishing between photographs and non-photographs in a binary classification system.
        </system_instructions>

        <optimization_goals>
        1. Enhance the prompt's ability to accurately differentiate between photographs and non-photographs.
        2. Maintain the binary nature of the classification (True for photographs, False for non-photographs).
        3. Improve clarity and precision in the instructions without increasing complexity unnecessarily.
        4. Ensure the prompt remains focused on the core task without introducing extraneous elements.
        </optimization_goals>

        <optimization_process>
        1. Carefully analyze the given prompt, paying attention to its structure and content.
        2. Identify areas that could be improved or clarified to better guide the classification process.
        3. Consider edge cases and potential ambiguities that the current prompt might not adequately address.
        4. Propose specific, targeted changes to enhance the prompt's effectiveness.
        5. Ensure that any modifications align with the provided constraints and maintain the required output format.
        </optimization_process>

        <key_considerations>
        - Focus on refining the classification guidelines and indicators for both photographs and non-photographs.
        - Consider adding or modifying examples to better illustrate the distinction between categories.
        - Evaluate whether the current thinking process could be improved to lead to more accurate classifications.
        - Ensure that the prompt remains concise and directly applicable to the task at hand.
        </key_considerations>

        <output_instructions>
        Provide your suggested improvements in a clear, structured manner. Explain the rationale behind each significant change you propose. Do not rewrite the entire prompt unless absolutely necessary; instead, focus on targeted enhancements that will have the most impact on classification accuracy.
        </output_instructions>
        """

    constraints = [
        "The prompt must return a binary numerical value of True or False. There must be no output except the binary value or True or False",
    ]

    in_context_examples = [
        """
    <task_description>
    Analyze the input image and determine whether it is a photograph or not. This is a binary classification task. Return True if the input is a photograph, and False otherwise. Do not provide any additional explanation or commentary.
    </task_description>

    <context>
    This classification will be used to quickly sort inputs into photographs and non-photographs. The majority of inputs may be documents or other non-photographic images, but the focus is on accurately identifying true photographs when they occur.
    </context>

    <classification_categories>
    1. Photograph: True
    2. Non-Photograph: False
    </classification_categories>

    <thinking_process>
    1. Observe the visual characteristics of the input.
    2. Identify key features that suggest whether it's a photograph or not.
    3. Consider any ambiguities or edge cases.
    4. Make a final determination based on the overall assessment.
    5. Return only True or False based on this determination.
    </thinking_process>

    <classification_guidelines>

    <photograph_indicators>
    - Realistic representation of real-world scenes or objects
    - Natural lighting, shadows, and textures
    - Depth of field and focus typical of camera lenses
    - Presence of photographic artifacts (e.g., lens flare, motion blur)
    </photograph_indicators>

    <non_photograph_indicators>
    - Stylized or abstract representations
    - Predominantly text-based content (e.g., documents, screenshots of text)
    - Graphical elements like charts, diagrams, or user interface components
    - Hand-drawn or digitally created illustrations
    - Computer-generated imagery or renderings
    </non_photograph_indicators>

    </classification_guidelines>

    <additional_instructions>
    - If the input is ambiguous, classify it based on the predominant characteristics.
    - Do not provide any explanation or reasoning in the output.
    - If no image is provided or there are technical issues preventing analysis, return False.
    </additional_instructions>

    <output_format>
    [True/False] Do not provide any additional explanation or commentary.
    </output_format>
    """, 
    """
    <task_description>
        Analyze the input image and determine whether it is a photograph or not. This is a binary classification task. Return True if the input is a photograph, and False otherwise. Do not provide any additional explanation or commentary.
        </task_description>
        <context>
        This classification will be used to quickly sort inputs into photographs and non-photographs. The goal is to accurately identify true photographs, distinguishing them from other types of images such as digital art, screenshots, or scanned documents.
        </context>
        <classification_categories>

        Photograph: True
        Non-Photograph: False
        </classification_categories>

        <thinking_process>

        Examine the overall visual characteristics of the input.
        Look for specific photographic elements and qualities.
        Assess the presence of non-photographic features.
        Weigh the evidence for and against the image being a photograph.
        Make a final determination and return the corresponding binary value.
        </thinking_process>

        <classification_guidelines>
        <photograph_indicators>

        Captured moment of real-world scene or subject
        Natural variations in lighting, including highlights and shadows
        Photographic depth of field effects (bokeh, focus gradients)
        Presence of camera-specific artifacts (lens flare, chromatic aberration)
        Continuous tone and color gradations typical of photographic processes
        Realistic textures and details consistent with photographic capture
        </photograph_indicators>

        <non_photograph_indicators>

        Vector graphics or sharp, perfectly geometric shapes
        Pixel-level patterns inconsistent with photographic noise
        Text overlays or user interface elements (unless photographed)
        Digital illustrations or 3D renderings
        Scanned documents or handwritten content
        Screenshots of software or digital content
        </non_photograph_indicators>

        </classification_guidelines>
        <additional_instructions>

        For composite images, classify as a photograph only if the base layer is clearly photographic.
        Heavily edited photographs should still be classified as photographs if the core content is photographic.
        Do not let the subject matter influence the classification; focus on the medium.
        If no image is provided or there are technical issues preventing analysis, return False.
        </additional_instructions>

        <output_format>
        Return True if the input is a photograph, and False otherwise.
        Do not return anything but the binary value True or False. Do not provide any additional explanation or commentary.
        </output_format>
    """,
    """
    <task_description>
            Determine if the input image is a photograph. Return True for a photograph, False for a non-photograph. Provide no explanation.
            </task_description>
            <context>
            This task aims to rapidly categorize images as photographs or non-photographs, focusing on the medium rather than the content.
            </context>
            <classification_categories>

            Photograph: True
            Non-Photograph: False
            </classification_categories>

            <thinking_process>

            Analyze image characteristics.
            Identify photographic vs non-photographic elements.
            Make a binary decision.
            Return the corresponding value.
            </thinking_process>

            <classification_guidelines>
            <photograph_indicators>

            Realistic capture of physical world
            Photographic lighting and shadow details
            Camera-specific visual qualities (focus, exposure)
            Natural imperfections and variations
            </photograph_indicators>

            <non_photograph_indicators>

            Digital graphics or illustrations
            Text-heavy content or user interfaces
            Perfectly uniform colors or patterns
            Clear evidence of digital creation or manipulation
            </non_photograph_indicators>

            </classification_guidelines>
            <additional_instructions>

            Classify based on primary image characteristics.
            Ignore subject matter; focus on medium.
            For ambiguous cases, lean towards non-photograph False.
            Return False if image is missing or unanalyzable.
            </additional_instructions>

            <output_format>
            [True/False] Do not provide any additional explanation or commentary.
            </output_format>
    """, 
    """<task_description>
            Classify input as photograph True or non-photograph False. No explanation.
            </task_description>

            <context>
            Rapid sorting of images into photo/non-photo categories.
            </context>

            <classification_categories>
            1. Photo: True
            2. Non-Photo: False
            </classification_categories>

            <thinking_process>
            1. Assess image.
            2. Identify key features.
            3. Decide.
            4. Return True or False.
            </thinking_process>

            <classification_guidelines>
            <photo_indicators>
            - Real-world capture
            - Natural lighting/shadows
            - Photographic artifacts
            </photo_indicators>

            <non_photo_indicators>
            - Digital creation
            - Text/UI elements
            - Artificial patterns
            </non_photo_indicators>
            </classification_guidelines>

            <additional_instructions>
            - Focus on medium, not content.
            - If unsure or no image, return False.
            </additional_instructions>

            <output_format>
            Return True if the input is a photograph, and False otherwise.
            Do not return anything but the binary value True or False. Do not provide any additional explanation or commentary.
            </output_format>"""
    ]


    optimizer = tg.TextualGradientDescent(
        engine=llm_api_eval,
        parameters=[system_prompt],
        optimizer_system_prompt=OPTIMIZER_SYSTEM_PROMPT,
        constraints=constraints,
        in_context_examples=in_context_examples
    )

    return system_prompt, model, model_evaluation, optimizer

def create_eval_function():
    def eval_fn(inputs):
        prediction = inputs['prediction']
        ground_truth = inputs['ground_truth_answer']
        return tg.Variable(int(prediction.value == ground_truth.value), 
                           requires_grad=True,
                           role_description="Evaluation result (1 for correct, 0 for incorrect)")
    return eval_fn

def process_file(csv_file, system_prompt, model, eval_fn, optimizer):
    csv_path = os.path.join(INPUT_DIR, csv_file)
    pdf_path = os.path.join(INPUT_DIR, os.path.splitext(csv_file)[0] + '.pdf')

    if not os.path.exists(pdf_path):
        logging.warning(f"PDF file not found for {csv_file}. Skipping this file.")
        return None

    logging.info(f"Processing file: {csv_file}")

    data = load_and_preprocess_data(csv_path, pdf_path)

    train_set, val_set, test_set = split_data(data)
    results = {"test_acc": [], "prompt": [], "validation_acc": []}
    
    train_model(train_set, val_set, system_prompt, model, eval_fn, optimizer, results)

    
    test_acc = evaluate_model(test_set, eval_fn, model, max_samples=100)
    
    return test_acc, system_prompt.get_value()

def split_data(data):
    np.random.shuffle(data)
    train_split = int(0.7 * len(data))
    val_split = int(0.85 * len(data))
    return data[:train_split], data[train_split:val_split], data[val_split:]

def train_model(train_set, val_set, system_prompt, model, eval_fn, optimizer, results):
    num_epochs = 3
    patience = 5
    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_epoch(train_set, system_prompt, model, eval_fn, optimizer, val_set, results)
        val_acc = evaluate_model(val_set, eval_fn, model, max_samples=5)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered. Best validation accuracy: {best_val_acc:.4f}")
            break

def train_epoch(train_set, system_prompt, model, eval_fn, optimizer, val_set, results, batch_size=20):
    for i in range(0, len(train_set), batch_size):
        batch = train_set[i:i+batch_size]
        optimizer.zero_grad()
        batch_loss = process_batch(batch, model, eval_fn)
        if batch_loss:
            batch_loss.backward()
            optimizer.step()

            # Run validation after each batch
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)

def process_batch(batch, model, eval_fn):
    losses = []
    for x, y in batch:
        try:
            x, y = prepare_input(x, y)
            response = model(x)
            loss = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            losses.append(loss)
        except BadRequestError as e:
            handle_bad_request_error(e)
    
    if losses:
        return tg.sum(losses)
    return None

def process_single_image(x, y, model, eval_fn):
    try:
        x, y = prepare_input(x, y)
        response = model(x)
        loss = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return loss
    except BadRequestError as e:
        handle_bad_request_error(e)
        return None

def prepare_input(x, y):
    return (tg.Variable(x, requires_grad=False, role_description="Input: base64 encoded image. Task: binary classification of photo or non-photo. Return 1 for photo and 0 otherwise."),
            tg.Variable(y, requires_grad=False, role_description="Input: groundtruth label. Task: return 1 for photo, 0 otherwise."))


def handle_bad_request_error(e):
    if "context_length_exceeded" in str(e):
        logging.warning(f"Context length exceeded for image. Skipping this image.")
    else:
        raise e

def update_gradients(system_prompt, recent_gradients):
    grad_text = system_prompt.get_gradient_text()
    recent_gradients.append(grad_text)


def evaluate_model(dataset, eval_fn, model, max_samples):
    return np.mean(eval_dataset(dataset, eval_fn, model, max_samples=max_samples))

def main():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {INPUT_DIR}")

    system_prompt, model, model_evaluation, optimizer = initialize_components()
    eval_fn = create_eval_function()

    results = {"test_acc": [], "prompt": [], "validation_acc": []}

    for csv_file in csv_files:
        result = process_file(csv_file, system_prompt, model, eval_fn, optimizer)
        if result:
            test_acc, final_prompt = result
            results["test_acc"].append(test_acc)
            results["prompt"].append(final_prompt)
            results["validation_acc"].append(test_acc)  

            logging.info(f"Completed processing file: {csv_file}")
            logging.info(f"Current test accuracy: {test_acc:.4f}")
            logging.info(f"Current prompt: {final_prompt}")

    print("Optimization complete. Final prompt:")
    print(system_prompt.value)
    print(f"Final test accuracy: {results['test_acc'][-1]:.4f}")

if __name__ == "__main__":
    main()