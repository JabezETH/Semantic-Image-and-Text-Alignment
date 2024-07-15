import json
import base64
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
import os
import autogen

from autogen import ConversableAgent



import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")




import json
import base64
import requests
from PIL import Image
from io import BytesIO

def json_to_image(image_path: str) -> Image.Image:
    """
    Converts an image file to a PIL Image object.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - PIL.Image.Image: Image as a PIL Image object.
    """
    return Image.open(image_path)

def read_img(json_path: str) -> dict:
    """
    Reads an image path from a JSON file, converts both a constant image and the image from JSON to JPEG, and sends them to the OpenAI API.

    Parameters:
    - json_path (str): Path to the JSON file containing the image path.
    - constant_json_path (str): Path to the JSON file containing the constant image path.
    - openai_api_key (str): OpenAI API key for authentication.

    Returns:
    - dict: Response from the OpenAI API.
    """
    constant_json_path = "/home/jabez/week_12/Semantic-Image-and-Text-Alignment/backend/comparison_image.json"
    # Load JSON data from files
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    with open(constant_json_path, 'r') as constant_json_file:
        constant_json_data = json.load(constant_json_file)
    
    # Extract the image paths from the JSON data
    image_path = json_data['output_path']
    constant_image_path = constant_json_data['comparison_image_path']

    # Convert image files to PIL Images
    constant_image = json_to_image(constant_image_path).convert("RGB")
    image = json_to_image(image_path).convert("RGB")

    # Convert images to base64 strings in JPEG format
    buffered1 = BytesIO()
    constant_image.save(buffered1, format="JPEG")
    base64_constant_image = base64.b64encode(buffered1.getvalue()).decode('utf-8')

    buffered2 = BytesIO()
    image.save(buffered2, format="JPEG")
    base64_image = base64.b64encode(buffered2.getvalue()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "compare the two image only object positions, brightness, and layer for each object. First rate 1 to 10 the similarity between image 1 and image 2. and suggest what to change in image 2 to make it like image 1"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_constant_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return response_data['choices'][0]['message']['content']

def json_to_image(image_path: str) -> Image.Image:
    """
    Converts an image file to a PIL Image object.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - PIL.Image.Image: Image as a PIL Image object.
    """
    return Image.open(image_path)

def encode_image(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path: str) -> str:
    """
    Describes an image by sending it to an API and getting the description.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - str: Description of the image.
    """
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

def describe_images_from_json(json_file_path: str) -> dict:
    """
    Describes images specified in a JSON file.

    Parameters:
    - json_file_path (str): Path to the JSON file containing image paths.

    Returns:
    - dict: Dictionary containing descriptions for the larger image and small images.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    larger_image_path = data.get('larger_image_path')
    small_images_paths = data.get('small_images_paths', [])

    # Describe larger image
    larger_image_description = describe_image(larger_image_path)

    # Describe small images
    small_images_descriptions = [
        {
            "image_path": image_path,
            "description": describe_image(image_path)
        }
        for image_path in small_images_paths
    ]

    return {
        "larger_image_description": larger_image_description,
        "small_images_descriptions": small_images_descriptions
    }


import cv2
import numpy as np
import os
import json

def get_position_from_constant(position_name, larger_image, smaller_image):
    """
    Get the (x, y) coordinates for a given position name based on the size of the larger and smaller images.

    Parameters:
    - position_name (str): Name of the position (e.g., "top right").
    - larger_image (numpy.ndarray): The larger image.
    - smaller_image (numpy.ndarray): The smaller image.

    Returns:
    - tuple: (x, y) coordinates for the given position.
    """
    larger_h, larger_w, _ = larger_image.shape
    smaller_h, smaller_w, _ = smaller_image.shape

    positions = {
        "top left": (0, 0),
        "top center": ((larger_w - smaller_w) // 2, 0),
        "top right": (larger_w - smaller_w, 0),
        "center left": (0, (larger_h - smaller_h) // 2),
        "center center": ((larger_w - smaller_w) // 2, (larger_h - smaller_h) // 2),
        "center right": (larger_w - smaller_w, (larger_h - smaller_h) // 2),
        "bottom left": (0, larger_h - smaller_h),
        "bottom center": ((larger_w - smaller_w) // 2, larger_h - smaller_h),
        "bottom right": (larger_w - smaller_w, larger_h - smaller_h),
    }

    return positions.get(position_name, (0, 0))

def blend_images(position_names: list, brightness_values: list, layers: list, alpha: float = 0.5, output_dir: str = './output') -> str:
    """
    Blends multiple small images by placing them on top of a larger image at specified positions and saves the blended image.
    
    Parameters:
    - position_names (list of str): List of position names for each small image, e.g., ["top left", "top right", "center center"].
    - brightness_values (list of float): List of brightness adjustment factors for each small image.
    - layers (list of int): List of layer values from 1 to 6 for each small image to determine the blending order.
    - alpha (float): Blending factor for transparency (0.0 to 1.0). Default is 0.5.
    - output_dir (str): Directory where the blended image will be saved. Default is './output'.
    
    Returns:
    - str: Path to the JSON file containing the output image information.
    """
    json_file_path = '/home/jabez/week_12/Semantic-Image-and-Text-Alignment/backend/constant_image.json'
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    larger_image_path = data['larger_image_path']
    small_images_paths = data['small_images_paths']

    print(f"Number of small images: {len(small_images_paths)}")
    print(f"Number of positions: {len(position_names)}")
    print(f"Number of brightness values: {len(brightness_values)}")
    print(f"Number of layers: {len(layers)}")

    if len(small_images_paths) != len(position_names):
        raise ValueError("The number of small images must match the number of position names.")
    
    if len(brightness_values) != len(small_images_paths):
        raise ValueError("The number of brightness values must match the number of small images.")
    
    if len(layers) != len(small_images_paths):
        raise ValueError("The number of layers must match the number of small images.")

    # Load larger image from file path
    larger_image = cv2.imread(larger_image_path, cv2.IMREAD_UNCHANGED)
    
    # Convert the larger image to RGBA if it is not already
    if larger_image.shape[2] != 4:
        larger_image = cv2.cvtColor(larger_image, cv2.COLOR_BGR2BGRA)

    # Create a blank image with the same size as the larger image
    blended_image = np.zeros_like(larger_image)
    blended_image[:, :, :] = larger_image

    # Sort the images by their layer values
    sorted_indices = sorted(range(len(layers)), key=lambda k: layers[k])

    # Blend each small image at its respective position in the order of layers
    for i in sorted_indices:
        small_image_path = small_images_paths[i]
        position_name = position_names[i]
        brightness_value = brightness_values[i]

        # Load small image from file path
        smaller_image = cv2.imread(small_image_path, cv2.IMREAD_UNCHANGED)

        # Convert the smaller image to RGBA if it is not already
        if smaller_image.shape[2] != 4:
            smaller_image = cv2.cvtColor(smaller_image, cv2.COLOR_BGR2BGRA)

        # Resize smaller image if necessary
        smaller_h, smaller_w, _ = smaller_image.shape
        larger_h, larger_w, _ = larger_image.shape
        if smaller_h > larger_h or smaller_w > larger_w:
            aspect_ratio = smaller_w / smaller_h
            if smaller_h > larger_h:
                smaller_h = larger_h
                smaller_w = int(smaller_h * aspect_ratio)
            if smaller_w > larger_w:
                smaller_w = larger_w
                smaller_h = int(smaller_w / aspect_ratio)
            smaller_image = cv2.resize(smaller_image, (smaller_w, smaller_h), interpolation=cv2.INTER_AREA)

        # Adjust the brightness of the smaller image
        hsv = cv2.cvtColor(smaller_image, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=brightness_value)
        smaller_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add an alpha channel manually
        alpha_channel = np.ones((smaller_image.shape[0], smaller_image.shape[1]), dtype=smaller_image.dtype) * 255
        smaller_image = cv2.merge((smaller_image, alpha_channel))

        # Get position for the smaller image
        x_offset, y_offset = get_position_from_constant(position_name, larger_image, smaller_image)

        # Overlay the smaller image onto the blended image
        for c in range(0, 3):
            blended_image[y_offset:y_offset+smaller_h, x_offset:x_offset+smaller_w, c] = (
                alpha * smaller_image[:, :, c] + (1 - alpha) * blended_image[y_offset:y_offset+smaller_h, x_offset:x_offset+smaller_w, c]
            )
        blended_image[y_offset:y_offset+smaller_h, x_offset:x_offset+smaller_w, 3] = smaller_image[:, :, 3]

    # Save blended image to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'blended_image.png')
    cv2.imwrite(output_file, blended_image)

    # Create JSON response
    json_data = {
        'output_path': output_file,
        'positions': position_names,
        'brightness_values': brightness_values,
        'layers': layers,
        'alpha': alpha
    }

    # Write JSON to file
    json_output_file = os.path.join(output_dir, 'output.json')
    with open(json_output_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    return json_output_file


llm_config2 = {"config_list": [{"model": "gpt-4", "api_key": openai_api_key}]}
code_execution_config = {"use_docker": False}

# Initialize the assistant agent with the given configurations
config_list = [
    {"model": "gpt-4", "api_key": openai_api_key, "api_type": "openai"},
]


llm_config={
        "temperature": 0,
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
             {
                        "name": "blend_images",
                        "description": "use this function to blend the images",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "position_names": {
                                    "type": "string",
                                    "description": "This is where you will position the blending"
                                },
                            },
                            "required": ["positions_str"]
                        }
                        },
             {
                        "name": "read_img",
                        "description": "use this to read the image blended from blended image",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "output_json": {
                                    "type": "object",
                                    "description": "This is the blended image"
                                },
                            },
                            "required": ["position"]
                        }
                        }
                        ],
}




import os
import autogen
from autogen import ConversableAgent

# Define the assistant agent that suggests tool calls.
img_blend_assistant = autogen.AssistantAgent(
    name="img_blend_assistant",
    code_execution_config=False,
    system_message="""You are a helpful AI assistant. 
The main problems you will be solving include:
- suggest different "positions", "brightness" and "layer" to make a good advertising based on the feedback from 'img_critic_assistant'
   - First read the images that are initiated using the describe_images_from_json function.
    make sure that the images will not overlap
    
- Your task:
    - Considering the the discription for each picture, find a way to position each picture to give good advertising based on the recommendation you got from 'img_critic_assistant'.
    - 'TERMINATE' when the image you blend got a rating 9.
- The number of small images must match the number of position names, layers and brightness.
- Example:
position_names = ["bottom center", "top right", "center center", "bottom left", "bottom center", "top center"]
layers = [
    6,  # Second layer from the top
    1,  # Topmost layer
    1,  # Third layer
    1,  # Fifth layer
    1,  # Sixth layer
    1   # Fourth layer
]
brightness_values = [2, 1.8, 1.2, 1.9, 1.1, 1.9]
    """,
    llm_config=llm_config2
)

img_critic_assistant = autogen.AssistantAgent(
    name="img_critic_assistant",
    code_execution_config=False,
    system_message="""You are an advertising image critic AI assistant. 
Your task is to critique the 'output.json' from 'img_blend_assistant'.
Recommend 'img_blend_assistant' for better advertising by comparing it to image 1, which is a good advertisement on the above metrics.
Rate the the blended image from 1 to 10 considering the brightness, position and layer.
Return 'TERMINATE' when the task is done.""",
    llm_config=llm_config2
)

# The user proxy agent is used for interacting with the assistant agent and executes tool calls.
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="Executor. Execute the functions recommended by the assistants.",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    code_execution_config=False
)

# Register functions for execution
img_blend_assistant.register_for_llm(name="blend_images", description="Image blender")(blend_images)
img_blend_assistant.register_for_llm(name="describe_images_from_json", description="describe_images_from_json")(describe_images_from_json)
img_critic_assistant.register_for_llm(name="read_img", description="Image reader")(read_img) 
user_proxy.register_for_execution(name="blend_images")(blend_images)
user_proxy.register_for_execution(name="read_img")(read_img)
user_proxy.register_for_execution(name="describe_images_from_json")(describe_images_from_json)


# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, img_blend_assistant, img_critic_assistant],
    messages=[],  # The initial messages in the chat
    max_round=10  # Maximum rounds of conversation
)

# Create group chat manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config2
)




def execute_agents():
    message = user_proxy.initiate_chat(
    manager, message="blend this images '/home/jabez/week_12/Semantic-Image-and-Text-Alignment/backend/constant_image.json' ")