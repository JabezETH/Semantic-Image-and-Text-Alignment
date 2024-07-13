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
    - openai_api_key (str): OpenAI API key for authentication.

    Returns:
    - dict: Response from the OpenAI API.
    """
    # Constant image path
    constant_image_path = "/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/_preview.png"

    # Load JSON data from file
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    # Extract the image path from the JSON data
    image_path = json_data['output_path']

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
                        "text": "compare the two image object positions. and suggest what to change in image 2 to make it like image 1"
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



def get_position_from_constant(position_name, larger_image, smaller_image):
    """
    Get the (x, y) coordinates for a given position name based on the size of the larger and smaller images.

    Parameters:
    - position_name (str): Name of the position (e.g., "top right").
    - larger_image (PIL.Image.Image): The larger image.
    - smaller_image (PIL.Image.Image): The smaller image.

    Returns:
    - tuple: (x, y) coordinates for the given position.
    """
    larger_w, larger_h = larger_image.size
    smaller_w, smaller_h = smaller_image.size

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

def blend_images(position_names: list, alpha: float = 0.5, output_dir: str = './output') -> str:
    """
    Blends multiple small images by placing them on top of a larger image at specified positions without overlapping and saves the blended image.

    Parameters:
    - position_names (list of str): List of position names for each small image, e.g., ["top left", "top right", "center center"].
    - alpha (float): Blending factor for transparency (0.0 to 1.0). Default is 0.5.
    - output_dir (str): Directory where the blended image will be saved. Default is './output'.

    Returns:
    - str: Path to the JSON file containing the output image information.
    """
    # Define paths to small images (example paths)
    small_images_paths = [
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/cta.jpg',
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/discover.png',
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/endframe_3.png',
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/engagement_animation_1.png',
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/engagement_instruction_1.png',
        '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/landing_endframe.jpg'
    ]

    if len(small_images_paths) != len(position_names):
        raise ValueError("The number of small images must match the number of position names.")

    # Load larger image from file path (example path)
    larger_image_path = '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/data/Assets/015efcdd8de3698ffc4dad6dabd6664a/endframe_1.jpg'
    larger_image = Image.open(larger_image_path)
    larger_image = larger_image.convert("RGBA")  # Ensure the larger image has an alpha channel

    # Create a blank image with the same size as the larger image
    blended_image = Image.new("RGBA", larger_image.size)

    # Paste the larger image onto the blank image
    blended_image.paste(larger_image, (0, 0))

    # Blend each small image at its respective position
    for i, small_image_path in enumerate(small_images_paths):
        # Load small image from file path
        smaller_image = Image.open(small_image_path)
        smaller_image = smaller_image.convert("RGBA")  # Ensure the smaller image has an alpha channel

        # Resize smaller image if necessary
        smaller_w, smaller_h = smaller_image.size
        larger_w, larger_h = larger_image.size
        if smaller_h > larger_h or smaller_w > larger_w:
            aspect_ratio = smaller_w / smaller_h
            if smaller_h > larger_h:
                smaller_h = larger_h
                smaller_w = int(smaller_h * aspect_ratio)
            if smaller_w > larger_w:
                smaller_w = larger_w
                smaller_h = int(smaller_w / aspect_ratio)
            smaller_image = smaller_image.resize((smaller_w, smaller_h))

        # Make the smaller image semi-transparent
        enhancer = ImageEnhance.Brightness(smaller_image)
        smaller_image = enhancer.enhance(alpha)

        # Get position for the smaller image
        x_offset, y_offset = get_position_from_constant(position_names[i], larger_image, smaller_image)

        # Paste the smaller image onto the blended image at the calculated position
        blended_image.paste(smaller_image, (x_offset, y_offset), smaller_image)

    # Save blended image to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'blended_image.png')
    blended_image.save(output_file)

    # Create JSON response
    json_data = {
        'output_path': output_file,
        'positions': position_names,
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
- suggest different "positions" to make a good advertising based on the feedback from 'img_critic_assistant'
    use this as example: "List of position names for each small image, e.g., ["top left", "top right", "center center", "bottom left", "bottom center", "bottom right"]"
    make sure you are giving only 6 positions
    make sure that the images will not overlap
- picture positions: ['shop now','Discover 12 unique tea flavours delivered to your door', 'Enjoy tea delivered to your home', hand pointing, 'tap to get letter box delivery of tea','off black generation picture']
- Your task:
    - Considering the above descriptions for each picture, find a way to position each picture to give good advertising based on the recommendation you got from 'img_critic_assistant'.
    - 'TERMINATE' when the image you blend looks like the feedback.
    """,
    llm_config=llm_config2
)

img_critic_assistant = autogen.AssistantAgent(
    name="img_critic_assistant",
    code_execution_config=False,
    system_message="""You are an advertising image critic AI assistant. 
Your task is to critique the 'output.json' from 'img_blend_assistant'.
positions: ['shop now','Discover 12 unique tea flavours delivered to your door', 'Enjoy tea delivered to your home', hand pointing, 'tap to get letter box delivery of tea','off black generation picture']
Recommend 'img_blend_assistant' for better advertising by comparing it to image 1, which is a good advertisementand critic on the above image positions only.
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
img_critic_assistant.register_for_llm(name="read_img", description="Image reader")(read_img) 
user_proxy.register_for_execution(name="blend_images")(blend_images)
user_proxy.register_for_execution(name="read_img")(read_img)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, img_blend_assistant, img_critic_assistant],
    messages=[],  # The initial messages in the chat
    max_round=15  # Maximum rounds of conversation
)

# Create group chat manager
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config2
)


def excute_agents():
    message = user_proxy.initiate_chat(
    manager, message="blend the images at diffrent positions")