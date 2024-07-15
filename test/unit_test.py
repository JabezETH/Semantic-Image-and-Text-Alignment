import json
import base64
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
import pytest
import requests_mock

from my_script import json_to_image, read_img, encode_image, describe_image, describe_images_from_json, get_position_from_constant, blend_images

# Test data
test_image_path = "test_image.jpg"
test_json_path = "test_image.json"
test_constant_json_path = "/home/jabez/week_12/Semantic-Image-and-Text-Alignment/backend/comparison_image.json"
test_output_json_path = "output.json"
test_larger_image_path = "larger_image.png"
test_small_images_paths = ["small_image_1.png", "small_image_2.png"]

# Mock data for images
mock_larger_image = np.zeros((500, 500, 4), dtype=np.uint8)
mock_smaller_image = np.zeros((100, 100, 4), dtype=np.uint8)


def test_json_to_image(mocker):
    # Mock Image.open
    mock_open = mocker.patch("PIL.Image.open", return_value=Image.new('RGB', (100, 100)))
    img = json_to_image(test_image_path)
    mock_open.assert_called_once_with(test_image_path)
    assert isinstance(img, Image.Image)

def test_encode_image(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data=b"image data"))
    encoded_image = encode_image(test_image_path)
    mock_open.assert_called_once_with(test_image_path, "rb")
    assert isinstance(encoded_image, str)

@pytest.fixture
def requests_mock_response():
    with requests_mock.Mocker() as m:
        yield m

def test_describe_image(requests_mock_response):
    base64_image = base64.b64encode(b"image data").decode('utf-8')
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a description of the image."
                }
            }
        ]
    }
    requests_mock_response.post("https://api.openai.com/v1/chat/completions", json=mock_response)
    description = describe_image(test_image_path)
    assert description == "This is a description of the image."

def test_describe_images_from_json(mocker, requests_mock_response):
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps({
        "larger_image_path": test_larger_image_path,
        "small_images_paths": test_small_images_paths
    })))

    mock_response = {
        "choices": [
            {
                "message": {
                    "content": "This is a description of the image."
                }
            }
        ]
    }
    requests_mock_response.post("https://api.openai.com/v1/chat/completions", json=mock_response)

    descriptions = describe_images_from_json(test_json_path)
    assert isinstance(descriptions, dict)
    assert "larger_image_description" in descriptions
    assert "small_images_descriptions" in descriptions

def test_get_position_from_constant():
    larger_image = mock_larger_image
    smaller_image = mock_smaller_image
    position = get_position_from_constant("top left", larger_image, smaller_image)
    assert position == (0, 0)

    position = get_position_from_constant("center center", larger_image, smaller_image)
    assert position == (200, 200)

    position = get_position_from_constant("bottom right", larger_image, smaller_image)
    assert position == (400, 400)

def test_blend_images(mocker):
    mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps({
        "larger_image_path": test_larger_image_path,
        "small_images_paths": test_small_images_paths
    })))

    mock_cv2_imread = mocker.patch("cv2.imread", side_effect=[mock_larger_image, mock_smaller_image, mock_smaller_image])
    mock_cv2_imwrite = mocker.patch("cv2.imwrite", return_value=True)
    mock_os_makedirs = mocker.patch("os.makedirs", return_value=True)

    position_names = ["top left", "center center"]
    brightness_values = [1.0, 1.5]
    layers = [1, 2]
    alpha = 0.5
    output_dir = './output'

    json_output_file = blend_images(position_names, brightness_values, layers, alpha, output_dir)

    mock_cv2_imread.assert_any_call(test_larger_image_path, cv2.IMREAD_UNCHANGED)
    mock_cv2_imread.assert_any_call(test_small_images_paths[0], cv2.IMREAD_UNCHANGED)
    mock_cv2_imread.assert_any_call(test_small_images_paths[1], cv2.IMREAD_UNCHANGED)
    mock_cv2_imwrite.assert_called_once_with(os.path.join(output_dir, 'blended_image.png'), mocker.ANY)
    mock_os_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    assert os.path.exists(json_output_file)
    with open(json_output_file, 'r') as f:
        output_data = json.load(f)
    assert "output_path" in output_data
    assert "positions" in output_data
    assert "brightness_values" in output_data
    assert "layers" in output_data
    assert output_data['output_path'] == os.path.join(output_dir, 'blended_image.png')
    assert output_data['positions'] == position_names
    assert output_data['brightness_values'] == brightness_values
    assert output_data['layers'] == layers

if __name__ == "__main__":
    pytest.main()
