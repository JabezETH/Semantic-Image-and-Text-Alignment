from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import json
from werkzeug.utils import secure_filename
import logging
import sys

sys.path.insert(1, '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/scripts')
import autogen_agents

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Directory where the Flask app is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'uploaded_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CONSTANT_JSON_PATH = os.path.join(BASE_DIR, 'constant_image.json')
COMPARISON_JSON_PATH = os.path.join(BASE_DIR, 'comparison_image.json')
latest_image = None  # Variable to store the latest generated image path

# Ensure the image directory and output directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_images():
    logging.debug("Upload endpoint called")

    # Check if the request contains a large image
    if 'large_image' not in request.files:
        logging.debug("No large image file in request")
        return jsonify({"error": "No large image file"}), 400

    large_image = request.files['large_image']
    if large_image.filename == '':
        logging.debug("No selected large image file")
        return jsonify({"error": "No selected large image file"}), 400

    large_image_path = os.path.join(IMAGE_DIR, secure_filename(large_image.filename))

    # Save the large image
    try:
        large_image.save(large_image_path)
        logging.debug(f"Saved large image to {large_image_path}")

        # Initialize JSON data with the large image path
        json_data = {
            "larger_image_path": large_image_path,
            "small_images_paths": []
        }

        # Handle small images
        for key in request.files:
            if key.startswith('small_image'):
                small_image = request.files[key]
                if small_image.filename != '':
                    small_image_path = os.path.join(IMAGE_DIR, secure_filename(small_image.filename))
                    small_image.save(small_image_path)
                    logging.debug(f"Saved small image to {small_image_path}")
                    json_data["small_images_paths"].append(small_image_path)

        # Update the JSON file with the new image paths
        with open(CONSTANT_JSON_PATH, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        logging.debug(f"Updated JSON file {CONSTANT_JSON_PATH} with {json_data}")

        return jsonify({"message": "Files uploaded successfully!"}), 200

    except Exception as e:
        logging.error(f"Error saving images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_image():
    global latest_image  # Declare global variable at the start of the function
    try:
        # Call the function that generates the image
        autogen_agents.execute_agents()
        # Update the path to the latest generated image
        latest_image = os.path.join(OUTPUT_DIR, 'blended_image.png')
        logging.debug(f"Generated image saved to {latest_image}")

        return jsonify({"message": "Image generated successfully!"}), 200

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/image', methods=['GET'])
def get_image():
    global latest_image  # Declare global variable at the start of the function
    if latest_image and os.path.exists(latest_image):
        return send_from_directory(OUTPUT_DIR, 'blended_image.png')
    else:
        return jsonify({"error": "No image found"}), 404

@app.route('/api/upload-comparison', methods=['POST'])
def upload_comparison_image():
    logging.debug("Upload comparison endpoint called")

    # Check if the request contains a comparison image
    if 'comparison_image' not in request.files:
        logging.debug("No comparison image file in request")
        return jsonify({"error": "No comparison image file"}), 400

    comparison_image = request.files['comparison_image']
    if comparison_image.filename == '':
        logging.debug("No selected comparison image file")
        return jsonify({"error": "No selected comparison image file"}), 400

    comparison_image_path = os.path.join(IMAGE_DIR, secure_filename(comparison_image.filename))

    # Save the comparison image
    try:
        comparison_image.save(comparison_image_path)
        logging.debug(f"Saved comparison image to {comparison_image_path}")

        # Update the JSON file with the comparison image path
        comparison_data = {
            "comparison_image_path": comparison_image_path
        }
        with open(COMPARISON_JSON_PATH, 'w') as json_file:
            json.dump(comparison_data, json_file, indent=2)
        logging.debug(f"Updated JSON file {COMPARISON_JSON_PATH} with {comparison_data}")

        return jsonify({"message": "Comparison image uploaded successfully!"}), 200

    except Exception as e:
        logging.error(f"Error saving comparison image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete-images', methods=['POST'])
def delete_images():
    global latest_image  # Declare global variable at the start of the function
    try:
        # Remove all files in the IMAGE_DIR
        for filename in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Remove the generated image
        if latest_image and os.path.isfile(latest_image):
            os.unlink(latest_image)

        # Reset latest_image
        latest_image = None

        # Clear the JSON files
        with open(CONSTANT_JSON_PATH, 'w') as json_file:
            json.dump({}, json_file)
        with open(COMPARISON_JSON_PATH, 'w') as json_file:
            json.dump({}, json_file)

        return jsonify({"message": "All images deleted successfully!"}), 200

    except Exception as e:
        logging.error(f"Error deleting images: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
