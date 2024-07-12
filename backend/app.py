from flask import Flask, send_from_directory, jsonify, request
import os
from flask_cors import CORS
import sys

sys.path.insert(1, '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/scripts')
import autogen_agents

app = Flask(__name__)
CORS(app)

# Directory where images are stored
IMAGE_DIR = '/home/jabez_kassa/week_12_updated/Semantic-Image-and-Text-Alignment/notebooks/output'
latest_image = None

@app.route('/api/image', methods=['GET'])
def get_image():
    global latest_image
    if latest_image and os.path.exists(os.path.join(IMAGE_DIR, latest_image)):
        return send_from_directory(IMAGE_DIR, latest_image)
    else:
        return jsonify({"error": "No image found"}), 404

@app.route('/api/generate', methods=['POST'])
def generate_image():
    global latest_image
    # Your logic to generate or modify an image
    autogen_agents.excute_agents()
    images = os.listdir(IMAGE_DIR)
    if images:
        latest_image = max(images, key=lambda x: os.path.getctime(os.path.join(IMAGE_DIR, x)))
        return jsonify({"message": "Image generated successfully!"})
    else:
        return jsonify({"error": "Failed to generate image"}), 500

if __name__ == '__main__':
    app.run(debug=True)
