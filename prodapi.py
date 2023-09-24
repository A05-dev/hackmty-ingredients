from flask import Flask, request, jsonify
from roboflow import Roboflow
import cv2  # opencv
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    rf = Roboflow(api_key="5fLOPWRg1OT4LIUuGTsX")
    project = rf.workspace().project("sandoval_fridge")
    model = project.version(3).model

    prediction = model.predict(data["image_path"], confidence=40, overlap=30).json()

    # Load the image
    image_path = data["image_path"]
    image = cv2.imread(image_path)

    # Iterate through predictions and draw squares
    for prediction in data["predictions"]:
        x = prediction["x"]
        y = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))  # Random color
        thickness = 4

        # Draw a rectangle on the image with adjusted position
        cv2.rectangle(image, (x-height,y-width), (x+width,y+height), color, thickness)

    # Save or display the image with adjusted rectangles
    output_filename = "output.png"
    cv2.imwrite(output_filename, image)

    return jsonify({"prediction": prediction, "image": output_filename})

if __name__ == '__main__':
    app.run(debug=True)
