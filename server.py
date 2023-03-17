import json
import io
import numpy as np
from PIL import Image
import onnxruntime
from sanic import Sanic
from sanic.response import json as sanic_json

# Load the ONNX ResNet model
session = onnxruntime.InferenceSession("./resnet.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create a Sanic server
app = Sanic(__name__)

# Define an endpoint for the model inference
@app.route("/", methods=["POST"])
async def predict(request):
    # Get the input image as bytes
    # print
    image_bytes = request.files["image"][0].body

    # Convert the image bytes to a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))
    

    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image).transpose(2, 0, 1)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Run the inference
    inputs = {input_name: image}
    outputs = session.run([output_name], inputs)
    output = outputs[0][0]

    class_idx = np.argmax(output)
     

    # Return the predicted class as a JSON object
    return sanic_json({"class": int(class_idx)})

# Start the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
