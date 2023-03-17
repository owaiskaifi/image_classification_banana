import requests
from PIL import Image
import json
import io

# Load the test image
image = Image.open("mtailor_mlops_assessment-main/n01440764_tench.jpeg")

# Convert the image to bytes
image_bytes = io.BytesIO()
image.save(image_bytes, format="JPEG")
image_bytes.seek(0)

# Send the image to the server
response = requests.post("http://localhost:8000/", files={"image": image_bytes})

# Parse the predicted class probabilities from the server response
results = json.loads(response.text)
predictions = results["class"]
print(predictions)
