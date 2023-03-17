

import banana_dev as banana
import base64

# Set your Banana Dev API key and model key
api_key = "3e8af2c6-4600-4d52-b82a-c00bf74247f5"
model_key = "a08eaef9-d735-4b3b-a04f-b257ae0891e0"

# Load the image as a binary file
with open("./n01440764_tench.jpeg", "rb") as f:
    image_bytes = f.read()

# Encode the image as a base64 string
image_base64 = base64.b64encode(image_bytes).decode()

# Define the model inputs
model_inputs = {"image": image_base64}

# Run the model
out = banana.run(api_key, model_key, model_inputs)

# Print the model output
print(out)
