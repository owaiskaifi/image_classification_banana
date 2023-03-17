# In this file, we define download_model
# It runs during container build time to get model weights built into the container

 

import torch

# Option 1: passing weights param as string
# model = torch.hub.load("pytorch/vision", "resnet50", weights="resnet50.pth")
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = torch.hub.load("pytorch/vision", "resnet50", weights="resnet50.pth")
    torch.save(model.state_dict(), "resnet50.pth")
if __name__ == "__main__":
    download_model()