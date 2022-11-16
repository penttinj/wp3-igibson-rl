import time
from torchvision import transforms
import torch
import numpy as np


class ObjectRecognition:
    def __init__(self) -> None:
        self.model = torch.hub.load(
            "pytorch/vision:v0.12.0", "mobilenet_v2", weights="MobileNet_V2_Weights.DEFAULT"
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("true true is cuda")
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print("Initialized MobileNetV2 model")

    def classify(self, img: np.ndarray) -> torch.Tensor:
        t0 = time.time_ns()
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
        with torch.no_grad():
            output = self.model(input_batch)
        # output[0] == Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        probs = torch.nn.functional.softmax(output[0], dim=0)
        t1 = time.time_ns()

        return probs
