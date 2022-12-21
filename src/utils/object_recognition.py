import sys
import time
from typing import List, Type
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch
import numpy as np


class ObjectRecognition:
    def __init__(self) -> None:
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        # self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
            print("true true is cuda")
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Read the categories
        with open("../assets/imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]
        print("Initialized MobileNetV3 model")

    def classify(self, img: np.ndarray) -> torch.Tensor:
        t0 = time.time_ns()
        img = np.int_(img * 255).astype("uint8")
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
        with torch.no_grad():
            output = self.model(input_batch)
        # output[0] == Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        probs = torch.nn.functional.softmax(output[0], dim=0)
        print("[object_recognition]probs.shape: torch.Tensor=", probs.shape)
        probs = torch.Tensor.cpu(probs)
        print("[object_recognition]probs.shape: torch.Tensor<cpu>=", probs.shape)
        probs = np.array(probs)
        print("[object_recognition]probs.shape: np.array=", probs.shape)
        sys.exit(0)
        t1 = time.time_ns()
        # Show top categories per image
        #top5_prob, top5_catid = torch.topk(probs, 5)
        #for i in range(top5_prob.size(0)):
        #    print(self.categories[top5_catid[i]], top5_prob[i].item())
        #print("--------")
        return probs

    def serialize_img(self, img: np.ndarray):
        RGBImg = List[List[List[float]]]
        img_np = np.reshape(img, (3, 120, 160))
        
        with open("img.json", "w") as f:
            img_: RGBImg = list(img_np)
            img_[0] = list(img_[0])
            img_[1] = list(img_[1])
            img_[2] = list(img_[2])
            f.write('{\n"img":')
            f.write("[\n")
            for channel in img_:
                f.write("[\n")
                for row in channel:
                    f.write("[\n")
                    print(*row, sep=",", file=f)
                    f.write("],\n")
                f.write("],\n")
            f.write("]\n")
            f.write("}")
            