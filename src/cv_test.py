import time
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch

#model = torch.hub.load(
#    "pytorch/vision:v0.12.0", "mobilenet_v2", weights="MobileNet_V2_Weights.DEFAULT"
#)
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
#model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model.eval()
print("devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    model.to("cuda")
    print("true true")
img = Image.open("../assets/kitchen_unsplash.jpg")

t0 = time.time_ns()
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(img)
print("input_tensor=", input_tensor)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")

with torch.no_grad():
    output = model(input_batch)
# output[0] == Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
probs = torch.nn.functional.softmax(output[0], dim=0)
t1 = time.time_ns()

# Read the categories
with open("../assets/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probs, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

print("time taken: ", (t1-t0) / 1000000, "ms")
