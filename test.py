import torch
print(torch.cuda.is_available())
device = torch.device('cuda')
print(f"using device: {device}")
print(type(device))