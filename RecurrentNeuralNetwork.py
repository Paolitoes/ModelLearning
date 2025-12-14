import torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cude')

torch.set_default_device(device)
print(f"Using Device = {torch.get_default_device()}")
