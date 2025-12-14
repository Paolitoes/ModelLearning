import torch
import string
import unicodedata

# Preparing Torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cude')

torch.set_default_device(device)
print(f"Using Device = {torch.get_default_device()}")


# Preparing The Data

allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in allowed_characters
            )


# Turning Names into Tensors

def letterToIndex(letter):
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
