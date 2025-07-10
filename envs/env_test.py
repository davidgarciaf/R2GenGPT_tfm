print("Importing libraries...")
import torch, transformers, peft, lightning
print("Done\n")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("lightning:", lightning.__version__)

print("\n GPU checks:\n")
try: 
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA CC :", torch.cuda.get_device_capability(0))
except:
    print("None GPU to check")
