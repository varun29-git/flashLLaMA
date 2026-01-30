
import torch
try:
    import bitsandbytes as bnb
    print("SUCCESS: bitsandbytes imported.")
    try:
        opt = bnb.optim.AdamW8bit([torch.tensor([1.0], requires_grad=True)])
        print(f"SUCCESS: Created AdamW8bit optimizer. Class: {type(opt)}")
    except Exception as e:
        print(f"FAILED to instantiate AdamW8bit: {e}")
except ImportError:
    print("FAILED: bitsandbytes not found.")
except Exception as e:
    print(f"FAILED: imports bitsandbytes raised: {e}")
