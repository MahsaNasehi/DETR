import gc
import torch

# Delete variables
# del model, optimizer # or any large tensors/objects you used
gc.collect()

# Empty CUDA cache
torch.cuda.empty_cache()
torch.cuda.ipc_collect()