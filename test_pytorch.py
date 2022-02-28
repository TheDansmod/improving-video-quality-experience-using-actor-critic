import torch
import torchvision.models as models
from nebullvm import optimize_torch_model

model = models.efficientnet_b0()
bs, input_sizes = 1, [(3, 256, 256)]
save_dir = "."
optimized_model = optimize_torch_model(model, batch_size=bs, input_sizes=input_sizes, save_dir=save_dir)
x = torch.randn((bs, *input_sizes[0]))
res = model(x)

