import torch
import torchvision.transforms as T
from torchvision.utils import make_grid

from gan import CycleGANConfig, Generator
from utils.checkpoints import load_checkpoint
from PIL import Image

config1 = CycleGANConfig(
    "none",
    "HazeGan",
    "v1",
    image_shape=(3, 64, 64),
    latent_dim=64,
    dropout=0,
    num_epochs=1, batch_size=8,
    lr=2e-4,
    betas=(0.5, 0.999),
    lambdas=(10, 0.5),
    residuals=5,
    blocks=(64, 128, 256, 512),
    writer=False,
)

generatorA = Generator(
    config1.image_shape[0],
    config1.latent_dim,
    config1.residuals,
    p=config1.dropout,
    coder_len=config1.coder_len,
).to(config1.device)
generatorB = Generator(
    config1.image_shape[0],
    config1.latent_dim,
    config1.residuals,
    p=config1.dropout,
    coder_len=config1.coder_len,
).to(config1.device)

file_path = "models/hazemaze/v1/checkpoint-2023-09-21 13_56_12.898988.pt"
others = load_checkpoint(
    file_path,
    {"generatorA": generatorA, "generatorB": generatorB},
)
step = others["step"]

image_path = "demo/img_2.png"
if __name__ == '__main__':
    hazy = Image.open(image_path)
    hazy = hazy.resize((hazy.size[0] // 4 * 4, hazy.size[1] // 4 * 4))
    hazy_arr = generatorA(T.Normalize(config1.mean, config1.std)(T.ToTensor()(hazy).unsqueeze(0)))
    dehazed_arr = generatorB(hazy_arr.to(config1.device)).cpu()
    grid_arr = make_grid(torch.cat([hazy_arr, dehazed_arr]), nrow=2, normalize=True)
    grid = T.ToPILImage()(grid_arr)
    grid.show()
