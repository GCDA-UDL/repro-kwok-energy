"""Held-out real workload: ResNet-50 training (PyTorch only, no torchvision).

ResNet-50 (He et al., bottleneck blocks [3, 4, 6, 3]) is defined here so the
workload depends only on torch and runs identically on every bench host. Data
loader workers synthesise ImageNet-size uint8 images and apply the usual CPU
augmentation (random resized crop, horizontal flip, normalisation); the GPU runs a
mixed-precision SGD step. No dataset download is needed, yet both the CPU data path
and the GPU kernels look like ordinary image-classification training.

Prints ``READY`` after the warm-up iterations, then trains until terminated.
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin: int, width: int, stride: int) -> None:
        super().__init__()
        cout = width * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        return F.relu(self.bn3(self.conv3(out)) + idt, inplace=True)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 64
        for width, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
            for i in range(blocks):
                layers.append(Bottleneck(cin, width, stride if i == 0 else 1))
                cin = width * Bottleneck.expansion
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class SyntheticImages(IterableDataset):
    """Endless stream of augmented 224x224 crops from random 256x256 uint8 images."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def __iter__(self):
        wi = get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0))
        g = torch.Generator().manual_seed(self.seed + (wi.id if wi else 0))
        while True:
            img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, generator=g)
            # random resized crop (scale 0.08-1, ratio 3/4-4/3), as in standard training
            area = 256 * 256 * rng.uniform(0.08, 1.0)
            ratio = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            w = min(256, max(8, int(round(math.sqrt(area * ratio)))))
            h = min(256, max(8, int(round(math.sqrt(area / ratio)))))
            top, left = rng.randint(0, 256 - h), rng.randint(0, 256 - w)
            crop = img[:, top:top + h, left:left + w].float().div_(255).unsqueeze(0)
            x = F.interpolate(crop, size=(224, 224), mode="bilinear", align_corners=False)[0]
            if rng.random() < 0.5:
                x = torch.flip(x, dims=(2,))
            yield (x - MEAN) / STD, rng.randrange(1000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = until terminated")
    args = ap.parse_args()

    torch.set_num_threads(1)  # loader workers do the CPU work, as in a normal job
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    model = ResNet50().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(SyntheticImages(), batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    t0 = time.time()
    t_ready = t0
    model.train()
    for it, (x, y) in enumerate(loader, 1):
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if it == args.warmup:
            torch.cuda.synchronize(dev)
            print("READY", flush=True)
            t_ready = time.time()
        if it > args.warmup and it % 50 == 0:
            el = time.time() - t_ready
            print(f"iter {it} imgs/s {(it - args.warmup) * args.batch / el:.1f}",
                  file=sys.stderr, flush=True)
        if stop["flag"] or (args.max_seconds and time.time() - t0 > args.max_seconds):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
