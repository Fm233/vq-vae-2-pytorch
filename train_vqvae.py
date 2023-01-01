import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

from tensorboardX import SummaryWriter


def train(epoch, loader, model, optimizer, scheduler, device, args, writer):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 8

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            args.steps += 1
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            writer.add_scalar('recon_loss', recon_loss.item(), args.steps)
            writer.add_scalar('latent_loss', latent_loss.item(), args.steps)
            writer.add_scalar('avg_mse', mse_sum / mse_n, args.steps)

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)
                    _, _, _, qt, qb = model.encode(sample)

                qt = qt.to(torch.float).unsqueeze(1) / 512
                qb = qb.to(torch.float).unsqueeze(1) / 512

                # print(f"[DEBUG] qt.shape = {qt.shape}")
                # print(f"[DEBUG] qb.shape = {qb.shape}")

                im = torch.cat([sample, out], 0).cpu()
                grid = make_grid(im, nrow=sample_size, range=(-1, 1), normalize=True)
                grid_qt = make_grid(qt, nrow=sample_size, range=(-1, 1), normalize=True)
                grid_qb = make_grid(qb, nrow=sample_size, range=(-1, 1), normalize=True)
                writer.add_image('sample', grid, args.steps)
                writer.add_image('quant_t', grid_qt, args.steps)
                writer.add_image('quant_b', grid_qb, args.steps)

                model.train()


def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        x_tilde, _ = model(images)
    return x_tilde


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    batch_size = 128 // args.n_gpu
    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=2
    )
    fixed_img, _ = next(iter(loader))
    print(f"[DEBUG] Shape is {fixed_img.shape}")

    model = VQVAE().to(device)
    summary(model, input_size=(batch_size, 3, args.size, args.size))

    writer = SummaryWriter('./{0}'.format(args.logdir))

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if (args.optimizer_checkpoint != ""):
        optimizer_checkpoint = torch.load(args.optimizer_checkpoint)
        optimizer.load_state_dict(optimizer_checkpoint)
        print("[DEBUG] Loaded optimizer checkpoint")

    if (args.checkpoint != ""):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        print("[DEBUG] Loaded checkpoint")

    if (args.args_checkpoint != ""):
        args = torch.load(args.args_checkpoint)
        print("[DEBUG] Loaded args checkpoint")

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    current_epoch = args.current_epoch
    model.train()

    for i in range(current_epoch - 1, args.epoch):
        args.current_epoch = i + 2
        train(i, loader, model, optimizer, scheduler, device, args, writer)

        if dist.is_primary():
            torch.save(model.state_dict(), f"pt/vqvae_{str(i + 1).zfill(3)}.pt")
            torch.save(optimizer.state_dict(), f"pt/optim_{str(i + 1).zfill(3)}.pt")
            torch.save(args, f"pt/args_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str, default="~/data/ffhq64/")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--optimizer-checkpoint", type=str, default="pt/optim_172.pt")
    parser.add_argument("--args-checkpoint", type=str, default="pt/args_172.pt")
    parser.add_argument("--checkpoint", type=str, default="pt/vqvae_172.pt")

    args = parser.parse_args()
    args.steps = 0

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
