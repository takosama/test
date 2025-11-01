from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from .models import SimpleCNN
from .eval import evaluate as eval_core
from .config import TrainConfig
from .data import create_dataloaders


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train SimpleCNN on MNIST")
    p.add_argument("--data-dir", type=str, default=r"c:\Users\takos\src\test\data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--target-acc", type=float, default=0.90)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deterministic", action="store_true", help="Enable deterministic CUDA")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Select device")
    p.add_argument("--gpu", type=int, default=0, help="GPU index to use when CUDA is selected")
    p.add_argument("--seed", type=int, default=42, help="Random seed (None to disable)")
    return p


def _select_device(args: argparse.Namespace) -> torch.device:
    """Select a device, ensuring CUDA devices include an index."""
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # CUDA path: validate index and always return cuda:<idx>
    count = torch.cuda.device_count()
    idx = args.gpu if 0 <= args.gpu < count else 0
    if idx != args.gpu:
        print(f"[WARN] GPU {args.gpu} not found. Using GPU {idx}.")
    return torch.device(f"cuda:{idx}")


def main() -> None:
    args = build_argparser().parse_args()
    device = _select_device(args)
    is_cuda = device.type == "cuda"
    non_blocking = is_cuda

    # Device info
    if is_cuda:
        # set_device expects an integer index or torch.device with index
        torch.cuda.set_device(device.index)
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Using device: cuda:{idx} - {torch.cuda.get_device_name(idx)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("Using device: CPU")

    if is_cuda and args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Deterministic cuDNN enabled")

    # Seeding
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        if is_cuda:
            torch.cuda.manual_seed_all(int(args.seed))

    cfg = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        target_acc=args.target_acc,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        device=args.device,
        gpu=args.gpu,
        seed=args.seed,
    )

    # Data
    print(f"Loading MNIST from: {args.data_dir}")
    train_loader, test_loader = create_dataloaders(cfg, device)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Model/optim
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    use_amp = is_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Train
    out_dir = Path(__file__).resolve().parent.parent
    best_acc = 0.0
    best_path = out_dir / "mnist_cnn_best.pt"
    last_path = out_dir / "mnist_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data = data.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            running_loss += float(loss.detach())

        train_loss = running_loss / max(1, len(train_loader))
        test_acc = eval_core(model, test_loader, device=device, is_cuda=is_cuda, non_blocking=non_blocking)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f}, test_acc={test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        if test_acc >= args.target_acc:
            print("Target accuracy reached. Stopping early.")
            break

    torch.save(model.state_dict(), last_path)
    print(f"Saved last to {last_path}, best to {best_path} (best_acc={best_acc*100:.2f}%)")
