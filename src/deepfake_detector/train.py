import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import ModelConfig, PreprocessorConfig
from .evaluator import _collate_fn
from .models import HybridCNNViT
from .preprocessor import Preprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str = "outputs"
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-4
    optimizer: str = "adam"   # "adam" | "sgd"
    patience: int = 5
    val_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_cfg: TrainConfig, model_cfg: ModelConfig, pre_cfg: PreprocessorConfig):
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(train_cfg.output_dir, "train_log.csv")
    best_ckpt = os.path.join(train_cfg.output_dir, "best_model.pt")

    # Data
    preprocessor = Preprocessor(pre_cfg)
    full_dataset = preprocessor.load_dataset(train_cfg.dataset_root, split="train")

    val_size = int(len(full_dataset) * train_cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=_collate_fn)

    # Model
    model = HybridCNNViT(model_cfg).to(train_cfg.device)
    criterion = nn.CrossEntropyLoss()

    if train_cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg.learning_rate, momentum=0.9)

    best_val_acc = 0.0
    patience_counter = 0

    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_accuracy\n")

        for epoch in range(1, train_cfg.epochs + 1):
            # --- Train ---
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                if batch is None:
                    continue
                imgs, labels, _ = batch
                imgs, labels = imgs.to(train_cfg.device), labels.to(train_cfg.device)

                optimizer.zero_grad()
                logits, _, _ = model.forward_batch(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)

            # --- Validate ---
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    imgs, labels, _ = batch
                    imgs, labels = imgs.to(train_cfg.device), labels.to(train_cfg.device)
                    logits, _, _ = model.forward_batch(imgs)
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += len(labels)

            val_acc = correct / total if total > 0 else 0.0
            logger.info(f"Epoch {epoch}/{train_cfg.epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            log_file.write(f"{epoch},{avg_loss:.6f},{val_acc:.6f}\n")
            log_file.flush()

            # --- Checkpoint & early stopping ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model.save_checkpoint(best_ckpt)
                logger.info(f"  Saved best checkpoint (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return best_ckpt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dataset_type", default="ff++")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        patience=args.patience,
    )
    model_cfg = ModelConfig()
    pre_cfg = PreprocessorConfig(dataset_type=args.dataset_type)

    train(train_cfg, model_cfg, pre_cfg)
