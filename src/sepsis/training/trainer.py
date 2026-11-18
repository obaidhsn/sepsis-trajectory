"""Training loop: AdamW, early stopping on validation AUPRC (or vitals MAE
for forecast-only models), best checkpoint kept."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import collate_windows
from ..evaluation.metrics import risk_metrics, vitals_metrics
from .losses import MultiTaskLoss


def _to(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict(model, dataset, batch_size=512, device="cpu", num_workers=0) -> dict:
    """Run the model over a PatientSequenceDataset and gather numpy arrays."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_windows, num_workers=num_workers)
    keep = ("y_vitals", "y_vitals_mask", "y_risk", "y_onset", "hour", "patient")
    store = {k: [] for k in keep}
    preds = {}
    for batch in loader:
        batch = _to(batch, device)
        out = model(batch)
        for k in keep:
            store[k].append(batch[k].cpu())
        for k, v in out.items():
            preds.setdefault(k, []).append(v.float().cpu())
    res = {k: torch.cat(v).numpy() for k, v in store.items()}
    for k, v in preds.items():
        v = torch.cat(v)
        if k == "risk":
            v = torch.sigmoid(v)
        elif k == "onset":
            v = torch.softmax(v, -1)
        res[f"pred_{k}"] = v.numpy()
    return res


class Trainer:
    def __init__(self, model, cfg: dict, out_dir: str | Path, device: str = "cpu"):
        self.model = model.to(device)
        self.cfg, self.device = cfg, device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t = cfg["train"]
        self.loss_fn = MultiTaskLoss(**t["loss"]).to(device)
        self.trainable = getattr(model, "trainable", True) and any(p.requires_grad for p in model.parameters())
        if self.trainable:
            self.opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    def _monitor(self, val_pred) -> float:
        """Higher is better."""
        if "pred_risk" in val_pred:
            return risk_metrics(val_pred["y_risk"], val_pred["pred_risk"])["auprc"]
        return -vitals_metrics(val_pred["y_vitals"], val_pred["pred_vitals"], val_pred["y_vitals_mask"])["mae"]

    def fit(self, train_ds, val_ds) -> dict:
        t = self.cfg["train"]
        ckpt = self.out_dir / "best.pt"
        if not self.trainable:
            torch.save(self.model.state_dict(), ckpt)
            return {"monitor": None, "best_val_monitor": None, "best_epoch": 0, "epochs_run": 0, "history": []}
        best, best_epoch, bad, history = -np.inf, 0, 0, []
        monitor = "val_auprc" if "risk" in t["tasks"] else "neg_val_vitals_mae"
        for epoch in range(1, t["epochs"] + 1):
            train_ds.resample()
            loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                                collate_fn=collate_windows, num_workers=t.get("num_workers", 0))
            self.model.train()
            losses, part_sums, t0 = [], {}, time.time()
            for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False, disable=not sys.stderr.isatty()):
                batch = _to(batch, self.device)
                loss, parts = self.loss_fn(self.model(batch), batch)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), t["grad_clip"])
                self.opt.step()
                losses.append(float(loss.detach()))
                for k, v in parts.items():
                    part_sums[k] = part_sums.get(k, 0.0) + v
            train_seconds = time.time() - t0
            val_pred = predict(self.model, val_ds, t["eval_batch_size"], self.device)
            score = self._monitor(val_pred)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            **{f"train_loss_{k}": v / len(losses) for k, v in part_sums.items()},
                            monitor: score, "train_seconds": train_seconds,
                            "seconds": time.time() - t0})
            print(f"epoch {epoch}: loss={np.mean(losses):.4f} {monitor}={score:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if score > best:
                best, best_epoch, bad = score, epoch, 0
                torch.save(self.model.state_dict(), ckpt)
            else:
                bad += 1
                if bad >= t["patience"]:
                    print("early stopping", flush=True)
                    break
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        return {"monitor": monitor, "best_val_monitor": best, "best_epoch": best_epoch,
                "epochs_run": len(history), "history": history}
