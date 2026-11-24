"""Training loop: AdamW, per-task model selection on validation, early stopping.

Model selection (`train.selection`):

* `per_task` (default): every epoch the validation set is scored on each task the
  model has (risk AUPRC, vitals MAE, onset macro-F1). A snapshot of the weights is
  kept at each task's best epoch, and at test time each head's output comes from
  its own snapshot (`PerTaskSelection`). Training stops when *no* task has improved
  by more than `train.min_delta` for `train.patience` epochs. This stops the noisy,
  early-peaking risk AUPRC from cutting off the vitals and onset heads.
* `risk_auprc`: the original protocol. One checkpoint, chosen and early-stopped
  on validation AUPRC alone (vitals MAE for forecast-only models).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import collate_windows
from ..evaluation.metrics import onset_metrics, risk_metrics, vitals_metrics
from .losses import MultiTaskLoss

# Per task: (validation metric name, sign so that higher is better)
TASK_MONITORS = {
    "risk": ("val_auprc", 1.0),
    "vitals": ("val_vitals_mae", -1.0),
    "onset": ("val_onset_macro_f1", 1.0),
}


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


def task_scores(val_pred: dict) -> dict:
    """Validation metric for every task present in the predictions."""
    s = {}
    if "pred_risk" in val_pred:
        s["val_auprc"] = risk_metrics(val_pred["y_risk"], val_pred["pred_risk"])["auprc"]
    if "pred_vitals" in val_pred:
        s["val_vitals_mae"] = vitals_metrics(val_pred["y_vitals"], val_pred["pred_vitals"],
                                             val_pred["y_vitals_mask"])["mae"]
    if "pred_onset" in val_pred:
        s["val_onset_macro_f1"] = onset_metrics(val_pred["y_onset"], val_pred["pred_onset"])["macro_f1"]
    return s


class PerTaskSelection(nn.Module):
    """Each task's output comes from the snapshot taken at that task's best epoch.

    Snapshots that share an epoch are run once, so a forward pass costs at most
    one model evaluation per distinct epoch.
    """

    def __init__(self, base: nn.Module, states: dict, task_epoch: dict):
        super().__init__()
        self.task_epoch = dict(task_epoch)
        self.epochs = sorted(set(task_epoch.values()))
        self.models = nn.ModuleList()
        for e in self.epochs:
            m = copy.deepcopy(base)
            m.load_state_dict(states[e])
            self.models.append(m)

    def forward(self, batch):
        outs = {e: m(batch) for e, m in zip(self.epochs, self.models)}
        return {task: outs[e][task] for task, e in self.task_epoch.items() if task in outs[e]}


class Trainer:
    def __init__(self, model, cfg: dict, out_dir: str | Path, device: str = "cpu"):
        self.model = model.to(device)
        self.selected = self.model  # what to evaluate after fit()
        self.cfg, self.device = cfg, device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t = cfg["train"]
        self.loss_fn = MultiTaskLoss(**t["loss"]).to(device)
        self.trainable = getattr(model, "trainable", True) and any(p.requires_grad for p in model.parameters())
        if self.trainable:
            self.opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    def load_selected(self, task_epoch: dict):
        """Rebuild the selected model from saved best_<task>.pt files (for --eval-only)."""
        states = {}
        for k, e in task_epoch.items():
            states[e] = torch.load(self.out_dir / f"best_{k}.pt", map_location="cpu")
        if len(states) == 1:
            self.model.load_state_dict(next(iter(states.values())))
            self.selected = self.model
        else:
            self.selected = PerTaskSelection(self.model, states, task_epoch).to(self.device)

    def _train_epoch(self, train_ds, epoch):
        t = self.cfg["train"]
        train_ds.resample()
        loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                            collate_fn=collate_windows, num_workers=t.get("num_workers", 0))
        self.model.train()
        losses, part_sums = [], {}
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
        return {"train_loss": float(np.mean(losses)),
                **{f"train_loss_{k}": v / len(losses) for k, v in part_sums.items()}}

    def fit(self, train_ds, val_ds) -> dict:
        t = self.cfg["train"]
        ckpt = self.out_dir / "best.pt"
        if not self.trainable:
            torch.save(self.model.state_dict(), ckpt)
            return {"selection": None, "monitor": None, "best_val_monitor": None, "best_epoch": 0,
                    "best_epoch_per_task": {}, "best_val_per_task": {}, "epochs_run": 0, "history": []}

        selection = t.get("selection", "per_task")
        min_delta = float(t.get("min_delta", 0.0))
        tasks = [k for k in TASK_MONITORS if k in t["tasks"]]
        if selection == "risk_auprc":  # original protocol: one monitor for every head
            tasks = ["risk"] if "risk" in tasks else ["vitals"]
        elif selection != "per_task":
            raise ValueError(f"unknown train.selection: {selection}")

        best = {k: -np.inf for k in tasks}
        best_epoch = {k: 0 for k in tasks}
        states, history, bad = {}, [], 0
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            row = {"epoch": epoch, **self._train_epoch(train_ds, epoch)}
            row["train_seconds"] = time.time() - t0
            val_pred = predict(self.model, val_ds, t["eval_batch_size"], self.device)
            scores = task_scores(val_pred)
            row.update(scores)
            row["seconds"] = time.time() - t0
            history.append(row)

            improved = []
            for k in tasks:
                name, sign = TASK_MONITORS[k]
                s = sign * scores[name]
                if s > best[k] + min_delta or epoch == 1:
                    best[k], best_epoch[k] = s, epoch
                    improved.append(k)
            if improved:
                states[epoch] = {n: v.detach().cpu().clone() for n, v in self.model.state_dict().items()}
                for e in [e for e in states if e not in best_epoch.values()]:
                    del states[e]  # snapshot no longer best for any task
                bad = 0
            else:
                bad += 1
            msg = " ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"epoch {epoch}: loss={row['train_loss']:.4f} {msg} "
                  f"best[{', '.join(improved) or '-'}] ({row['seconds']:.0f}s)", flush=True)
            if bad >= t["patience"]:
                print(f"early stopping: no task improved for {bad} epochs", flush=True)
                break

        all_tasks = [k for k in TASK_MONITORS if k in t["tasks"]]
        if selection == "risk_auprc":
            task_epoch = {k: best_epoch[tasks[0]] for k in all_tasks}
        else:
            task_epoch = {k: best_epoch[k] for k in all_tasks}
        for k, e in task_epoch.items():
            torch.save(states[e], self.out_dir / f"best_{k}.pt")
        main = "risk" if "risk" in tasks else tasks[0]
        torch.save(states[best_epoch[main]], ckpt)  # single checkpoint of the main task
        if len(set(task_epoch.values())) == 1:
            self.model.load_state_dict(states[next(iter(task_epoch.values()))])
            self.selected = self.model
        else:
            self.selected = PerTaskSelection(self.model, states, task_epoch).to(self.device)

        name, sign = TASK_MONITORS[main]
        return {"selection": selection, "monitor": name, "best_val_monitor": sign * best[main],
                "best_epoch": best_epoch[main],
                "best_epoch_per_task": task_epoch,
                "best_val_per_task": {k: TASK_MONITORS[k][1] * best[k] for k in tasks},
                "epochs_run": len(history), "history": history}
