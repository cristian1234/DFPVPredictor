# simvp/core/recorder.py
import os
import numpy as np
import torch
from typing import Optional

class Recorder:
    """
    Guarda checkpoints durante el entrenamiento:
      - save_best: cuando baja la val_loss (mejor modelo)
      - save_last: siempre al final de cada época (último modelo + alias latest.pth)
    Embebe: state_dict, config, optimizer, scheduler, epoch.
    """
    def __init__(self, verbose: bool = False, delta: float = 0.0):
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.val_loss_min = np.inf   # usar np.inf (no np.Inf)
        self.delta = float(delta)

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str, **payload):
        """
        Llamar después de calcular val_loss para guardar el BEST si mejora.
        payload puede incluir: config, optimizer, scheduler, epoch.
        """
        score = -float(val_loss)
        if self.best_score is None:
            self.best_score = score
            self._save_best(val_loss, model, path, **payload)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self._save_best(val_loss, model, path, **payload)

    # ---------------- private helpers ----------------

    def _pack(self, model: torch.nn.Module, payload: dict) -> dict:
        blob = {'state_dict': model.state_dict()}
        cfg = payload.get('config', None)
        if cfg is not None:
            blob['config'] = cfg
        opt = payload.get('optimizer', None)
        if opt is not None:
            blob['optimizer'] = opt.state_dict()
        sch = payload.get('scheduler', None)
        if sch is not None:
            blob['scheduler'] = sch.state_dict()
        ep = payload.get('epoch', None)
        if ep is not None:
            blob['epoch'] = int(ep)
        return blob

    def _save_best(self, val_loss: float, model: torch.nn.Module, path: str, **payload):
        os.makedirs(path, exist_ok=True)
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving BEST...')
        torch.save(self._pack(model, payload), os.path.join(path, 'checkpoint_best.pth'))
        self.val_loss_min = float(val_loss)

    # ---------------- public API ----------------

    def save_last(self, model: torch.nn.Module, path: str, **payload):
        """
        Guardar el último checkpoint de la época y además un alias latest.pth.
        Llamar SIEMPRE al final de cada época (aunque no mejore).
        """
        os.makedirs(path, exist_ok=True)
        payload_blob = self._pack(model, payload)
        # último de la época
        torch.save(payload_blob, os.path.join(path, 'checkpoint_last.pth'))
        # alias conveniente para auto-resume
        torch.save(payload_blob, os.path.join(path, 'latest.pth'))
