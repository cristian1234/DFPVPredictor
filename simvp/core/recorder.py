import os
import gc
import numpy as np
import torch
from typing import Optional

class Recorder:
    """
    Guarda checkpoints durante el entrenamiento:
      - save_best: cuando baja la val_loss (mejor modelo)
      - save_last: siempre al final de cada época (último modelo + alias latest.pth)
    Embebe: state_dict, config, optimizer, scheduler, epoch.
    Con limpieza agresiva de memoria post-save.
    """
    def __init__(self, verbose: bool = False, delta: float = 0.0, save_optim: bool = True, save_sched: bool = True):
        self.verbose = verbose
        self.best_score: Optional[float] = None
        self.val_loss_min = np.inf
        self.delta = float(delta)
        self.save_optim = save_optim
        self.save_sched = save_sched

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
        # ✅ forzar copia del modelo en CPU para evitar retención en MPS/GPU
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        blob = {'state_dict': state_dict_cpu}

        cfg = payload.get('config', None)
        if cfg is not None:
            blob['config'] = cfg

        if self.save_optim:
            opt = payload.get('optimizer', None)
            if opt is not None:
                blob['optimizer'] = opt.state_dict()

        if self.save_sched:
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
        payload_blob = self._pack(model, payload)
        torch.save(payload_blob, os.path.join(path, 'checkpoint_best.pth'))
        self.val_loss_min = float(val_loss)

        # 🧹 limpieza de memoria
        del payload_blob
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

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

        # 🧹 limpieza de memoria
        del payload_blob
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
