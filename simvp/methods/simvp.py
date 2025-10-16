import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from timm.utils import AverageMeter

from simvp.models import SimVP_Model
from .base_method import Base_method
import gc

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.
    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, config):
        return SimVP_Model(**config).to(self.device)

    def _predict(self, batch_x):
        # Permite long pred (> pre_seq_length)
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, loss_mean, **kwargs):
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)

        train_pbar = tqdm(train_loader)
        for batch_x, batch_y in train_pbar:
            self.model_optim.zero_grad()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            pred_y = self._predict(batch_x)
            loss = self.criterion(pred_y, batch_y)
            loss.backward()
            self.model_optim.step()
            if not self.by_epoch:
                self.scheduler.step()

            num_updates += 1
            loss_mean += loss.item()
            losses_m.update(loss.item(), batch_x.size(0))
            runner.call_hook('after_train_iter')
            runner._iter += 1

            train_pbar.set_description(f'train loss: {loss.item():.4f}')

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        # 🧹 limpieza mínima por época
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return num_updates, loss_mean

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []

        # ⚠️ Si no hay datos → devolvemos valores vacíos
        if len(vali_loader) == 0:
            return np.array([]), np.array([]), 0.0

        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_val_iter')

            pred_y = self._predict(batch_x)
            loss = self.criterion(pred_y, batch_y)

            preds_lst.append(pred_y.detach().cpu().numpy())
            trues_lst.append(batch_y.detach().cpu().numpy())

            runner.call_hook('after_val_iter')

            # 👉 Cortamos tras 1000 muestras para no comer RAM infinita
            if (i + 1) * batch_x.shape[0] > 1000:
                break

            vali_pbar.set_description(f'vali loss: {loss.mean().item():.4f}')
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        # 🧹 liberar memoria intermedia
        del preds_lst, trues_lst, batch_x, batch_y, pred_y
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return preds, trues, total_loss

    def test_one_epoch(self, runner, test_loader, **kwargs):
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []

        if len(test_loader) == 0:
            return np.array([]), np.array([]), np.array([])

        test_pbar = tqdm(test_loader)
        for batch_x, batch_y in test_pbar:
            batch_x = batch_x.to(self.device)
            runner.call_hook('before_val_iter')
            pred_y = self._predict(batch_x)

            inputs_lst.append(batch_x.detach().cpu().numpy())
            trues_lst.append(batch_y.detach().cpu().numpy())
            preds_lst.append(pred_y.detach().cpu().numpy())

            runner.call_hook('after_val_iter')

        inputs = np.concatenate(inputs_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        preds = np.concatenate(preds_lst, axis=0)

        # 🧹 liberar buffers
        del inputs_lst, trues_lst, preds_lst, batch_x, batch_y, pred_y
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return inputs, trues, preds
