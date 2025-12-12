import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearSSM(nn.Module):
    """
    x_{t+1} = A x_t + B u_t

    x_t: (B, state_dim)
    u_t: (B, action_dim)
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # A: (state_dim x state_dim)
        self.A = nn.Parameter(0.01 * torch.randn(state_dim, state_dim))
        # B: (state_dim x action_dim)
        self.B = nn.Parameter(0.01 * torch.randn(state_dim, action_dim))

    def step(self, x_t, u_t):
        """
        One-step prediction.
        x_t: (B, state_dim)
        u_t: (B, action_dim)
        returns x_hat_{t+1}: (B, state_dim)
        """
        x_next = x_t @ self.A.T + u_t @ self.B.T
        return x_next

    def rollout(self, x0, u_seq):
        """
        Multi-step rollout in open-loop.

        x0:    (B, state_dim)
        u_seq: (B, H, action_dim)  # controls for H steps

        returns:
            xs: (B, H+1, state_dim)  # [x0, x1, ..., xH]
        """
        B, H, _ = u_seq.shape
        xs = [x0]
        x_t = x0
        for t in range(H):
            u_t = u_seq[:, t, :]
            x_t = self.step(x_t, u_t)
            xs.append(x_t)
        xs = torch.stack(xs, dim=1)
        return xs






class LinearSSMTrainer:
    def __init__(
        self,
        model: LinearSSM,
        train_loader,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        log_interval: int = 100,
    ):
        """
        Trainer for one-step supervised learning:
            x_{t+1} = A x_t + B u_t

        Args:
            model:       LinearSSM instance
            train_loader: DataLoader yielding (x_t, u_t, x_tp1)
            lr:          Learning rate for Adam
            device:      torch.device ('cuda' or 'cpu'). If None, auto-detect.
            log_interval: Print training loss every N batches
        """
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        self.log_interval = log_interval

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _step_batch(self, batch):
        """
        One training step on a batch.
        batch: (x_t, u_t, x_tp1)
        """
        x_t, u_t, x_tp1 = batch

        x_t = x_t.to(self.device)
        u_t = u_t.to(self.device)
        x_tp1 = x_tp1.to(self.device)

        x_pred = self.model.step(x_t, u_t)
        loss = F.mse_loss(x_pred, x_tp1)

        return loss

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self._step_batch(batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"[Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)}] "
                    f"Train Loss: {avg_loss:.6f}"
                )

        avg_epoch_loss = total_loss / max(1, num_batches)
        return avg_epoch_loss

    @torch.no_grad()
    def evaluate(self, val_loader: Optional[torch.utils.data.DataLoader] = None):
        if val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            loss = self._step_batch(batch)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def fit(self, num_epochs: int):
        """
        Train for num_epochs. Returns history dict with train/val losses.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        best_state_dict = None

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.evaluate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss is not None:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

        # Optionally restore best validation model
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Restored best model (Val Loss = {best_val_loss:.6f})")

        return history





class LatentLinearSSM(nn.Module):
    """
    JEPA-style latent-linear SSM:
        z_t = encoder(x_t)
        z_{t+1}_pred = A z_t + B u_t
        loss = MSE(z_{t+1}_pred, encoder(x_{t+1}))
    """
    def __init__(self, obs_dim, action_dim, latent_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # ----------------------
        # Encoder network E(x)
        # ----------------------
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        # ----------------------
        # Learnable A, B in latent space
        # ----------------------
        self.A = nn.Parameter(0.01 * torch.randn(latent_dim, latent_dim))
        self.B = nn.Parameter(0.01 * torch.randn(latent_dim, action_dim))

    def encode(self, x):
        """Encode raw observation into latent z."""
        return self.encoder(x)

    def step(self, x_t, u_t):
        """
        JEPA-style one-step prediction:
            z_t = encoder(x_t)
            z_pred_tp1 = A z_t + B u_t
        """
        z_pred_tp1 = x_t @ self.A.T + u_t @ self.B.T
        return z_pred_tp1

    def compute_loss(self, x_t, u_t, x_tp1):
        """
        JEPA latent prediction loss:
            z_{t+1}_pred vs z_{t+1}_true
        """
        z_tp1_pred = self.step(x_t, u_t)
        z_tp1_true = self.encode(x_tp1).detach()  # stop gradient like JEPA

        loss = F.mse_loss(z_tp1_pred, z_tp1_true)
        return loss



class LatentSSMTrainer:
    def __init__(self, model:LatentLinearSSM, train_loader, lr=1e-3, device=None):
        self.model = model
        self.train_loader = train_loader

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _batch_loss(self, batch):
        x_t, u_t, x_tp1 = batch
        x_t = x_t.to(self.device)
        u_t = u_t.to(self.device)
        x_tp1 = x_tp1.to(self.device)

        return self.model.compute_loss(x_t, u_t, x_tp1)

    def train_one_epoch(self):
        self.model.train()
        total = 0
        for x_t, u_t, x_tp1 in self.train_loader:
            loss = self._batch_loss((x_t, u_t, x_tp1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total += loss.item()
        return total / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader):
        if val_loader is None:
            return None

        self.model.eval()
        total = 0
        for x_t, u_t, x_tp1 in val_loader:
            loss = self._batch_loss((x_t, u_t, x_tp1))
            total += loss.item()
        return total / len(val_loader)

    def fit(self, epochs):
        for e in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss = self.evaluate()
            print(f"Epoch {e}: train={train_loss:.6f} | val={val_loss}")
