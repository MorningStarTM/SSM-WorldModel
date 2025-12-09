import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def one_hot_encode(actions, num_actions):
    """
    Convert an array of actions into one-hot encoding.
    
    Args:
    - actions (np.array): Array of integer actions.
    - num_actions (int): Total number of possible actions (size of action space).
    
    Returns:
    - np.array: One-hot encoded actions, shape (num_samples, num_actions).
    """
    actions = np.array(actions)  
    one_hot_actions = np.zeros((len(actions), num_actions), dtype=np.float32)
    one_hot_actions[np.arange(len(actions)), actions] = 1
    return np.array(one_hot_actions)


def load_npz_files_from_folder(folder_path, start=0, end=100):
    all_observations = []
    all_rewards = []
    all_actions = []
    all_dones = []
    all_next_observations = []

    folder = os.listdir(folder_path)

    for filename in folder[start:end]:
        if filename.endswith(".npz"):
            file_path = os.path.join(folder_path, filename)
            npz_data = np.load(file_path, allow_pickle=True)

            obs = np.asarray(npz_data['obs'], dtype=np.float32)
            obs_next = np.asarray(npz_data['obs_next'], dtype=np.float32)
            reward = np.asarray(npz_data['reward'], dtype=np.float32)
            action = np.asarray(npz_data['action'], dtype=np.int64)
            done = np.asarray(npz_data['done'], dtype=bool)

            all_observations.append(obs)
            all_rewards.append(reward)
            all_actions.append(action)
            all_dones.append(done)
            all_next_observations.append(obs_next)

    # Concatenate along first axis
    observations = np.concatenate(all_observations, axis=0).astype(np.float32)
    rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.int64)
    dones = np.concatenate(all_dones, axis=0).astype(bool)
    next_observations = np.concatenate(all_next_observations, axis=0).astype(np.float32)

    # Normalize rewards (optional)
    reward_min = rewards.min()
    reward_max = rewards.max()
    rewards = (rewards - reward_min) / (reward_max - reward_min + 1e-8)

    one_hot_actions = one_hot_encode(actions, 179).astype(np.float32)

    return observations, rewards, one_hot_actions, dones, next_observations




class GridSSMDataset(Dataset):
    """
    Dataset for one-step dynamics:
        x_t, u_t -> x_{t+1}

    Uses:
        observations       -> x_t
        one_hot_actions    -> u_t
        next_observations  -> x_{t+1}
    """
    def __init__(self, folder_path, start=0, end=100, device="cpu"):
        super().__init__()
        (
            observations,
            rewards,
            one_hot_actions,
            dones,
            next_observations,
        ) = load_npz_files_from_folder(folder_path, start=start, end=end)

        # Store as tensors
        self.x_t = torch.from_numpy(observations).float()          # (N, state_dim)
        self.u_t = torch.from_numpy(one_hot_actions).float()       # (N, action_dim)
        self.x_tp1 = torch.from_numpy(next_observations).float()   # (N, state_dim)

        # Optionally keep rewards/dones if needed later
        self.rewards = torch.from_numpy(rewards).float()           # (N,)
        self.dones = torch.from_numpy(dones).bool()                # (N,)

        self.device = device  # not strictly needed here, but kept for future

    def __len__(self):
        return self.x_t.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            x_t:    (state_dim,)
            u_t:    (action_dim,)
            x_tp1:  (state_dim,)
        """
        return self.x_t[idx], self.u_t[idx], self.x_tp1[idx]


def create_dataloaders(
    folder_path: str,
    start: int = 0,
    end: int = 100,
    batch_size: int = 128,
    val_split: float = 0.2,
    shuffle_dataset: bool = True,
    num_workers: int = 0,
):
    """
    Helper to create train & validation DataLoaders from NPZ folder.

    Args:
        folder_path:   path to folder with .npz files
        start, end:    slice of files to load (passed to load_npz_files_from_folder)
        batch_size:    batch size for loaders
        val_split:     fraction of dataset to use for validation (0–1)
        shuffle_dataset: whether to shuffle before splitting
        num_workers:   DataLoader workers

    Returns:
        train_loader, val_loader
    """
    full_dataset = GridSSMDataset(folder_path, start=start, end=end)

    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    if shuffle_dataset:
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
