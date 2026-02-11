from imports import torch, nn, F, device

class GAUSS_RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sharp_sig_temp=0.7, dropout_p=0.3, weight_decay=0.0, sigma=1.0):
        super(GAUSS_RBM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_visible = visible_dim
        self.num_hidden = hidden_dim
        self.sharp_sig_temp = sharp_sig_temp
        self.weight_decay = weight_decay

        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim, dtype=torch.float32) * 0.01)
        self.v_bias = nn.Parameter(torch.randn(visible_dim, dtype=torch.float32) * 1e-2)
        self.h_bias = nn.Parameter(torch.randn(hidden_dim, dtype=torch.float32) * 1e-2)

        self.sigma = torch.ones(self.num_visible, dtype=torch.float32, requires_grad=False)

        self.dropout = nn.Dropout(p=dropout_p)

        #self.to(self.device)

    def initialize(self, W, v_bias, h_bias):
        self.W = W.to(device)
        self.v_bias = v_bias.to(device)
        self.h_bias = h_bias.to(device)

    def reinitialize(self, vis_dim, hid_dim, sharp_sig_temp=0.7, dropout_p=0.3, weight_decay=0.0, sigma=1.0):
        self.num_visible = int(vis_dim)
        self.num_hidden = int(hid_dim)
        self.sharp_sig_temp = sharp_sig_temp
        self.weight_decay = weight_decay

        self.W = nn.Parameter((torch.randn(self.num_visible, self.num_hidden, dtype=torch.float32) * 0.01).to(device))
        self.v_bias = nn.Parameter((torch.randn(self.num_visible, dtype=torch.float32) * 1e-2).to(device))
        self.h_bias = nn.Parameter((torch.randn(self.num_hidden, dtype=torch.float32) * 1e-2).to(device))
        self.sigma = torch.ones(self.num_visible, dtype=torch.float32, requires_grad=False).to(device)

        self.dropout = nn.Dropout(p=dropout_p)

    def sample_hidden(self, v, eps=1e-6):
        v = v.to(device)
        sigma2 = ((self.sigma ** 2) + eps).to(device)

        pre_act = torch.matmul(v / sigma2, self.W) + self.h_bias
        p_h_given_v = torch.sigmoid(pre_act)
        #p_h_given_v = self.dropout(p_h_given_v)
        p_h_given_v = torch.clamp(p_h_given_v, eps, 1.0 - eps)

        h_sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, h_sample

    def sample_visible(self, h):
        h = h.to(device)
        v_mean = torch.matmul(h, self.W.T) + self.v_bias

        noise = torch.randn_like(v_mean).to(device) * self.sigma
        v_sample = v_mean + noise

        v_mean = torch.nan_to_num(v_mean, nan=0.0, posinf=1e6, neginf=-1e6)
        v_sample = torch.nan_to_num(v_sample, nan=0.0, posinf=1e6, neginf=-1e6)

        return v_mean, v_sample

    def gibbs_step(self, v):
        _, h = self.sample_hidden(v)
        v_new, _ = self.sample_visible(h)
        return v_new.detach()

    def energy(self, v):
        v = v.to(self.device)
        sigma2 = self.sigma ** 2

        term_vis = 0.5 * (((v - self.v_bias) ** 2) / sigma2).sum(dim=1)
        #W_scaled = self.W / sigma2.unsqueeze(1)
        activ = torch.matmul(v / sigma2, self.W) + self.h_bias
        term_hid = F.softplus(activ).sum(dim=1)

        Fv = term_vis - term_hid
        return Fv

    def compute_energy_gap(self, x):
        _, h_sample = self.sample_hidden(x)
        v_neg, _ = self.sample_visible(h_sample)
        e_pos = self.energy(x)
        e_neg = self.energy(v_neg)
        return e_pos.mean().item(), e_neg.mean().item(), (e_neg.mean() - e_pos.mean()).item()

    def forward(self, v):
        return self.sample_hidden(v)

def _seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------------
# Sparsity helper
# -------------------------------
def apply_sparsity(arr: np.ndarray, sparsity: float):
    """
    sparsity: float in [0, 1], fraction of zeros to enforce.
    """
    if sparsity <= 0.0:
        return arr
    mask = np.random.rand(*arr.shape) < sparsity
    arr[mask] = 0
    return arr

# -------------------------------
# 1. Independent Poisson counts
# -------------------------------
def poisson_independent(n_samples: int, n_features: int,
                        lam: float = 1.0, sparsity: float = 0.0, seed: int = 0):
    _seed_everything(seed)
    samples = np.random.poisson(lam=lam, size=(n_samples, n_features)).astype(np.float32)
    samples = apply_sparsity(samples, sparsity)
    return torch.from_numpy(samples)

# -------------------------------
# 2. Factor model with correlated counts
# -------------------------------
def factor_model_counts(n_samples: int, n_features: int, n_factors: int = 5,
                        factor_strength: float = 1.0, noise: float = 0.5,
                        sparsity: float = 0.0, seed: int = 0):
    _seed_everything(seed)

    F = np.random.normal(size=(n_samples, n_factors)).astype(np.float32)
    W = np.random.normal(scale=factor_strength, size=(n_factors, n_features)).astype(np.float32)
    bias = np.random.normal(loc=0.0, scale=0.5, size=(n_features,)).astype(np.float32)

    logits = F @ W + bias
    logits += np.random.normal(scale=noise, size=logits.shape).astype(np.float32)

    rates = np.exp(logits)
    samples = np.random.poisson(lam=rates).astype(np.float32)

    samples = apply_sparsity(samples, sparsity)
    return torch.from_numpy(samples)

# -------------------------------
# 3. Mixture of count modes
# -------------------------------
def mixture_of_count_modes(n_samples: int, n_features: int, n_modes: int = 4,
                           base_rate: float = 1.0, rate_variation: float = 0.5,
                           sparsity: float = 0.0, seed: int = 0):
    _seed_everything(seed)

    samples_per_mode = [n_samples // n_modes] * n_modes
    for i in range(n_samples % n_modes):
        samples_per_mode[i] += 1

    mode_rates = np.random.normal(loc=base_rate,
                                  scale=rate_variation,
                                  size=(n_modes, n_features))
    mode_rates = np.clip(mode_rates, 0.1, None)

    xs = []
    assignments = []
    for mode_idx, count in enumerate(samples_per_mode):
        samp = np.random.poisson(lam=mode_rates[mode_idx],
                                 size=(count, n_features)).astype(np.float32)
        xs.append(samp)
        assignments.extend([mode_idx]*count)

    data = np.vstack(xs)
    data = apply_sparsity(data, sparsity)

    perm = np.random.permutation(n_samples)
    data = data[perm]
    assignments = np.array(assignments)[perm]

    return torch.from_numpy(data), assignments

# -------------------------------
# Generic wrapper
# -------------------------------
def make_dataset_counts(n_samples=10000, n_features=64, kind='factor',
                        sparsity=0.0, seed=0, **kwargs):
    if kind == 'independent':
        return poisson_independent(
            n_samples, n_features,
            lam=kwargs.get('lam', 1.0),
            sparsity=sparsity,
            seed=seed
        )

    elif kind == 'factor':
        return factor_model_counts(
            n_samples, n_features,
            n_factors=kwargs.get('n_factors', 5),
            factor_strength=kwargs.get('factor_strength', 1.0),
            noise=kwargs.get('noise', 0.5),
            sparsity=sparsity,
            seed=seed
        )

    elif kind == 'mixture':
        data, _ = mixture_of_count_modes(
            n_samples, n_features,
            n_modes=kwargs.get('n_modes', 4),
            base_rate=kwargs.get('base_rate', 1.0),
            rate_variation=kwargs.get('rate_variation', 0.5),
            sparsity=sparsity,
            seed=seed
        )
        return data

    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

def get_dataloader(data: torch.Tensor, batch_size: int = 64, shuffle: bool = True, drop_last: bool = True):
    """
    Wrap a tensor into a DataLoader. Expects data as torch.Tensor (N, D).
    """
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_test_split(data: torch.Tensor, test_frac: float = 0.1, seed: int = 0):
    _seed_everything(seed)
    n = data.size(0)
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_test
    train_set, test_set = random_split(data, [n_train, n_test])
    # random_split returns Subset objects with TensorDataset compatibility; convert to plain tensors
    train_tensor = torch.stack([train_set[i] for i in range(len(train_set))]).squeeze()
    test_tensor = torch.stack([test_set[i] for i in range(len(test_set))]).squeeze()
    return train_tensor, test_tensor


def save_dataset(data: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)


