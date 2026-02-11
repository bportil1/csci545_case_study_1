import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import os
import math
from geomloss import SamplesLoss
from optimizers import FireflyOptimizer, contrastive_divergence_training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GAUSS_EBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sharp_sig_temp=0.7, dropout_p=0.3, weight_decay=0.0, sigma=1.0):
        super(GAUSS_EBM, self).__init__()

        self.device = device

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

class EBMVisualizer:
    def __init__(self, data_points, energies):
        """
        data_points: array-like of shape (N, 2) for 2D visualization
        energies: array-like of shape (N,) corresponding energy scores from EBM
        """
        self.data_points = data_points
        self.energies = energies
        self.threshold = np.percentile(energies, 90)  # default 90th percentile
        self.app = Dash(__name__)
        self.layout_app()

    def layout_app(self):
        self.app.layout = html.Div([
            html.H2("Interactive Outlier Detector", style={'textAlign': 'center'}),
            html.Div([
                dcc.Graph(id='scatter-plot'),
                html.Label("Energy Threshold:"),
                dcc.Slider(
                    id='threshold-slider',
                    min=float(np.min(self.energies)),
                    max=float(np.max(self.energies)),
                    step=0.01,
                    value=float(self.threshold),
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '70%', 'margin': 'auto'})
        ])

        @self.app.callback(
            Output('scatter-plot', 'figure'),
            Input('threshold-slider', 'value')
        )
        def update_plot(threshold):
            x = self.data_points[:, 0]
            y = self.data_points[:, 1]
            colors = ['red' if e > threshold else 'blue' for e in self.energies]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=colors, size=10),
                text=[f"Energy: {e:.2f}" for e in self.energies],
                hoverinfo='text'
            ))
            fig.update_layout(
                title=f'Outliers highlighted above threshold {threshold:.2f}',
                xaxis_title='X',
                yaxis_title='Y',
                showlegend=False
            )
            return fig

    def run(self):
        self.app.run(debug=True)

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
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_test_split(data: torch.Tensor, test_frac: float = 0.1, seed: int = 0):
    _seed_everything(seed)
    n = data.size(0)
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_test
    train_set, test_set = random_split(data, [n_train, n_test])
    train_tensor = torch.stack([train_set[i] for i in range(len(train_set))]).squeeze()
    test_tensor = torch.stack([test_set[i] for i in range(len(test_set))]).squeeze()
    return train_tensor, test_tensor

def save_dataset(data: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

def total_correlation(v, encoder, dataset_size=None):
    h_logits, h = encoder.sample_hidden(v)
    B, H = h.shape
    N = dataset_size if dataset_size else B
    logN = math.log(N)
    log_q_h_given_x = (
            h.unsqueeze(1) * (-F.softplus(-h).unsqueeze(0)) +
            (1 - h.unsqueeze(1)) * (-F.softplus(h).unsqueeze(0))
    ).sum(dim=2)
    log_q_h = torch.logsumexp(log_q_h_given_x, dim=1) - logN
    log_q_hj = (
            h.unsqueeze(1) * (-F.softplus(-h).unsqueeze(0)) +
            (1 - h.unsqueeze(1)) * (-F.softplus(h).unsqueeze(0))
    )
    log_q_hj = torch.logsumexp(log_q_hj, dim=1) - logN
    tc = (log_q_h - log_q_hj.sum(dim=1)).mean()
    return tc

def em_loss(decoding, data):
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    em = sinkhorn(decoding, data)
    return em.detach().cpu().numpy()

def rbm_evaluation(encoder, x, compute_lam=False, eps=1e-8):
    v_recon, h_sample = encoder.forward(x)
    v_recon = v_recon.to(device)
    if compute_lam:
        lam = encoder.sample_lambda(x, h_sample)
        h_recon, _ = encoder.sample_visible(h_sample, lam)
    else:
        h_recon, _ = encoder.sample_visible(v_recon)
    h_recon = h_recon.to(device)
    l1 = nn.L1Loss(reduction='mean')
    l1_loss = np.log1p(l1(h_recon, x).item() + eps)
    recon_loss = np.mean(np.log1p(((x-h_recon)**2).detach().cpu().numpy() + eps))
    tc = np.log1p(total_correlation(x, encoder).detach().cpu() + eps)
    em = np.log1p(em_loss(h_recon, x) + eps)
    energy_diff = abs(np.mean(encoder.compute_energy_gap(x)[2] + eps))
    if np.isnan(energy_diff):
        energy_diff = 8e8
    return (tc + l1_loss + recon_loss + energy_diff), [recon_loss, tc, em, l1_loss, energy_diff]

if __name__ == "__main__":
    num_features = 2
    data = make_dataset_counts(n_samples=50, n_features=num_features, kind='mixture', sparsity=.4, seed=42, n_modes=4, base_rate=2.0, rate_variation=0.7)
    ebm = GAUSS_EBM(num_features, 1)
    ff_opt = FireflyOptimizer(rbm_evaluation, ebm, data, 'gauss', 20, 20)
    hyperparams = ff_opt.optimize()
    ebm.reinitialize(num_features, 1, hyperparams[2], hyperparams[3], hyperparams[4], hyperparams[5])
    dataloader = get_dataloader(data, int(hyperparams[6]))
    ebm, logs = contrastive_divergence_training(dataloader, ebm)
    energies = ebm.energy(data)
    energies = energies.detach().cpu().numpy()
    viz = EBMVisualizer(data, energies)
    viz.run()
