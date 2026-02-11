import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import json
from optimizers import FireflyOptimizer

class GAUSS_EBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim, sharp_sig_temp=0.7, dropout_p=0.3, weight_decay=0.0, sigma=1.0):
        super(GAUSS_EBM, self).__init__()

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

def contrastive_divergence_training(dataloader,
                                    rbm,
                                    k=1,
                                    epochs=1000,
                                    lr=1e-2,
                                    batch_size=64,
                                    #device=None,
                                    persistent=False,
                                    lr_momentum=0.0,
                                    weight_decay=0.0,
                                    use_optimizer=False,
                                    verbose=True,
                                    clamp_visible=True):

    xs = []
    count = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        xs.append(x)

    x_all = torch.cat(xs, dim=0).to(device)
    if isinstance(dataloader, torch.Tensor):
        dataset_tensor = dataloader.to(device)
        def _iter_loader():
            n = dataset_tensor.size(0)
            for i in range(0, n, batch_size):
                yield dataset_tensor[i:i+batch_size]
        loader = _iter_loader()
        loader_factory = lambda: _iter_loader()
    else:
        loader_factory = lambda: dataloader

    logs = defaultdict(list)
    best_recon = float('inf')

    v_W = torch.zeros_like(rbm.W.data, device=device)
    v_vb = torch.zeros_like(rbm.v_bias.data, device=device)
    v_hb = torch.zeros_like(rbm.h_bias.data, device=device)
    persistent_chain = None
    eps = 1e-6

    for epoch in range(epochs):
        t0 = time.time()
        epoch_recon = []
        epoch_w_update_norms = []

        loader = loader_factory()
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device).float()

            if rbm_param == 'bern' and clamp_visible:
                batch = batch.clamp(eps, 1 - eps)

            b = batch.size(0)

            pos_h_prob, pos_h_sample = rbm.sample_hidden(batch)
            pos_assoc = torch.matmul(batch.t(), pos_h_prob)
            pos_h_mean = pos_h_prob.mean(dim=0)
            pos_v_mean = batch.mean(dim=0)

            if persistent and persistent_chain is not None:
                v_neg = persistent_chain
            else:
                v_neg = batch.clone().detach()

            for _step in range(k):
                v_neg = rbm.gibbs_step(v_neg)
                if rbm_param == 'bern' and clamp_visible:
                    v_neg = v_neg.clamp(eps, 1 - eps)
                
            if persistent:
                persistent_chain = v_neg.detach()

            neg_h_prob_final, _ = rbm.sample_hidden(v_neg)
                
            neg_assoc = torch.matmul(v_neg.t(), neg_h_prob_final)  # (V, H)
            neg_h_mean = neg_h_prob_final.mean(dim=0)
            neg_v_mean = v_neg.mean(dim=0)

            grad_hb = (pos_h_mean - neg_h_mean)

            #elif rbm_param == 'gauss':
            sigma2 = rbm.sigma ** 2
            grad_W = (pos_assoc - neg_assoc) / (float(b) * sigma2.unsqueeze(1))
            grad_vb = (pos_v_mean - neg_v_mean) / sigma2
            grad_W = grad_W - rbm.weight_decay * rbm.W.data

            if use_optimizer and hasattr(rbm, 'optimizer') and rbm.optimizer is not None:
                opt = rbm.optimizer
                opt_lr = opt.param_groups[0]['lr'] if 'lr' in opt.param_groups[0] else 1.0
                opt.zero_grad()
                rbm.W.grad = - (grad_W / (opt_lr if opt_lr != 0 else 1.0))
                rbm.v_bias.grad = - (grad_vb / (opt_lr if opt_lr != 0 else 1.0))
                rbm.h_bias.grad = - (grad_hb / (opt_lr if opt_lr != 0 else 1.0))
                opt.step()
                update_norm = torch.norm(torch.stack([
                        (opt_lr * rbm.W.grad).view(-1),
                        (opt_lr * rbm.v_bias.grad).view(-1),
                        (opt_lr * rbm.h_bias.grad).view(-1)
                ]))
            else:
                v_W = lr_momentum * v_W + lr * grad_W
                v_vb = lr_momentum * v_vb + lr * grad_vb
                v_hb = lr_momentum * v_hb + lr * grad_hb
                rbm.W.data.add_(v_W)
                rbm.v_bias.data.add_(v_vb)
                rbm.h_bias.data.add_(v_hb)
                update_norm = torch.norm(torch.cat([
                        v_W.view(-1),
                        v_vb.view(-1),
                        v_hb.view(-1)
                ]))
            with torch.no_grad():
                recon_h_prob, h_sample = rbm.sample_hidden(batch)
                recon_v_prob, _ = rbm.sample_visible(recon_h_prob)
                recon_err = torch.mean((batch - recon_v_prob)**2)
                if isinstance(recon_err, torch.Tensor):
                    recon_err = recon_err.detach().cpu().numpy()
                epoch_recon.append(recon_err)
                epoch_w_update_norms.append(update_norm.item())

        epoch_time = time.time() - t0

        epoch_recon = np.asarray(epoch_recon)
        mean_recon = float(np.mean(epoch_recon)) if len(epoch_recon)>0 else 0.0
        mean_update_norm = float(np.mean(epoch_w_update_norms)) if epoch_w_update_norms else 0.0

        logs['epoch'].append(epoch)
        #logs['free_energy'].append(rbm.energy(x_all).mean().item())
        logs['energy_diff'].append(rbm.compute_energy_gap(x_all)[2])
        logs['recon_error'].append(mean_recon)
        logs['update_norm'].append(mean_update_norm)
        logs['epoch_time_s'].append(epoch_time)

        if verbose:
            print(f"[CD-k][{rbm_param}] Epoch {epoch+1}/{epochs} | energy_diff {logs['energy_diff'][-1]:.6e} | recon_mse {mean_recon:.6e} | avg_update_norm {mean_update_norm:.6e} | time {epoch_time:.2f}s")
        if np.isnan(logs['energy_diff'][-1]):
            return rbm, dict(logs)
        if mean_recon < best_recon:
            best_recon = mean_recon
            count = 0
        else:
            count += 1

        #if count >= 50:
        #    print("EARLY STOP TRIGGERED")
        #    return rbm, dict(logs)


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
    train_tensor = torch.stack([train_set[i] for i in range(len(train_set))]).squeeze()
    test_tensor = torch.stack([test_set[i] for i in range(len(test_set))]).squeeze()
    return train_tensor, test_tensor

def save_dataset(data: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

def save_and_plot_histories(firefly_positions_history, fitness_history, alpha_history, objectives_history, output_prefix="firefly_results"):
    pos_df = pd.DataFrame.from_dict(firefly_positions_history, orient="index")
    pos_df.to_csv(f"{output_prefix}/firefly_positions.csv", index_label="Iteration")

    fit_df = pd.DataFrame.from_dict(fitness_history, orient="index")
    fit_df.to_csv(f"{output_prefix}/fitness.csv", index_label="Iteration")

    alpha_df = pd.DataFrame.from_dict(alpha_history, orient="index")
    alpha_df.to_csv(f"{output_prefix}/alpha.csv", index_label="Iteration")

    objectives_df = pd.DataFrame.from_dict(objectives_history, orient='index')
    objectives_df.to_csv(f'{output_prefix}/objectives.csv', index_label='Iteration')

    fig = go.Figure()

    avg_fitness = fit_df.mean(axis=1)
    fig.add_trace(go.Scatter(
        y=avg_fitness,
        mode='lines',
        name='Average Fitness'
    ))

    avg_alpha = alpha_df.mean(axis=1)
    fig.add_trace(go.Scatter(
        y=avg_alpha,
        mode='lines',
        name='Average Alpha'
    ))

    fig.update_layout(
        title="Firefly Algorithm Optimization",
        xaxis_title="Iteration",
        yaxis_title="Value",
        template="plotly_white"
    )

    pyo.plot(fig, filename=f"{output_prefix}/optimization_results.html", auto_open=False)

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
    data = make_dataset_counts(n_samples=2000, n_features=num_features, kind='mixture', sparsity=1, seed=42, n_modes=8, base_rate=2.0, rate_variation=0.7)
    ebm = GAUSS_EBM(num_features, 1)
    ff_opt = FireflyOptimizer(rbm_evaluation, ebm, data, 'gauss', 20, 20)

    

    viz = EBMVisualizer(data, energies)
    viz.run()
