import torch
import torch.nn as nn
import torch.nn.functional as F
from click import style
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL
from dash import callback_context
import os
import math
import csv
import json
from geomloss import SamplesLoss
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp
from tqdm import tqdm
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
        #h = h.to(device)
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
        #v = v.to(self.device)
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
    def __init__(self, ebm, data_points, energies):
        self.ebm = ebm.to('cpu')
        self.data_points = data_points
        self.energies = energies
        self.threshold = np.percentile(energies, 90)
        self.user_points = []       # list of [x, y]
        self.user_distances = []    # list of distances from center
        self.user_classifications = []  # store classification for each point
        self.user_energies = []
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.layout_app()

    def layout_app(self):
        self.app.layout = html.Div([
            html.H2("Interactive Outlier Detector", style={'textAlign': 'center'}),

            # ----------------------------
            # TOP: Two Plots Side by Side
            # ----------------------------
            html.Div([
                html.Div([
                    dcc.Graph(id='empirical-data'),
                    html.Div(
                        "Blue heatmap shows empirical density. Black dots represent observed data samples.",
                        style={
                            "textAlign": "center",
                            "marginTop": "5px",
                            "fontSize": "14px"
                        }
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(id='energy-heatmap'),
                    html.Div(
                        "Red points: flagged as outliers (energy above threshold). "
                        "White points: considered in-distribution.",
                        style={
                            "textAlign": "center",
                            "marginTop": "5px",
                            "fontSize": "14px"
                        }
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),

            ], style={'width': '95%', 'margin': 'auto'}),
            html.H4("Energy Threshold", style={'textAlign': 'center'}),
            html.Div(
                [
                    # Background track (simulates a colored rail)
                    # The actual slider on top
                    dcc.Slider(
                        id='threshold-slider',
                        min=float(np.min(self.energies)),
                        max=float(np.max(self.energies)),
                        step=0.01,
                        value=float(self.threshold),
                        marks={
                            float(np.min(self.energies)): {
                                "label": f"{np.min(self.energies):.2f}",
                                "style": {"color": "#2c3e50", "fontWeight": "bold"}
                            },
                            float(np.max(self.energies)): {
                                "label": f"{np.max(self.energies):.2f}",
                                "style": {"color": "#2c3e50", "fontWeight": "bold"}
                            }
                        },
                        tooltip={
                            "always_visible": True,          # show tooltip all the time
                            "placement": "bottom",
                            "style": {
                                "backgroundColor": "ffffff",  # dark blue background
                                "color": "white",               # white text
                                "fontWeight": "bold",
                                "borderRadius": "4px",
                                "padding": "2px 6px",
                                "fontSize": "12px"
                            },
                        }
                    ),
                ],
                style={
                    "width": "60%",
                    "margin": "40px auto",
                    "position": "relative",
                }
            ),

            # ----------------------------
            # BOTTOM: All Points Only
            # ----------------------------
            html.Div([
                dcc.Graph(id='points-plot'),
                html.Div(
                    "All points including user-added points. Outliers red, in-distribution black.",
                    style={"textAlign": "center", "marginTop": "5px", "fontSize": "14px"}
                )
            ], style={'width': '95%', 'margin': 'auto', 'marginTop': '30px'}),

            # ----------------------------
            # CONTROLS
            # ----------------------------
            html.Div([
                html.H4("Test a New Point"),

                dcc.Input(
                    id='input-x',
                    type='number',
                    placeholder='X value',
                    style={'marginRight': '10px'}
                ),

                dcc.Input(
                    id='input-y',
                    type='number',
                    placeholder='Y value',
                    style={'marginRight': '10px'}
                ),

                html.Button(
                    "Add Point",
                    id='add-point-btn',
                    n_clicks=0
                ),

                html.Div(id='classification-output', style={'marginTop': '10px'}),
                html.Div([
                    html.H5("User-added Points:"),
                    html.Div(
                        id='user-points-list',
                        style={
                            'border': '1px solid #ccc',
                            'height': '150px',
                            'overflowY': 'scroll',
                            'padding': '5px'
                        }
                    )
                ])
            ],
                style={'textAlign': 'center', 'marginTop': '30px'})
        ])

        @self.app.callback(
            Output('empirical-data', 'figure'),
            Output('energy-heatmap', 'figure'),
            Output('points-plot', 'figure'),
            Output('classification-output', 'children'),
            Output('user-points-list', 'children'),
            Input('threshold-slider', 'value'),
            Input('add-point-btn', 'n_clicks'),
            Input({'type': 'remove-btn', 'index': ALL}, 'n_clicks'),
            State('input-x', 'value'),
            State('input-y', 'value')
        )

        def update_plots(threshold, add_nclicks, remove_nclicks_list, x_val, y_val):
            # ----------------------------
            # Remove button clicked
            # ----------------------------
            ctx = callback_context
            #print('first: ', ctx.triggered[0]['prop_id'])

            e_id = ctx.triggered[0]['prop_id']
            event = ctx.triggered[0]['prop_id'].split('.')[0]
            #print(add_nclicks, remove_nclicks_list)

            if  ctx.triggered and len(remove_nclicks_list)>0:
                triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
                try:
                    trigger_dict = eval(triggered_id)
                    if trigger_dict['type'] == 'remove-btn':
                        idx_to_remove = trigger_dict['index']
                        if 0 <= idx_to_remove < len(self.user_points):
                            self.user_points.pop(idx_to_remove)
                            self.user_energies.pop(idx_to_remove)
                            self.user_classifications.pop(idx_to_remove)
                            self.user_distances.pop(idx_to_remove)
                except:
                    pass

            # ----------------------------
            # Add new point
            # ----------------------------
            if event == 'add-point-btn' and add_nclicks > 0 and x_val is not None and y_val is not None:
                try:
                    new_point = [float(x_val), float(y_val)]
                except (TypeError, ValueError):
                    new_point = None
                if new_point is not None:
                    self.user_points.append(new_point)
                    center = np.mean(self.data_points, axis=0)
                    distance = np.linalg.norm(np.array(new_point) - center)
                    self.user_distances.append(distance)

                    user_tensor = torch.tensor([new_point], dtype=torch.float32)
                    with torch.no_grad():
                        energy = self.ebm.energy(user_tensor).cpu().numpy()[0]
                    self.user_energies.append(energy)
                    classification = "OUTLIER" if energy > threshold else "IN-DISTRIBUTION"
                    self.user_classifications.append(classification)

            # ----------------------------
            # Build user table
            # ----------------------------
            user_table_rows = []
            for i, pt in enumerate(self.user_points):
                energy = self.user_energies[i]
                classification = self.user_classifications[i]
                distance = self.user_distances[i]
                color = "red" if classification == "OUTLIER" else "green"
                user_table_rows.append(
                    html.Tr([
                        html.Td(i+1),
                        html.Td(f"{pt[0]:.2f}"),
                        html.Td(f"{pt[1]:.2f}"),
                        html.Td(f"{energy:.2f}", style={'color': color, 'fontWeight': 'bold'}),
                        html.Td(classification, style={'color': color, 'fontWeight': 'bold'}),
                        html.Td(f"{distance:.2f}"),
                        html.Td(html.Button("Remove", id={'type': 'remove-btn', 'index': i}, n_clicks=0))
                    ])
                )

            user_table = html.Table(
                [html.Tr([
                    html.Th("Index"), html.Th("X"), html.Th("Y"),
                    html.Th("Energy"), html.Th("Classification"), html.Th("Distance to Center"), html.Th("Remove")
                ])] + user_table_rows,
                style={'width': '90%', 'borderCollapse': 'collapse', 'textAlign': 'center'}
            )

            # ----------------------------
            # Compute ranges/colors
            # ----------------------------
            x = self.data_points[:, 0]
            y = self.data_points[:, 1]
            all_x = list(x)
            all_y = list(y)
            all_colors = ['red' if e > threshold else 'black' for e in self.energies]

            if self.user_points:
                user_array = np.array(self.user_points)
                all_x.extend(user_array[:, 0])
                all_y.extend(user_array[:, 1])
                user_colors = ['red' if c == "OUTLIER" else 'black' for c in self.user_classifications]
                all_colors.extend(user_colors)

            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
            y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
            x_range = [x_min - x_pad, x_max + x_pad]
            y_range = [y_min - y_pad, y_max + y_pad]

            # ----------------------------
            # Top-left: Empirical data distribution
            # ----------------------------
            emp_fig = go.Figure()
            emp_fig.add_trace(go.Histogram2d(
                x=x, y=y, nbinsx=30, nbinsy=30,
                colorscale=[[0.0, "#f0f9ff"], [0.5, "#74a9cf"], [1.0, "#023858"]]
            ))
            emp_fig.add_trace(go.Scatter(
                x=x, y=y, mode='markers', marker=dict(color='black', size=4), showlegend=False
            ))
            emp_fig.update_layout(title="Empirical Data Distribution", xaxis_title="X", yaxis_title="Y")                                  #xaxis=dict(range=), yaxis=dict(range=y_range))

            # ----------------------------
            # Top-right: Energy heatmap
            # ----------------------------
            x_lin = np.linspace(min(x), max(x), 100)
            y_lin = np.linspace(min(y), max(y), 100)
            xx, yy = np.meshgrid(x_lin, y_lin)
            grid = np.column_stack([xx.ravel(), yy.ravel()])
            grid_tensor = torch.tensor(grid, dtype=torch.float32)
            with torch.no_grad():
                grid_energy = self.ebm.energy(grid_tensor).cpu().numpy()
            energy_grid = grid_energy.reshape(xx.shape)

            heat_fig = go.Figure()
            heat_fig.add_trace(go.Contour(
                x=x_lin, y=y_lin, z=energy_grid,
                colorscale=[[0.0, "#f0f9ff"], [0.5, "#74a9cf"], [1.0, "#023858"]],
                contours=dict(showlines=False), colorbar=dict(title="Energy")
            ))
            heat_fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=all_colors[:len(x)], size=6, line=dict(width=1, color='black')),
                showlegend=False
            ))
            heat_fig.update_layout(title=f"Energy Heatmap (Threshold: {threshold:.2f})",
                                   xaxis_title="X", yaxis_title="Y")
                                   #xaxis=dict(range=x_range), yaxis=dict(range=y_range))

            # ----------------------------
            # Bottom: Points only
            # ----------------------------
            points_fig = go.Figure()
            points_fig.add_trace(go.Scatter(
                x=all_x, y=all_y, mode='markers',
                marker=dict(color=all_colors, size=8, line=dict(width=1, color='black')),
                showlegend=False
            ))
            if self.user_points:
                points_fig.add_trace(go.Scatter(
                    x=user_array[:, 0],
                    y=user_array[:, 1],
                    mode='markers',
                    marker=dict(color='orange', size=10, symbol='x'),
                    name='User Points'
                ))
            points_fig.update_layout(title="Points (User Points Added)")
                                     #xaxis=dict(range=x_range), yaxis=dict(range=y_range))

            # ----------------------------
            # Latest classification text
            # ----------------------------
            classification_text = ""
            if self.user_points:
                classification_text = f"Latest Point: {self.user_points[-1]} → {self.user_classifications[-1]}"

            return emp_fig, heat_fig, points_fig, classification_text, user_table

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

def train_test_split_tensor(data, test_ratio=0.2, shuffle=True, seed=None):
    n = data.shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return data[train_idx], data[test_idx]

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

def pairwise_sq_dist(X, Y):
    XX = (X**2).sum(axis=1)[:, None]
    YY = (Y**2).sum(axis=1)[None, :]
    return XX + YY - 2 * (X @ Y.T)

def em_loss(decoding, data):
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    em = sinkhorn(decoding, data)
    return em.detach().cpu().numpy()

def kl_divergence(p, q):
    kl = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log())
    return kl.sum().mean()

def contractive_loss(z, x):
    x = x.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    loss = z.pow(2).sum()
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )[0]
    if grad is None:
        return (torch.tensor(0., requires_grad=True, device=x.device)).detach().cpu()
    return (grad.pow(2).sum() / x.size(0)).detach().cpu()

def compute_divergences(decoding, basedata):
    kl = kl_divergence(decoding, basedata)
    em = em_loss(decoding, basedata)
    return em, kl

def make_betas(T):
    t = np.arange(0, T + 1)
    return (t / T) ** 3

def logmeanexp(a, dim=0):
    m, _ = torch.max(a, dim=dim, keepdim=True)
    return (m + torch.log(torch.mean(torch.exp(a - m), dim=dim, keepdim=True))).squeeze(dim)

def ais_logz(rbm, betas, K=500, gibbs_steps_per_beta=1, base_visible_bias=None):
    D = rbm.v_bias.shape[0]
    device = rbm.W.device
    T = len(betas) - 1
    if base_visible_bias is None:
        raise ValueError("base_visible_bias must be provided (logit of train data mean).")
    b0 = torch.tensor(base_visible_bias, device=device, dtype=torch.float32)
    logZ0 = torch.sum(torch.log1p(torch.exp(b0))).item()
    p0 = torch.sigmoid(b0)
    v = (torch.rand(K, D, device=device) < p0.unsqueeze(0)).float()
    logw = torch.zeros(K, device=device)
    for t in tqdm(range(1, T + 1), desc="AIS"):
        beta_prev = betas[t - 1]
        beta = betas[t]
        delta = beta - beta_prev
        F_v = rbm.energy(v)
        E0_v = -(b0.unsqueeze(0) * v).sum(dim=1)
        logw = logw - delta * (E0_v + F_v)
        for _ in range(gibbs_steps_per_beta):
            beff = (1.0 - beta) * b0 + beta * rbm.v_bias
            ceff = beta * rbm.h_bias
            Weff = beta * rbm.W
            logits_h = F.linear(v, Weff.t(), ceff)
            p_h = torch.sigmoid(logits_h)
            h = (torch.rand_like(p_h) < p_h).float()
            logits_v = F.linear(h, Weff, beff)
            p_v = torch.sigmoid(logits_v)
            v = (torch.rand_like(p_v) < p_v).float()
    logw = logw.cpu()
    logZ_est = logZ0 + float(logmeanexp(logw, dim=0).item())
    return logZ_est, logw.detach().cpu().numpy()

def measure_log_likelihood(rbm, data):
    H = data.shape[1]
    AIS_K = 500
    AIS_T = 2000
    betas = make_betas(AIS_T)
    bandwidth_grid = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.45, 1.50]
    train, test = train_test_split_tensor(data)
    train_mean = train.mean(axis=0)
    p_clip = np.clip(train_mean.detach().cpu().numpy(), 1e-6, 1 - 1e-6)
    b0 = np.log(p_clip) - np.log(1 - p_clip)
    logZ_est, logw = ais_logz(rbm, betas, K=AIS_K, gibbs_steps_per_beta=1, base_visible_bias=b0)
    train = train.to(rbm.W.device)
    test = test.to(rbm.W.device)
    all_test_v = test
    with torch.no_grad():
        F_test = rbm.energy(all_test_v).cpu().numpy()
    logp_test = -F_test - logZ_est
    mean_nll = -logp_test.mean()
    with ((torch.no_grad())):
        Z_train, _ = rbm.sample_hidden(train)
        Z_train = Z_train.cpu().numpy()
        Z_test, _ = rbm.sample_hidden(test)
        Z_test = Z_test.cpu().numpy()
    mu = Z_train.mean(axis=0, keepdims=True)
    sd = Z_train.std(axis=0, keepdims=True) + 1e-8
    Z_train_std = (Z_train - mu) / sd
    Z_test_std  = (Z_test - mu) / sd
    subsample_for_cv = 2000
    if Z_train_std.shape[0] > subsample_for_cv:
        idx = np.random.choice(Z_train_std.shape[0], subsample_for_cv, replace=False)
        Ztr_cv = Z_train_std[idx]
    else:
        Ztr_cv = Z_train_std
    best_h, best_ll = select_bandwidth_loocv(Ztr_cv, bandwidth_grid)
    kde = KernelDensity(kernel='gaussian', bandwidth=best_h)
    kde.fit(Z_train_std)
    log_density_train = kde.score_samples(Z_train_std)
    log_density_test  = kde.score_samples(Z_test_std)
    avg_log_like_train = np.mean(log_density_train)
    avg_log_like_test  = np.mean(log_density_test)
    ll_test_parzen = parzen_loglik(Z_train_std, Z_test_std, best_h)
    mean_ll_parzen = ll_test_parzen.mean()
    bits_per_dim_latent = -mean_ll_parzen / (H * math.log(2))
    return logZ_est, mean_nll, mean_ll_parzen, bits_per_dim_latent, avg_log_like_train, avg_log_like_test, avg_log_like_train, avg_log_like_test

def parzen_loglik(Z_train, Z_test, h):
    N, d = Z_train.shape
    D2 = pairwise_sq_dist(Z_test, Z_train)
    const = -0.5 * d * math.log(2 * math.pi * (h ** 2))
    logK = -0.5 * D2 / (h ** 2) + const
    ll = logsumexp(logK, axis=1) - math.log(N)
    return ll

def select_bandwidth_loocv(Z_train, grid):
    best_h, best_ll = None, -np.inf
    for h in grid:
        D2 = pairwise_sq_dist(Z_train, Z_train)
        np.fill_diagonal(D2, np.inf)
        const = -0.5 * Z_train.shape[1] * math.log(2 * math.pi * (h ** 2))
        logK = -0.5 * D2 / (h ** 2) + const
        ll_i = logsumexp(logK, axis=1) - math.log(Z_train.shape[0] - 1)
        avg_ll = ll_i.mean()
        if avg_ll > best_ll:
            best_ll, best_h = avg_ll, h
    return best_h, best_ll

def normalize_global(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return nn.Parameter((tensor - min_val) / (max_val - min_val))

def partition_fcn_approx(encoder, n_chains=10, n_intermediate=10, seed=42):
    W = encoder.W
    torch.manual_seed(seed)
    D, k = W.shape
    betas = torch.linspace(0, 1, steps=n_intermediate).to(device)
    v = torch.bernoulli(torch.full((n_chains, D), .05)).to(device)
    log_W = torch.zeros(n_chains).to(device)
    for i in range(1, n_intermediate):
        beta_prev = betas[i-1]
        beta_curr = betas[i]
        def energy(v_sample, beta):
            h_probs, h_sample = encoder.sample_hidden(v_sample)
            vx = torch.matmul(v_sample, W)
            energy_val = -torch.sum(vx * h_sample, dim=1)
            energy_val -= torch.matmul(v_sample, encoder.v_bias)
            energy_val -= torch.matmul(h_sample, encoder.h_bias)
            return energy_val, h_sample

        E_prev, _ = energy(v, beta_prev)
        E_curr, h = energy(v, beta_curr)
        log_W += -(E_curr - E_prev)
    log_Z_base = D * torch.log(torch.tensor(2.0))
    log_Z_est = log_Z_base + torch.logsumexp(log_W, dim=0)
    return log_Z_est.item()

def pairwise_joint_scores(encoder, eps=1e-10, output_basepath=".", write_out=True):
    os.makedirs(output_basepath, exist_ok=True)
    W = encoder.W
    b = encoder.v_bias
    c = encoder.h_bias
    bias_matrix = b[:, None] + c[None, :]
    energy = -W - bias_matrix
    energy = normalize_global(energy)
    unnorm_joint = torch.exp(-energy.T)
    Z_approx = partition_fcn_approx(encoder)
    unnorm_joint /= Z_approx
    p_joint = unnorm_joint / unnorm_joint.sum()
    p_x = p_joint.sum(dim=0)
    p_h = p_joint.sum(dim=1)
    pmi = torch.log(p_joint + eps) - torch.log(p_h[:, None] + eps) - torch.log(p_x[None, :] + eps)
    pmi = (pmi - pmi.mean(dim=0)) / (pmi.std(dim=0) + eps)
    if write_out:
        pmi_df = pd.DataFrame(pmi.T.detach().cpu().numpy())
        pmi_df.to_csv(f'{output_basepath}/pmi_{encoder.num_visible}_{encoder.num_hidden}.csv', header=False, index=False)
        joint_df = pd.DataFrame(p_joint.T.detach().cpu().numpy())
        joint_df.to_csv(f'{output_basepath}/joint_{encoder.num_visible}_{encoder.num_hidden}.csv', header=False, index=False)
        pri_v_df = pd.DataFrame(p_x.detach().cpu().numpy())
        pri_v_df.to_csv(f'{output_basepath}/pri_v_{encoder.num_visible}_{encoder.num_hidden}.csv', header=False, index=False)
        pri_h_df = pd.DataFrame(p_h.detach().cpu().numpy())
        pri_h_df.to_csv(f'{output_basepath}/pri_h_{encoder.num_visible}_{encoder.num_hidden}.csv', header=False, index=False)
    return pmi, p_joint, p_x, p_h

def check_rbm_layers(ebm, x, output_basepath='rbm_layer_diagnostics.csv'):
    results = []
    v_recon, h_sample = ebm.forward(x)
    v_recon, _ = ebm.sample_visible(v_recon)
    recon_loss = torch.nn.functional.mse_loss(v_recon.to(device), x.to(device)).item()
    em, kl = compute_divergences(v_recon.to(device), x.to(device))
    tc = total_correlation(x.to(device), ebm.to(device))
    ct = contractive_loss(v_recon, x)
    mlogZ_est, mean_nll, mean_ll_parzen, bits_per_dim_latent, avg_log_like_train, avg_log_like_test, avg_log_like_train, avg_log_like_test = measure_log_likelihood(ebm, x)
    neg_energy, pos_energy, energy_diff = ebm.compute_energy_gap(x.to(device))
    free_energy = ebm.energy(x.to(device))
    result_row = {
        'layer': 0,
        'visible_dim': ebm.num_visible,
        'hidden_dim': ebm.num_hidden,
        'reconstruction_mse': recon_loss,
        'neg_energy': neg_energy,
        'pos_energy': pos_energy,
        'energy_diff': energy_diff,
        'free_energy': free_energy,
        'Total_correlation': tc,
        'Contractive_penalty': ct,
        'KL_Loss': kl,
        'EM_Loss': em,
        'AIS_logZ_Est': mlogZ_est,
        'AIS_mean_NLL': mean_nll,
        'Parzen_mean_LL': mean_ll_parzen,
        'bits_per_dim_latent': bits_per_dim_latent,
        'train_ll_density: ': avg_log_like_train,
        'test_ll_density: ': avg_log_like_test
    }
    results.append(result_row)
        #data_hm = x.detach().cpu().numpy()
        #pca, z = compress_to_2d(data_hm)
        #energy = energy_on_grid(rbm, x, device=device)
        #plot_energy_map(x, z, energy, f"{output_basepath}/energy_map.png")
    path = f'{output_basepath}/rbm_layer_diagnostics.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

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

def save_ebm(model, ebm_type, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if ebm_type == 'stud_t':
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "visible_dim": model.num_visible,
                "hidden_dim": model.num_hidden,
                "nu": model.nu,
                "sigma_sq": model.sigma,
            }
        }
    elif ebm_type == 'gauss':
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "visible_dim": model.num_visible,
                "hidden_dim": model.num_hidden,
                "sharp_sig_temp": model.sharp_sig_temp,
                "dropout_p": model.dropout.p,
                "weight_decay": model.weight_decay,
                "sigma": model.sigma
            }
        }
    elif ebm_type == 'bern':
        checkpoint = {
            "state_dict": model.state_dict(),
            "config": {
                "visible_dim": model.num_visible,
                "hidden_dim": model.num_hidden,
                "sharp_sig_temp": model.sharp_sig_temp,
                "dropout_p": model.dropout,
                "weight_decay": model.weight_decay,
            }
        }
    torch.save(checkpoint, path)

def load_ebm(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg.get("dropout_p"), torch.nn.Dropout):
        cfg["dropout_p"] = cfg["dropout_p"].p
    if isinstance(cfg.get("sigma"), torch.Tensor):
        cfg["sigma"] = cfg["sigma"].cpu()
    model = GAUSS_EBM(**cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

if __name__ == "__main__":
    '''
    num_features = 2
    os.makedirs('./results', exist_ok=True)
    data = make_dataset_counts(n_samples=2000, n_features=num_features, kind='mixture', sparsity=.4, seed=42, n_modes=4, base_rate=2.0, rate_variation=0.7)
    #data = make_dataset_counts(n_samples=2000, n_features=num_features, kind='factor', sparsity=.2, seed=42, n_factors=4, noise=.8)
    ebm = GAUSS_EBM(num_features, 1)
    ff_opt = FireflyOptimizer(rbm_evaluation, ebm, data, 'gauss', 3, 3, output_path_base="./results")
    hyperparams = ff_opt.optimize()
    ebm.reinitialize(num_features, 1, hyperparams[2], hyperparams[3], hyperparams[4], hyperparams[5])
    dataloader = get_dataloader(data, int(hyperparams[6]))
    ebm, logs = contrastive_divergence_training(dataloader, ebm)
    df = pd.DataFrame(data.detach().cpu().numpy())
    df.to_csv("./results/aug_data.csv")
    pairwise_joint_scores(ebm, output_basepath='./results')
    check_rbm_layers(ebm, data, './results')
    save_ebm(ebm, 'gauss', './results/gauss_ebm.pt')
    
    #data_tensor = torch.tensor(data, dtype=torch.float32)
    energies = ebm.energy(data.to(device))
    energies = energies.detach().cpu().numpy()
    df = pd.DataFrame(energies)
    df.to_csv("./results/energies.csv")
    '''
    data = pd.read_csv("./results/aug_data.csv", index_col=0).values
    energies = pd.read_csv("./results/energies.csv", index_col=0).values.flatten()
    ebm = load_ebm('./results/gauss_ebm.pt')

    ebm = ebm.to('cpu')
    #ebm.W = ebm.W.to('cpu')
    #ebm.v_bias = ebm.v_bias.to('cpu')
    #ebm.h_bias = ebm.h_bias.to('cpu')
    #print(ebm.to(device), ebm.W.device, ebm.v_bias.device, ebm.h_bias.device)
    #print("energies: ", energies)
    viz = EBMVisualizer(ebm, data, energies)
    viz.run()