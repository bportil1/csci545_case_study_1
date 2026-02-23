'''
Script: csci545_cs_1.py
Name: Bryan Portillo
Date: 2/12/2026
Course: CSCI 545 - HCI

Description: This script is intended to be the basic tools needed for a basic outlier detection system using an energy based
approach(more information may be found in the report). The rest of this script is structured as follows:

    1) 'GAUSS_EBM': The Gaussian Energy Based Model(EBM) class implementation
    2) 'EBM_VISUALIZER': The plotly visualization class built specifically to be used with an energy based model
    3) 'FireflyOptimizer': A vanilla firefly optimization algorithm tailored to the optimization of the EBM implemented here
    4) 'contrastive_divergence_training': an implementation for the contrastive divergence training scheme
    5) Functions needed for general data manipulation
    6) Functions to generate synthetic data
    7) Functions implementing the necessary computations for neural network training
    8) Functions to evaluate and log the quality of the final produced EBM
    9) Functions to facilitate the users ability to read and write EBM instantiations
    10) The main function that runs the program

Usage:
    Run this file with

        python3 csci545_cs_1.py

    This trains the model and automatically produces the visualization. At the moment this script is
    set to produce a synthetic 2d dataset for ease of visualization. The hyperparameters for the optimization class, EBM
    and synthetic generation can all be adjusted according to the users needs or they can learned using the optimizer.


Note on AI Assistance:
Parts of this code (e.g., visualization class notably and other assorted code excerpts)
were generated or assisted by a large language model (LLM) to help with syntax
and structure. All logic, data handling, and final modifications were verified
and completed by the author.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import os
import math
import csv
import argparse
from geomloss import SamplesLoss
from sklearn.neighbors import KernelDensity
from scipy.special import logsumexp
from tqdm import tqdm
import time
import plotly.offline as pyo
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash import callback_context

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAUSS_EBM(nn.Module):
    '''
    Vanilla implementation of a Gaussian EBM. This is borrowed from other work so is at this moment in work in progress
    and as such there are parameters that are not currently in use. See the report for further formal definitions.
    '''
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
        p_h_given_v = torch.clamp(p_h_given_v, eps, 1.0 - eps)
        h_sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, h_sample

    def sample_visible(self, h):
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
        sigma2 = self.sigma ** 2
        term_vis = 0.5 * (((v - self.v_bias) ** 2) / sigma2).sum(dim=1)
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
    '''
    A visualization tool designed to help users better understand probabilities and how they relate to the concept of
    'energy' as presented in EBM models. This tool also demonstrates how this learning scheme can be used in the context
    of a simple outlier detector. As before this is borrowed from other work so may
    have artifacts such as parameters used in other settings.
    '''
    def __init__(self, ebm, data_points, energies):
        self.ebm = ebm.to("cpu")
        self.data_points = data_points
        self.energies = energies
        self.threshold = np.percentile(energies, 90)

        self.user_points = []
        self.user_energies = []
        self.user_distances = []

        self.center = np.mean(self.data_points, axis=0)

        x = self.data_points[:, 0]
        y = self.data_points[:, 1]
        self.x_lin = np.linspace(min(x), max(x), 100)
        self.y_lin = np.linspace(min(y), max(y), 100)

        xx, yy = np.meshgrid(self.x_lin, self.y_lin)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        grid_tensor = torch.tensor(grid, dtype=torch.float32)

        with torch.no_grad():
            grid_energy = self.ebm.energy(grid_tensor).cpu().numpy()

        self.energy_grid = grid_energy.reshape(xx.shape)

        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.layout_app()
        self._callbacks()

    def classify(self, energies, threshold):
        return [
            "OUTLIER" if e > threshold else "IN-DISTRIBUTION"
            for e in energies
        ]

    def recompute_user_metrics(self, threshold):
        if not self.user_points:
            self.user_energies = []
            self.user_distances = []
            self.user_classifications = []
            return

        tensor = torch.tensor(self.user_points, dtype=torch.float32)

        with torch.no_grad():
            self.user_energies = list(
                self.ebm.energy(tensor).cpu().numpy()
            )

        self.user_distances = [
            np.linalg.norm(np.array(p) - self.center)
            for p in self.user_points
        ]

        self.user_classifications = self.classify(
            self.user_energies, threshold
        )

        combined = list(zip(
            self.user_points,
            self.user_energies,
            self.user_distances,
            self.user_classifications
        ))

        combined.sort(key=lambda x: x[1], reverse=True)

        if combined:
            (self.user_points,
             self.user_energies,
             self.user_distances,
             self.user_classifications) = map(list, zip(*combined))

    def layout_app(self):
        self.app.layout = html.Div([

            html.H2("Interactive Outlier Detector", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='empirical-data',
                              style={'height': '80vh'}),
                    html.Div(
                        "Note: Blue heatmap shows empirical density. Black dots represent observed data samples.",
                        style={
                            "textAlign": "center",
                            "marginTop": "5px",
                            "fontSize": "18px"
                        }
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),

                # Create heatmap of energy landscape
                # This visualization provides users with a clear view of which regions are high vs low probability.
                # HCI principle: Supports interpretability and trust by making the model's probabilistic behavior visible.
                html.Div([
                    dcc.Graph(id='energy-heatmap',
                              style={'height': '80vh'}),
                    html.Div(
                        "Note: Red points: flagged as outliers (energy above threshold). "
                        "White points: considered in-distribution.",
                        style={
                            "textAlign": "center",
                            "marginTop": "5px",
                            "fontSize": "18px"
                        }
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),

            ], style={'width': '95%', 'margin': 'auto'}),

            html.H3("Energy Threshold", style={'textAlign': 'center'}),

            # Slider for adjusting energy threshold
            # HCI principle: Gives non-technical users control over outlier classification, supporting satisfaction and trust.
            html.Div(
                [
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
                            "always_visible": True,
                            "placement": "bottom",
                            "style": {
                                "backgroundColor": "ffffff",
                                "color": "white",
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

            html.Div([
                dcc.Graph(id='points-plot',
                          style={'height': '70vh'}),
                html.Div(
                    "All points including user-added points. Outliers red, in-distribution black.",
                    style={"textAlign": "center", "marginTop": "5px", "fontSize": "18px"}
                )
            ], style={'width': '95%', 'margin': 'auto', 'marginTop': '30px'}),

            html.Div([
                html.H3("Test a New Point"),

                dcc.Input(id='input-x', type='number', placeholder='X'),
                dcc.Input(id='input-y', type='number', placeholder='Y'),

                # Allow user to add new points and immediately see classification
                # HCI principle: Real-time feedback reduces cognitive load and increases efficiency for decision making.
                html.Button("Add Point", id='add-point-btn'),
                html.Button("Export CSV", id='export-btn'),

                dcc.Download(id="download-data"),

                html.Div(id='classification-output'),

                html.H4("User Points"),

                dash_table.DataTable(
                    id="user-points-table",
                    columns=[
                        {"name": "Index", "id": "Index"},
                        {"name": "X", "id": "X", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Y", "id": "Y", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Energy", "id": "Energy", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Classification", "id": "Classification"},
                        {"name": "Distance", "id": "Distance", "type": "numeric", "format": {"specifier": ".2f"}},
                        {
                            "name": "Remove",
                            "id": "Remove",
                            "presentation": "markdown"
                        },
                    ],
                    editable=True,
                    row_selectable="single",
                    row_deletable=True,
                    sort_action="native",
                    filter_action="native",
                    style_table={
                        "height": "220px",
                        "overflowY": "auto"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "6px"
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{Classification} = "IN-DISTRIBUTION"', "column_id": "Classification"},
                            "color": "green",
                            "fontWeight": "bold"
                        },
                        {
                            "if": {"filter_query": '{Classification} = "OUTLIER"', "column_id": "Classification"},
                            "color": "red",
                            "fontWeight": "bold"
                        }
                    ]
                )

            ], style={'marginTop': '30px',
                      'textAlign': 'center'})
        ])

    def _callbacks(self):
        @self.app.callback(
            Output('empirical-data', 'figure'),
            Output('energy-heatmap', 'figure'),
            Output('points-plot', 'figure'),
            Output('classification-output', 'children'),
            Output('user-points-table', 'data'),
            Output('user-points-table', 'columns'),
            Input('threshold-slider', 'value'),
            Input('add-point-btn', 'n_clicks'),
            Input('user-points-table', 'data'),
            Input('user-points-table', 'selected_rows'),
            State('input-x', 'value'),
            State('input-y', 'value'),
        )
        def update_all(threshold,
                       add_clicks,
                       table_data,
                       selected_rows,
                       x_val,
                       y_val):

            ctx = callback_context
            trigger = (
                ctx.triggered[0]['prop_id'].split('.')[0]
                if ctx.triggered else None
            )

            if table_data:
                self.user_points = [[float(row["X"]), float(row["Y"])] for row in table_data]
            else:
                self.user_points = []

            if trigger == "add-point-btn":
                if x_val is not None and y_val is not None:
                    self.user_points.append(
                        [float(x_val), float(y_val)]
                    )

            self.recompute_user_metrics(threshold)

            df = pd.DataFrame({
                "Index": list(range(1,
                                    len(self.user_points) + 1)),
                "X": [p[0] for p in self.user_points],
                "Y": [p[1] for p in self.user_points],
                "Energy": self.user_energies,
                "Classification":
                    self.user_classifications,
                "Distance": self.user_distances
            })

            columns = [
                {"name": "Index", "id": "Index"},
                {"name": "X", "id": "X",
                 "type": "numeric",
                 "format": {"specifier": ".2f"}},
                {"name": "Y", "id": "Y",
                 "type": "numeric",
                 "format": {"specifier": ".2f"}},
                {"name": "Energy", "id": "Energy",
                 "type": "numeric",
                 "format": {"specifier": ".3f"}},
                {"name": "Classification",
                 "id": "Classification"},
                {"name": "Distance", "id": "Distance",
                 "type": "numeric",
                 "format": {"specifier": ".2f"}},
            ]

            x = self.data_points[:, 0]
            y = self.data_points[:, 1]

            emp_fig = go.Figure()
            emp_fig.add_trace(go.Histogram2d(x=x, y=y, colorscale=[[0.0, "#f0f9ff"], [0.5, "#74a9cf"], [1.0, "#023858"]],
                                             hoverinfo='skip'
                                             ))
            emp_fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color='black', size=4),
                hoverinfo='skip'
            ))

            heat_fig = go.Figure()
            heat_fig.add_trace(go.Contour(
                x=self.x_lin,
                y=self.y_lin,
                z=self.energy_grid,
                contours=dict(showlines=False),
                colorscale=[[0.0, "#f0f9ff"], [0.5, "#74a9cf"], [1.0, "#023858"]],
                hovertemplate='Energy: %{z:.3f}<extra></extra>'
            ))

            base_colors = [
                'red' if e > threshold else 'black'
                for e in self.energies
            ]

            heat_fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=base_colors,
                            size=6),
                hovertemplate='Energy: %{customdata:.3f}<extra></extra>',
                customdata=self.energies[:, None]
            ))

            points_fig = go.Figure()

            all_x = list(x) + \
                    [p[0] for p in self.user_points]
            all_y = list(y) + \
                    [p[1] for p in self.user_points]

            user_colors = [
                'red' if c == "OUTLIER"
                else 'black'
                for c in self.user_classifications
            ]

            all_colors = base_colors + user_colors
            points_fig.add_trace(go.Scatter(
                x=all_x, y=all_y, mode='markers',
                marker=dict(color=all_colors, size=8, line=dict(width=1, color='black')),
                showlegend=False,
                hovertemplate='Energy: %{customdata:.3f}<extra></extra>',
                customdata=list(self.energies) + self.user_energies
            ))

            if self.user_points:
                user_pts = np.array(self.user_points)  # Convert list to array
                points_fig.add_trace(go.Scatter(
                    x=user_pts[:, 0],
                    y=user_pts[:, 1],
                    mode='markers',
                    marker=dict(color='orange', size=18, symbol='x'),
                    name='User Points'
                ))

            if selected_rows:
                idx = selected_rows[0]
                if idx < len(self.user_points):
                    pt = self.user_points[idx]
                    points_fig.add_trace(go.Scatter(
                        x=[pt[0]],
                        y=[pt[1]],
                        mode='markers',
                        marker=dict(
                            color='green',
                            size=18,
                            symbol='circle-open',
                            line=dict(width=5,
                                      color='green')
                        )
                    ))

            text = ""
            if self.user_points:
                text = (
                    f"Latest: "
                    f"{self.user_points[0]} → "
                    f"{self.user_classifications[0]}"
                )

            return (
                emp_fig,
                heat_fig,
                points_fig,
                text,
                df.to_dict("records"),
                columns
            )

        @self.app.callback(
            Output("download-data", "data"),
            Input("export-btn", "n_clicks"),
            prevent_initial_call=True
        )
        def export_csv(n_clicks):

            df = pd.DataFrame({
                "X": [p[0] for p in self.user_points],
                "Y": [p[1] for p in self.user_points],
                "Energy": self.user_energies
            })

            return dcc.send_data_frame(df.to_csv,
                                       "user_points.csv",
                                       index=False)

    def run(self):
        self.app.run(debug=True)

class FireflyOptimizer:
    '''
    An implementation of a vanilla firefly optimization algorithm.As before this is borrowed from other work so may
    have artifacts such as parameters used in other settings.
    '''
    def __init__(self, objective_fcn, encoder, data, rbm_param='gauss', pop_test=20, iterations=100, gamma=1, alpha=.02, output_path_base='./'):
        self.objective_fcn = objective_fcn
        self.rbm = encoder
        self.data = data.to(device)
        self.rbm_param = rbm_param

        self.pop_test = pop_test
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.output_path_base = output_path_base

        if rbm_param=='stud_t':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, nu, sigma, scale, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (1, 9), (.1, 1), (.1, 1), (8, 32), (4, 4)]
        elif rbm_param=='gauss':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, sigma, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (1, 9), (8, 32), (4, 4)]
        elif rbm_param=='bern':
            ### {lr, momentum, sharp_sig, dropout, weight_decay, batch_size, layer_length}
            self.bounds = [(1e-4, 1e-8), (1e-6, .999), (.1, .999), (1e-6, .9), (1e-6, .9), (8, 32), (4, 4)]

        self.pop_positions = self.initialize_positions('initial')
        self.pop_attractiveness = np.ones(self.pop_test)
        self.pop_fitness = np.zeros(self.pop_test)
        self.pop_alpha = np.ones(self.pop_test)
        self.pop_objectives = [_ for _ in range(self.pop_test)]

        self.firefly_positions_history = {}
        self.fitness_history = {}
        self.alpha_history = {}
        self.objectives_history = {}
        self.compute_fitness()

    def compute_objective(self, position):
        train_out = 0
        #trainer = Trainer(self.data, epochs=1500)
        vis_dim = int(len(next(iter(self.data))))

        if self.rbm_param == 'stud_t':
            self.rbm.reinitialize(vis_dim, int(position[9]), position[2], position[3], position[4], position[5], position[6], position[7])
            dataloader = get_dataloader(self.data, int(position[8]))
            self.rbm, train_out = contrastive_divergence_training(dataloader, self.rbm, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[8])
        elif self.rbm_param == 'gauss':
            self.rbm.reinitialize(vis_dim, int(position[7]), position[2], position[3], position[4], position[5])
            dataloader = get_dataloader(self.data, int(position[6]))
            self.rbm, train_out = contrastive_divergence_training(dataloader, self.rbm, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[6])
        elif self.rbm_param == 'bern':
            self.rbm.reinitialize(vis_dim, int(position[6]), position[2], position[3], position[4])
            dataloader = get_dataloader(self.data, int(position[5]))
            self.rbm, train_out = contrastive_divergence_training(dataloader, self.rbm, epochs= 1500, lr=position[0], lr_momentum=position[1], batch_size=position[5])

        if isinstance(train_out, dict):
            if self.rbm_param == 'stud_t':
                return self.objective_fcn(self.rbm, self.data, compute_lam=True)
            else:
                return self.objective_fcn(self.rbm, self.data)

        if train_out >= 1e9 or np.isnan(train_out):
            return 1e9, 1e9
        else:
            if self.rbm_param == 'stud_t':
                return self.objective_fcn(self.rbm, self.data, compute_lam=True)
            else:
                return self.objective_fcn(self.rbm, self.data)

    def compute_fitness(self):
        for idx in range(self.pop_test):
            fitness, objectives = self.compute_objective(self.pop_positions[idx])
            self.pop_fitness[idx] = fitness
            self.pop_objectives[idx] = objectives

    def generate_new_position(self):
        new_pos = np.zeros(len(self.bounds))
        if self.rbm_param=='stud_t':
            for idx in [0,1,2,3,4,5,6,7]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[8]
            layer_low, layer_high = self.bounds[9]
            possible_batches = np.array([2**i for i in range(3, 7)])
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[8] = int(np.random.choice(possible_batches))
            new_pos[9] = int(np.random.randint(layer_low, layer_high + 1))

        elif self.rbm_param=='gauss':
            for idx in [0,1,2,3,4,5]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[6]
            layer_low, layer_high = self.bounds[7]
            possible_batches = np.array([2**i for i in range(3, 7)])  # 8,16,32,64
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[6] = int(np.random.choice(possible_batches))
            new_pos[7] = int(np.random.randint(layer_low, layer_high + 1))

        elif self.rbm_param=='bern':
            for idx in [0,1,2,3,4]:
                low, high = self.bounds[idx]
                new_pos[idx] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            low, high = self.bounds[2]
            new_pos[2] = np.random.uniform(low, high)
            low, high = self.bounds[3]
            new_pos[3] = np.random.uniform(low, high)
            batch_low, batch_high = self.bounds[5]
            layer_low, layer_high = self.bounds[6]
            possible_batches = np.array([2**i for i in range(3, 7)])
            possible_batches = possible_batches[(possible_batches >= batch_low) & (possible_batches <= batch_high)]
            new_pos[5] = int(np.random.choice(possible_batches))
            new_pos[6] = int(np.random.randint(layer_low, layer_high + 1))
        return new_pos

    def initialize_positions(self, stage='initial'):
        if stage == 'initial':
            return [self.generate_new_position() for _ in range(self.pop_test)]

    def l2_norm(self, ff_idx_1, ff_idx_2):
        diff = self.pop_fitness[ff_idx_1] - self.pop_fitness[ff_idx_2]
        diff = np.clip(diff, -700, 700)
        return 1/(1 + np.exp(diff)+1e-6)

    def compute_attractiveness(self, idx1, idx2):
        norm1 = self.l2_norm(idx1, idx2)
        self.pop_attractiveness[idx1] = (norm1 * self.pop_attractiveness[idx1]) + 1e-6
        return self.pop_attractiveness[idx1]

    def compute_positions(self):
        new_positions = self.pop_positions.copy()
        for i in range(self.pop_test):
            for j in range(self.pop_test):
                if i == j:
                    continue
                new_positions[i] = self.update_position(
                    pos1=new_positions[i],
                    pos2=self.pop_positions[j],
                    fitness1=self.pop_fitness[i],
                    fitness2=self.pop_fitness[j],
                    bounds=self.bounds
                )
        self.pop_positions = new_positions

    def update_position(self, pos1, pos2, fitness1, fitness2, bounds, gamma=1.0, alpha_base=0.02, mutation_prob=0.1):
        pos1 = np.array(pos1, dtype=float)
        pos2 = np.array(pos2, dtype=float)
        distance = np.linalg.norm(pos1 - pos2)
        beta = 1 / (1 + np.exp(fitness1 - fitness2) + 1e-6)
        attraction = beta * np.exp(-gamma * distance**2) * (pos2 - pos1)
        alpha = alpha_base * (1 + np.random.randn(*pos1.shape))
        random_term = alpha * (np.random.rand(*pos1.shape) - 0.5)
        new_pos = pos1 + attraction + random_term
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        new_pos = np.clip(new_pos, lower_bounds, upper_bounds)
        if self.rbm_param == 'stud_t':
            discrete_idx = [8, 9]
        elif self.rbm_param == 'gauss':
            discrete_idx = [6, 7]
        elif self.rbm_param == 'bern':
            discrete_idx = [5, 6]
        for idx in discrete_idx:
            new_pos[idx] = int(np.round(new_pos[idx]))
        for i in range(len(new_pos)):
            if np.random.rand() < mutation_prob:
                perturb = (upper_bounds[i] - lower_bounds[i]) * 0.05
                new_pos[i] += np.random.uniform(-perturb, perturb)
                new_pos[i] = np.clip(new_pos[i], lower_bounds[i], upper_bounds[i])
                if i in discrete_idx:
                    new_pos[i] = int(np.round(new_pos[i]))
        return new_pos

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def optimize(self):
        print("Beggining Hd-Firefly-SA Optimization")
        last_alpha = float('inf')
        nonincreasing_alpha_counter = 0
        hdfa_ctr = 0
        min_reg_fitness = float('inf')
        new_fitness = float('inf')
        best_position = None
        best_fitness = float('inf')
        avg_fitness = 0
        while hdfa_ctr < self.iterations:
            print("Current HdFa Iteration: ", hdfa_ctr)

            self.compute_positions()
            self.compute_fitness()

            avg_fitness = self.pop_fitness[self.pop_fitness != 1e9]
            if avg_fitness.size > 0:
                avg_fitness = np.min(avg_fitness)
            else:
                self.pop_positions = self.initialize_positions('initial')
                continue
                #avg_fitness = 1e9

            for idx in range(self.pop_test):
                new_fitness = self.pop_fitness[idx]
                self.pop_alpha[idx] = np.abs(new_fitness - avg_fitness)

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_position = self.pop_positions[idx]

            alpha_avg = np.max(self.pop_alpha) - np.min(self.pop_alpha)

            self.firefly_positions_history[hdfa_ctr] = self.pop_positions.copy()
            self.fitness_history[hdfa_ctr] = self.pop_fitness.copy()
            self.alpha_history[hdfa_ctr] = self.pop_alpha.copy()
            self.objectives_history[hdfa_ctr] = self.pop_objectives.copy()
            hdfa_ctr += 1

            print("Current Alpha Average: ", alpha_avg)
            print('HDFA Iteration: ', hdfa_ctr)
            print('Steps Without Increasing Alpha: ', nonincreasing_alpha_counter)

            if last_alpha >= alpha_avg:
                nonincreasing_alpha_counter += 1
            else:
                nonincreasing_alpha_counter = 0
            last_alpha = alpha_avg
            if nonincreasing_alpha_counter == 10 or alpha_avg == 0 or hdfa_ctr == self.iterations or math.isnan(nonincreasing_alpha_counter):
                print("Early Convergence")
                break

        print("FF Optimization Complete")
        print("Final Hd-FF Position Estimate: ", best_position, ' with fitness: ', best_fitness)
        save_and_plot_histories(self.firefly_positions_history, self.fitness_history, self.alpha_history, self.objectives_history, self.output_path_base)
        return best_position

def contrastive_divergence_training(dataloader, rbm, k=1, epochs=1000, lr=1e-2, batch_size=64, persistent=False,
                                    lr_momentum=0.0, weight_decay=0.0, use_optimizer=False, verbose=True, clamp_visible=True):
    '''
    An implementation for the base contrastive divergence with persistence(CD-K) algorithm. As before this is borrowed from other work so may
    have artifacts such as parameters used in other settings.
    '''

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

            #if rbm_param == 'bern' and clamp_visible:
            #    batch = batch.clamp(eps, 1 - eps)

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
                #if rbm_param == 'bern' and clamp_visible:
                #    v_neg = v_neg.clamp(eps, 1 - eps)

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
        logs['energy_diff'].append(rbm.compute_energy_gap(x_all)[2])
        logs['recon_error'].append(mean_recon)
        logs['update_norm'].append(mean_update_norm)
        logs['epoch_time_s'].append(epoch_time)

        if verbose:
            print(f"[CD-k][Gaussian] Epoch {epoch+1}/{epochs} | energy_diff {logs['energy_diff'][-1]:.6e} | recon_mse {mean_recon:.6e} | avg_update_norm {mean_update_norm:.6e} | time {epoch_time:.2f}s")
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
    return rbm, dict(logs)

"""
DATA MANIPULATION, LOGGING UTILITY FUNCTIONS
"""

def get_dataloader(data: torch.Tensor, batch_size: int = 64, shuffle: bool = True, drop_last: bool = True):
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

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

    pyo.plot(fig, filename=f"{output_prefix}/optimization_results.html", auto_open=False)

def get_dataloader(data, batch_size=64, shuffle=True, drop_last = True):
    ds = TensorDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def train_test_split(data, test_frac = 0.1):
    n = data.size(0)
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_test
    train_set, test_set = random_split(data, [n_train, n_test])
    train_tensor = torch.stack([train_set[i] for i in range(len(train_set))]).squeeze()
    test_tensor = torch.stack([test_set[i] for i in range(len(test_set))]).squeeze()
    return train_tensor, test_tensor

def train_test_split_tensor(data, test_ratio=0.2, shuffle=True):
    n = data.shape[0]
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    test_size = int(n * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return data[train_idx], data[test_idx]

def save_dataset(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

"""
SYNTHETIC DATA GENERATION FUNCTIONS
"""

def apply_sparsity(arr, sparsity):

    if sparsity <= 0.0:
        return arr
    mask = np.random.rand(*arr.shape) < sparsity
    arr[mask] = 0
    return arr

def poisson_independent(n_samples, n_features, lam = 1.0, sparsity = 0.0, seed = 0):
    samples = np.random.poisson(lam=lam, size=(n_samples, n_features)).astype(np.float32)
    samples = apply_sparsity(samples, sparsity)
    return torch.from_numpy(samples)

def factor_model_counts(n_samples, n_features, n_factors = 5,
                        factor_strength = 1.0, noise = 0.5,
                        sparsity = 0.0, seed = 0):

    F = np.random.normal(size=(n_samples, n_factors)).astype(np.float32)
    W = np.random.normal(scale=factor_strength, size=(n_factors, n_features)).astype(np.float32)
    bias = np.random.normal(loc=0.0, scale=0.5, size=(n_features,)).astype(np.float32)

    logits = F @ W + bias
    logits += np.random.normal(scale=noise, size=logits.shape).astype(np.float32)

    rates = np.exp(logits)
    samples = np.random.poisson(lam=rates).astype(np.float32)

    samples = apply_sparsity(samples, sparsity)
    return torch.from_numpy(samples)

def mixture_of_count_modes(n_samples, n_features, n_modes = 4,
                           base_rate = 1.0, rate_variation = 0.5,
                           sparsity = 0.0, seed = 0):
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

"""
NEURAL COMPUTATION UTILITIES
"""

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

def normalize_global(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return nn.Parameter((tensor - min_val) / (max_val - min_val))

'''
STATISTICAL COMPUTATION FUNCTIONS
'''

def make_betas(T):
    t = np.arange(0, T + 1)
    return (t / T) ** 3

def logmeanexp(a, dim=0):
    m, _ = torch.max(a, dim=dim, keepdim=True)
    return (m + torch.log(torch.mean(torch.exp(a - m), dim=dim, keepdim=True))).squeeze(dim)

def ais_logz_gaussian(rbm, betas, K=500, gibbs_steps_per_beta=1):
    device = rbm.W.device
    D = rbm.num_visible
    sigma = rbm.sigma.to(device)

    mu0 = rbm.v_bias

    logZ0 = 0.5 * D * math.log(2 * math.pi)
    logZ0 += torch.log(sigma).sum().item()

    v = mu0 + torch.randn(K, D, device=device) * sigma

    logw = torch.zeros(K, device=device)

    for t in tqdm(range(1, len(betas)), desc="AIS"):
        beta_prev = betas[t - 1]
        beta = betas[t]

        F_v = rbm.energy(v)
        logw -= (beta - beta_prev) * F_v

        for _ in range(gibbs_steps_per_beta):

            logits_h = beta * (
                    torch.matmul(v / (sigma**2), rbm.W)
                    + rbm.h_bias
            )
            p_h = torch.sigmoid(logits_h)
            h = torch.bernoulli(p_h)

            mean_v = beta * torch.matmul(h, rbm.W.T) + rbm.v_bias
            v = mean_v + torch.randn_like(mean_v) * sigma

    logZ_est = logZ0 + logmeanexp(logw, dim=0).item()

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
    logZ_est, logw = ais_logz_gaussian(rbm, betas, K=AIS_K, gibbs_steps_per_beta=1)
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

'''
RBM HEALTH MEASUREMENT FUNCTIONS
'''

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

'''
RBM READ/WRITE FUNCTIONS
'''

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(use_existing_data=False):
    if not use_existing_data:
        num_features = 2
        os.makedirs('./results', exist_ok=True)
        data = make_dataset_counts(n_samples=2000, n_features=num_features, kind='mixture', sparsity=.4, seed=42, n_modes=4, base_rate=2.0, rate_variation=0.7)
        ebm = GAUSS_EBM(num_features, 1)
        ff_opt = FireflyOptimizer(rbm_evaluation, ebm, data, 'gauss', 5, 5, output_path_base="./results")
        hyperparams = ff_opt.optimize()
        ebm.reinitialize(num_features, 1, hyperparams[2], hyperparams[3], hyperparams[4], hyperparams[5])
        dataloader = get_dataloader(data, int(hyperparams[6]))
        ebm, logs = contrastive_divergence_training(dataloader, ebm)
        df = pd.DataFrame(data.detach().cpu().numpy())
        df.to_csv("./results/aug_data.csv")
        pairwise_joint_scores(ebm, output_basepath='./results')
        check_rbm_layers(ebm, data, './results')
        save_ebm(ebm, 'gauss', './results/gauss_ebm.pt')

        energies = ebm.energy(data.to(device))
        energies = energies.detach().cpu().numpy()
        df = pd.DataFrame(energies)
        df.to_csv("./results/energies.csv")

    # Load existing data and EBM for visualization
    data = pd.read_csv("./results/aug_data.csv", index_col=0).values
    data = torch.tensor(data, dtype=torch.float32)
    data_np = data.numpy()
    energies = pd.read_csv("./results/energies.csv", index_col=0).values.flatten()
    ebm = load_ebm('./results/gauss_ebm.pt')
    ebm = ebm.to('cpu')

    viz = EBMVisualizer(ebm, data_np, energies)
    viz.run()


if __name__ == "__main__":
    '''
    Main workflow:
    1. Generate or load dataset
    2. Train Gaussian EBM (with hyperparameter optimization)
    3. Compute energies and metrics
    4. Launch interactive visualization (EBMVisualizer)
    HCI principle: Users (technical and non-technical) can explore energy landscapes, add points, and adjust thresholds,
    increasing trust, satisfaction, and efficiency in decision-making.

    To use run 
    
        python3 csci545_cs_1.py
        
    and the script will create a 2-d synthetic dataset to give a simple proof of concept.
    
    To run with a pre-trained model and results ensure previously written files are written in a ./results
    directory in the same place as this script and run 
    
        python3 csci545_cs_1.py --use_existing_data      
        
    Final Note on submission I will include results from a generated dataset with 2000 points for quick evaluation
    '''

    parser = argparse.ArgumentParser(description="Train or visualize a Gaussian EBM.")
    parser.add_argument(
        "--use_existing_data",
        action="store_true",
        help="Use pre-generated data and pre-trained EBM for visualization only, otherwise synthetic data will be generated"
             " as a proof of concept"
    )
    args = parser.parse_args()

    main(use_existing_data=args.use_existing_data)