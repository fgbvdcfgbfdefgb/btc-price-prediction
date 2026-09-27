import os
import time
import copy
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = [1, 5, 15, 30, 60, 240, 720]
HORIZON_NAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "12h"]

class MultiHorizonBTCNet(nn.Module):
    """
    Multi-Horizon Deep Neural Network for Bitcoin Price & Direction Forecasting.
    Simultaneously forecasts expected returns and directional probabilities
    across 7 horizons (1m, 5m, 15m, 30m, 1h, 4h, 12h).
    """
    def __init__(self, input_dim=48, hidden_dim=128, num_layers=3, dropout=0.15, num_horizons=7):
        super(MultiHorizonBTCNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_horizons = num_horizons
        
        # Input Projection & Feature Attention Gating
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Residual Backbone Layers
        self.res_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_layers):
            self.res_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            
        # Multi-Horizon Prediction Heads (Regression + Direction Logits)
        self.return_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_horizons)
        ])
        
        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x):
        gated_x = x * self.feature_gate(x)
        h = F.gelu(self.in_proj(gated_x))
        h = self.in_norm(h)
        
        for layer, norm, drop in zip(self.res_layers, self.norms, self.dropouts):
            residual = h
            h = layer(h)
            h = drop(h)
            h = norm(h + residual)
            
        # Predict for all 7 horizons
        pred_returns = torch.cat([head(h) for head in self.return_heads], dim=-1)     # [B, 7]
        pred_dir_logits = torch.cat([head(h) for head in self.direction_heads], dim=-1) # [B, 7]
        
        return pred_returns, pred_dir_logits

    def predict_direction_probs(self, x):
        _, logits = self.forward(x)
        return torch.sigmoid(logits)


class EvolutionaryIndividual:
    """
    Candidate individual in the evolutionary population.
    Carries neural network weights, hyperparameters, and fitness metrics.
    """
    def __init__(self, input_dim=48, hidden_dim=128, num_layers=3, dropout=0.15, lr=1e-3, weight_decay=1e-5):
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay
        }
        self.model = MultiHorizonBTCNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fitness = -float('inf')
        self.metrics = {}
        self.generation = 0
        self.id = random.randint(100000, 999999)

    def get_weights(self):
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    def mutate(self, mutation_rate=0.06, mutation_std=0.02):
        """
        Applies adaptive Gaussian mutation to network weights and hyperparams.
        """
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_std
                param.add_(mask.float() * noise)
                
        # Occasional hyperparameter mutation
        if random.random() < 0.2:
            self.config["lr"] = max(1e-5, min(1e-2, self.config["lr"] * (1.0 + np.random.normal(0, 0.2))))
        if random.random() < 0.15:
            self.config["dropout"] = max(0.0, min(0.5, self.config["dropout"] + np.random.normal(0, 0.05)))


def crossover(parent1: EvolutionaryIndividual, parent2: EvolutionaryIndividual, alpha=0.5):
    """
    Performs arithmetic crossover between two parent individuals.
    Inherits matching parameter shapes and interpolates weights smoothly.
    """
    # Prefer dominant parent config
    dom_parent, sub_parent = (parent1, parent2) if parent1.fitness >= parent2.fitness else (parent2, parent1)
    child = EvolutionaryIndividual(**dom_parent.config)
    w_dom = dom_parent.get_weights()
    w_sub = sub_parent.get_weights()
    
    child_weights = {}
    for k, v in w_dom.items():
        if k in w_sub and w_sub[k].shape == v.shape and v.is_floating_point():
            gamma = np.random.uniform(0.5 - alpha/2, 0.5 + alpha/2)
            child_weights[k] = gamma * v + (1.0 - gamma) * w_sub[k]
        else:
            child_weights[k] = v.clone()
            
    child.set_weights(child_weights)
    return child


def evaluate_individual_fitness(model, X_val, Y_val_reg, Y_val_dir, device="cpu", batch_size=2048):
    """
    Evaluates individual fitness on validation dataset:
    Fitness is a combination of multi-horizon Directional Accuracy, PnL Sharpe, and Loss.
    """
    model.eval()
    model.to(device)
    
    X_t = torch.from_numpy(X_val).to(device)
    
    total_samples = len(X_val)
    all_pred_returns = []
    all_pred_dir = []
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            bx = X_t[i:i+batch_size]
            p_ret, p_logits = model(bx)
            all_pred_returns.append(p_ret.cpu())
            all_pred_dir.append((p_logits > 0).float().cpu())
            
    pred_returns = torch.cat(all_pred_returns, dim=0).numpy()
    pred_dir = torch.cat(all_pred_dir, dim=0).numpy()
    
    # Calculate directional accuracy per horizon
    accuracies = {}
    total_da = 0.0
    sharpe_estimates = {}
    
    for idx, (h, name) in enumerate(zip(HORIZONS, HORIZON_NAMES)):
        da = np.mean(pred_dir[:, idx] == Y_val_dir[:, idx])
        accuracies[f"DA_{name}"] = float(da)
        total_da += da
        
        # Simulated position return (1 if bullish, -1 if bearish)
        pos = np.where(pred_dir[:, idx] > 0.5, 1.0, -1.0)
        strategy_returns = pos * Y_val_reg[:, idx]
        std_ret = np.std(strategy_returns) + 1e-8
        sharpe = np.mean(strategy_returns) / std_ret * np.sqrt(525600 / h)
        sharpe_estimates[f"Sharpe_{name}"] = float(sharpe)

    mean_da = total_da / len(HORIZONS)
    mean_sharpe = np.mean(list(sharpe_estimates.values()))
    mse_loss = float(np.mean((pred_returns - Y_val_reg) ** 2))
    
    # Composite Fitness Metric
    fitness = (mean_da * 100.0) + (np.clip(mean_sharpe, -5, 5) * 2.0) - (mse_loss * 50.0)
    
    metrics = {
        "mean_da": float(mean_da),
        "mean_sharpe": float(mean_sharpe),
        "mse_loss": float(mse_loss),
        "accuracies": accuracies,
        "sharpe_estimates": sharpe_estimates
    }
    return fitness, metrics
