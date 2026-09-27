import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evolutionary_model import MultiHorizonBTCNet, HORIZONS, HORIZON_NAMES

CHECKPOINT_BEST = "checkpoints/best_model.pt"
DATA_NPZ = "data/btc_features_targets.npz"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate_best_model(model_path=CHECKPOINT_BEST, data_path=DATA_NPZ):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model checkpoint {model_path} not found. Run training first.")

    print(f"Loading best model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    model = MultiHorizonBTCNet(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load test data
    print(f"Loading out-of-sample test dataset from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    Y_test_reg = data["Y_test_reg"]
    Y_test_dir = data["Y_test_dir"]
    prices_test = data["prices_test"]
    timestamps_test = pd.to_datetime(data["timestamps_test"])
    
    n_samples = len(X_test)
    print(f"Evaluating across {n_samples:,} out-of-sample test intervals...")
    
    with torch.no_grad():
        X_t = torch.from_numpy(X_test)
        pred_returns_t, pred_dir_logits_t = model(X_t)
        pred_dir_probs_t = torch.sigmoid(pred_dir_logits_t)
        
    pred_returns = pred_returns_t.numpy()
    pred_dir_probs = pred_dir_probs_t.numpy()
    pred_dir = (pred_dir_probs > 0.5).astype(np.float32)
    
    results = {}
    
    print("\n" + "="*80)
    print(f"{'Horizon':<10} | {'DA (%)':<10} | {'Win Rate (%)':<14} | {'Cum Return (%)':<16} | {'Sharpe Ratio':<14} | {'Max DD (%)':<12}")
    print("="*80)
    
    benchmark_bnh_return = (prices_test[-1] / prices_test[0] - 1.0) * 100.0
    
    pnl_series = {}
    
    for idx, (h, name) in enumerate(zip(HORIZONS, HORIZON_NAMES)):
        actual_dir = Y_test_dir[:, idx]
        actual_ret = Y_test_reg[:, idx]
        p_dir = pred_dir[:, idx]
        
        # Metrics
        da = np.mean(p_dir == actual_dir) * 100.0
        
        # Position: +1 for Long, -1 for Short
        positions = np.where(p_dir > 0.5, 1.0, -1.0)
        trade_returns = positions * actual_ret
        
        # Win rate
        wins = np.sum(trade_returns > 0)
        total_trades = len(trade_returns)
        win_rate = (wins / total_trades) * 100.0
        
        # Non-overlapping stepped or sub-sampled cumulative equity curve
        step = max(1, h)
        sub_returns = trade_returns[::step]
        cum_ret_series = np.cumsum(sub_returns)
        cum_return_pct = (np.exp(np.sum(sub_returns)) - 1.0) * 100.0
        
        # Sharpe ratio
        annual_factor = np.sqrt(525600 / h)
        sharpe = (np.mean(sub_returns) / (np.std(sub_returns) + 1e-8)) * annual_factor
        
        # Max Drawdown
        equity_curve = np.exp(cum_ret_series)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = np.min(drawdown) * 100.0
        
        results[name] = {
            "horizon_minutes": int(h),
            "directional_accuracy_pct": round(float(da), 2),
            "win_rate_pct": round(float(win_rate), 2),
            "cumulative_return_pct": round(float(cum_return_pct), 2),
            "annualized_sharpe": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_dd), 2)
        }
        
        pnl_series[name] = cum_ret_series
        
        print(f"{name:<10} | {da:>8.2f}% | {win_rate:>12.2f}% | {cum_return_pct:>14.2f}% | {sharpe:>12.2f} | {max_dd:>10.2f}%")
        
    print("="*80)
    print(f"Out-of-Sample Buy & Hold Benchmark Return: {benchmark_bnh_return:.2f}%")
    print("="*80 + "\n")
    
    # Generate Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Directional Accuracy by Horizon
    h_names = list(results.keys())
    accs = [results[k]["directional_accuracy_pct"] for k in h_names]
    axes[0, 0].bar(h_names, accs, color="#2563eb", alpha=0.85, edgecolor="#1e3a8a")
    axes[0, 0].axhline(50.0, color="red", linestyle="--", label="Random Baseline (50%)")
    axes[0, 0].set_title("Directional Accuracy by Forecasting Horizon", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(45, max(60, max(accs) + 5))
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio by Horizon
    sharpes = [results[k]["annualized_sharpe"] for k in h_names]
    axes[0, 1].bar(h_names, sharpes, color="#10b981", alpha=0.85, edgecolor="#065f46")
    axes[0, 1].axhline(0.0, color="black", linestyle="-")
    axes[0, 1].set_title("Annualized Sharpe Ratio by Horizon", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative Strategy Performance
    for key in ["1h", "4h", "12h"]:
        if key in pnl_series:
            sub_pnl = np.exp(pnl_series[key]) * 100.0 - 100.0
            axes[1, 0].plot(range(len(sub_pnl)), sub_pnl, label=f"Strategy ({key})")
    axes[1, 0].set_title("Simulated Multi-Horizon Strategy Cumulative Returns", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Cumulative Return (%)")
    axes[1, 0].set_xlabel("Trade Intervals")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Predicted Probability Distribution (1h Horizon)
    idx_1h = HORIZON_NAMES.index("1h")
    axes[1, 1].hist(pred_dir_probs[:, idx_1h], bins=50, color="#8b5cf6", alpha=0.8, edgecolor="#4c1d95")
    axes[1, 1].axvline(0.5, color="red", linestyle="--", label="Decision Threshold (0.5)")
    axes[1, 1].set_title("Prediction Probability Distribution (1-Hour Horizon)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Bullish Probability P(Up)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "evaluation_metrics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved evaluation charts to {plot_path}")
    
    # Save test results JSON
    eval_json_path = os.path.join(PLOTS_DIR, "test_evaluation_results.json")
    with open(eval_json_path, "w") as f:
        json.dump({
            "results_by_horizon": results,
            "benchmark_bnh_return_pct": round(float(benchmark_bnh_return), 2),
            "test_samples": n_samples
        }, f, indent=2)
        
    return results

if __name__ == "__main__":
    evaluate_best_model()
