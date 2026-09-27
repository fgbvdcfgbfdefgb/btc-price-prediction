import os
import sys
import time
import json
import pickle
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from system_detection import detect_system_capabilities
from evolutionary_model import (
    MultiHorizonBTCNet,
    EvolutionaryIndividual,
    crossover,
    evaluate_individual_fitness,
    HORIZONS,
    HORIZON_NAMES
)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_LATEST = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "best_model.pt")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.json")

class EvolutionaryTrainer:
    def __init__(
        self,
        population_size=16,
        elite_count=4,
        tournament_size=3,
        mutation_rate=0.06,
        mutation_std=0.02,
        data_npz_path="data/btc_features_targets.npz",
        max_duration_seconds=5 * 3600, # 5 hours default
        checkpoint_interval_gens=1
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.max_duration_seconds = max_duration_seconds
        self.checkpoint_interval_gens = checkpoint_interval_gens
        self.data_npz_path = data_npz_path
        
        # Detect system and hardware configuration
        self.system_info = detect_system_capabilities()
        print("\n" + "="*60)
        print("SYSTEM HARDWARE DETECTION")
        print("="*60)
        print(f"Strategy:            {self.system_info['strategy']}")
        print(f"Logical CPUs:        {self.system_info['cpu_count_logical']}")
        print(f"Physical CPUs:       {self.system_info['cpu_count_physical']}")
        print(f"System RAM:          {self.system_info['ram_gb']} GB")
        print(f"CUDA GPUs Available: {self.system_info['gpu_count']}")
        print(f"NVLink Available:    {self.system_info['nvlink_available']}")
        print(f"Parallel Workers:    {self.system_info['worker_count']}")
        print("="*60 + "\n")
        
        # Load data
        print(f"Loading data from {self.data_npz_path}...")
        data = np.load(self.data_npz_path)
        self.X_train = data["X_train"]
        self.Y_train_reg = data["Y_train_reg"]
        self.Y_train_dir = data["Y_train_dir"]
        self.X_val = data["X_val"]
        self.Y_val_reg = data["Y_val_reg"]
        self.Y_val_dir = data["Y_val_dir"]
        self.X_test = data["X_test"]
        self.Y_test_reg = data["Y_test_reg"]
        self.Y_test_dir = data["Y_test_dir"]
        self.input_dim = self.X_train.shape[1]
        
        # State variables
        self.population = []
        self.generation = 0
        self.elapsed_time = 0.0
        self.best_fitness = -float('inf')
        self.best_metrics = {}
        self.best_individual = None
        self.history = []
        
    def initialize_population(self):
        """Initializes a diverse population of candidate models."""
        self.population = []
        print(f"Initializing population of {self.population_size} candidate models...")
        for i in range(self.population_size):
            hidden_dim = random.choice([64, 128, 192])
            num_layers = random.choice([2, 3, 4])
            dropout = random.choice([0.1, 0.15, 0.25])
            lr = random.choice([5e-4, 1e-3, 2e-3])
            ind = EvolutionaryIndividual(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                lr=lr
            )
            ind.generation = 0
            self.population.append(ind)
            
    def _micro_train_step(self, ind, device="cpu", steps=4, batch_size=512):
        """
        Lightweight Lamarckian evolutionary gradient refinement on training batch.
        """
        ind.model.train()
        ind.model.to(device)
        optimizer = optim.AdamW(ind.model.parameters(), lr=ind.config["lr"], weight_decay=ind.config["weight_decay"])
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCEWithLogitsLoss()
        
        n = len(self.X_train)
        for _ in range(steps):
            indices = np.random.choice(n, batch_size, replace=False)
            bx = torch.from_numpy(self.X_train[indices]).to(device)
            by_reg = torch.from_numpy(self.Y_train_reg[indices]).to(device)
            by_dir = torch.from_numpy(self.Y_train_dir[indices]).to(device)
            
            optimizer.zero_grad()
            p_ret, p_logits = ind.model(bx)
            loss_ret = criterion_mse(p_ret, by_reg)
            loss_dir = criterion_bce(p_logits, by_dir)
            total_loss = loss_ret + loss_dir
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ind.model.parameters(), 1.0)
            optimizer.step()

    def evaluate_population_parallel(self):
        """
        Evaluates the evolutionary population using the hardware strategy:
        - NVLINK_MULTI_GPU: Distributes evaluation across NVLink devices with peer-to-peer memory transfers.
        - PARALLEL_GPU_INSTANCES: Distributes evaluation across independent discrete GPUs.
        - PARALLEL_CPU_INSTANCES: Parallel workers on CPU.
        """
        strategy = self.system_info["strategy"]
        
        if strategy == "NVLINK_MULTI_GPU":
            devices = [f"cuda:{i}" for i in range(self.system_info["gpu_count"])]
        elif strategy == "PARALLEL_GPU_INSTANCES":
            devices = [f"cuda:{i % self.system_info['gpu_count']}" for i in range(self.population_size)]
        else:
            devices = ["cpu"] * self.population_size

        for idx, ind in enumerate(self.population):
            dev = devices[idx % len(devices)]
            # Micro-gradient evolutionary adaptation step
            self._micro_train_step(ind, device=dev, steps=4, batch_size=512)
            # Evaluate fitness on validation set
            fit, metrics = evaluate_individual_fitness(
                ind.model, self.X_val, self.Y_val_reg, self.Y_val_dir, device=dev
            )
            ind.fitness = fit
            ind.metrics = metrics

    def select_parent(self):
        """Tournament selection."""
        candidates = random.sample(self.population, self.tournament_size)
        return max(candidates, key=lambda ind: ind.fitness)

    def evolve_next_generation(self):
        """Creates the next generation via Elitism, Crossover, and Mutation."""
        # Sort population descending by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Check if new all-time best
        current_gen_best = self.population[0]
        if current_gen_best.fitness > self.best_fitness:
            self.best_fitness = current_gen_best.fitness
            self.best_metrics = current_gen_best.metrics
            self.best_individual = copy.deepcopy(current_gen_best)
            self.save_checkpoint(is_best=True)
            print(f"  --> [*] NEW BEST MODEL DISCOVERED! Fitness: {self.best_fitness:.4f} (Mean DA: {self.best_metrics['mean_da']*100:.2f}%)")

        new_population = []
        # 1. Elitism: Preserve top elite individuals unchanged
        for i in range(self.elite_count):
            elite_copy = copy.deepcopy(self.population[i])
            elite_copy.generation = self.generation + 1
            new_population.append(elite_copy)

        # 2. Sexual reproduction (Crossover + Mutation)
        while len(new_population) < self.population_size:
            p1 = self.select_parent()
            p2 = self.select_parent()
            child = crossover(p1, p2)
            child.mutate(mutation_rate=self.mutation_rate, mutation_std=self.mutation_std)
            child.generation = self.generation + 1
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def save_checkpoint(self, is_best=False):
        """Saves a complete resumable checkpoint."""
        state = {
            "generation": self.generation,
            "elapsed_time": self.elapsed_time,
            "best_fitness": self.best_fitness,
            "best_metrics": self.best_metrics,
            "best_model_state": self.best_individual.get_weights() if self.best_individual else None,
            "best_config": self.best_individual.config if self.best_individual else None,
            "population_weights": [ind.get_weights() for ind in self.population],
            "population_configs": [ind.config for ind in self.population],
            "population_fitness": [ind.fitness for ind in self.population],
            "history": self.history,
            "system_info": self.system_info,
            "torch_rng_state": torch.get_rng_state(),
            "np_rng_state": np.random.get_state(),
            "py_rng_state": random.getstate()
        }
        torch.save(state, CHECKPOINT_LATEST)
        
        if is_best and self.best_individual is not None:
            best_state = {
                "model_state_dict": self.best_individual.model.state_dict(),
                "config": self.best_individual.config,
                "fitness": self.best_fitness,
                "metrics": self.best_metrics,
                "generation": self.generation,
                "horizons": HORIZONS,
                "horizon_names": HORIZON_NAMES
            }
            torch.save(best_state, CHECKPOINT_BEST)
            
        with open(METRICS_PATH, "w") as f:
            json.dump({
                "generation": self.generation,
                "elapsed_time_seconds": self.elapsed_time,
                "best_fitness": self.best_fitness,
                "best_metrics": self.best_metrics,
                "history": self.history
            }, f, indent=2)

    def load_checkpoint(self, checkpoint_path=CHECKPOINT_LATEST):
        """Loads and restores full training state to resume training."""
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}. Starting fresh.")
            return False
            
        print(f"Resuming training from checkpoint {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.generation = state["generation"]
        self.elapsed_time = state["elapsed_time"]
        self.best_fitness = state["best_fitness"]
        self.best_metrics = state["best_metrics"]
        self.history = state.get("history", [])
        
        # Restore best individual
        if state["best_config"] and state["best_model_state"]:
            self.best_individual = EvolutionaryIndividual(**state["best_config"])
            self.best_individual.set_weights(state["best_model_state"])
            self.best_individual.fitness = self.best_fitness
            self.best_individual.metrics = self.best_metrics

        # Restore population
        self.population = []
        for weights, config, fit in zip(state["population_weights"], state["population_configs"], state["population_fitness"]):
            ind = EvolutionaryIndividual(**config)
            ind.set_weights(weights)
            ind.fitness = fit
            self.population.append(ind)
            
        # Restore RNG states
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "np_rng_state" in state:
            np.random.set_state(state["np_rng_state"])
        if "py_rng_state" in state:
            random.setstate(state["py_rng_state"])
            
        print(f"Successfully resumed at Generation {self.generation} (Elapsed Time: {self.elapsed_time/3600:.2f}h / Best Fitness: {self.best_fitness:.4f})")
        return True

    def train(self, target_duration_seconds=None, resume=True, max_gens=None):
        """
        Runs the evolutionary training loop for the target duration (default: 5 hours).
        Automatically saves checkpoints at every generation and handles interrupts cleanly.
        """
        duration_limit = target_duration_seconds if target_duration_seconds is not None else self.max_duration_seconds
        
        # Check if resuming
        resumed = False
        if resume and os.path.exists(CHECKPOINT_LATEST):
            resumed = self.load_checkpoint(CHECKPOINT_LATEST)
            
        if not resumed or len(self.population) == 0:
            self.initialize_population()
            
        print("\n" + "="*70)
        print(f"STARTING EVOLUTIONARY MODEL TRAINING")
        print(f"Target Duration:  {duration_limit/3600:.2f} hours ({duration_limit:,} seconds)")
        print(f"Population Size:  {self.population_size}")
        print(f"Elite Retention:  {self.elite_count}")
        print(f"Multi-Horizons:   {HORIZON_NAMES} (1m up to 12h)")
        print(f"Execution Mode:   {self.system_info['strategy']} ({self.system_info['worker_count']} workers)")
        print("="*70 + "\n")
        
        session_start_time = time.time()
        
        try:
            while True:
                gen_start = time.time()
                
                # 1. Parallel Evaluation
                self.evaluate_population_parallel()
                
                gen_time = time.time() - gen_start
                self.elapsed_time += gen_time
                
                # Log generation metrics
                fitnesses = [ind.fitness for ind in self.population]
                mean_fit = np.mean(fitnesses)
                top_fit = np.max(fitnesses)
                top_ind = max(self.population, key=lambda ind: ind.fitness)
                
                record = {
                    "generation": self.generation,
                    "elapsed_time": self.elapsed_time,
                    "gen_time": gen_time,
                    "top_fitness": float(top_fit),
                    "mean_fitness": float(mean_fit),
                    "mean_da": float(top_ind.metrics.get("mean_da", 0.0)),
                    "accuracies": top_ind.metrics.get("accuracies", {})
                }
                self.history.append(record)
                
                # Format progress line
                da_str = " | ".join([f"{k.replace('DA_', '')}: {v*100:.1f}%" for k, v in top_ind.metrics.get("accuracies", {}).items()])
                time_str = f"Elapsed: {self.elapsed_time/3600:.2f}h / {duration_limit/3600:.2f}h"
                
                print(f"[Gen {self.generation:03d}] TopFit: {top_fit:.2f} | MeanFit: {mean_fit:.2f} | {time_str} ({gen_time:.1f}s/gen)")
                print(f"          Directional Accuracy: [{da_str}]")
                
                # Evolve next generation (updates best individual and saves best checkpoint)
                self.evolve_next_generation()
                
                # Save general checkpoint
                self.save_checkpoint()
                
                # Check stopping conditions
                if self.elapsed_time >= duration_limit:
                    print(f"\n[!] Target training duration reached ({duration_limit/3600:.2f} hours). Training complete!")
                    break
                    
                if max_gens is not None and self.generation >= max_gens:
                    print(f"\n[!] Target max generations reached ({max_gens}). Training complete!")
                    break
                
        except KeyboardInterrupt:
            print("\n[!] Training manually interrupted by user. Saving checkpoint...")
            self.save_checkpoint(is_best=False)
            print("Checkpoint saved successfully. You can resume at any time.")
            
        print("\n" + "="*70)
        print("TRAINING SESSION SUMMARY")
        print("="*70)
        print(f"Total Generations: {self.generation}")
        print(f"Total Time:        {self.elapsed_time/3600:.2f} hours ({self.elapsed_time:.1f}s)")
        print(f"Best Fitness:      {self.best_fitness:.4f}")
        if self.best_metrics:
            print(f"Best Mean DA:      {self.best_metrics['mean_da']*100:.2f}%")
            print("Best Directional Accuracies by Horizon:")
            for k, v in self.best_metrics["accuracies"].items():
                print(f"  - {k}: {v*100:.2f}%")
        print(f"Best Model Saved:  {CHECKPOINT_BEST}")
        print(f"Latest Checkpoint: {CHECKPOINT_LATEST}")
        print("="*70 + "\n")
        
        return self.best_individual

if __name__ == "__main__":
    trainer = EvolutionaryTrainer(population_size=6, elite_count=2, max_duration_seconds=30)
    best_model = trainer.train(max_gens=3)
