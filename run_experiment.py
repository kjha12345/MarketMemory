"""
Main Experiment Script

Runs the complete experiment comparing Baseline RL vs RARL
"""

import numpy as np
import argparse
from trainer import Trainer
from evaluator import Evaluator
from market_env import MarketEnvironment
from statistical_analysis import analyze_multiple_runs, print_statistical_summary
from utils import (
    plot_training_curves,
    plot_price_comparison,
    plot_metrics_comparison,
    plot_improvement_summary,
    print_comparison_summary,
    save_results_to_csv
)


def main():
    parser = argparse.ArgumentParser(description="RARL vs RL Economic Decision-Making Experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase1_timesteps", type=int, default=50000, help="Phase 1 training timesteps")
    parser.add_argument("--phase2_timesteps", type=int, default=100000, help="Phase 2 training timesteps")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of independent runs for statistical analysis")
    parser.add_argument("--train_multiple_agents", action="store_true", help="Train separate agents for each run (more rigorous but slower)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RARL vs RL Economic Decision-Making Experiment")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Phase 1 timesteps: {args.phase1_timesteps}")
    print(f"Phase 2 timesteps: {args.phase2_timesteps}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Train multiple agents: {args.train_multiple_agents}")
    print("=" * 60)
    
    # Environment configuration
    env_config = {
        "episode_length": 100,
        "base_demand": 2.0,  # Increased from 1.0 to make environment more profitable
        "base_cost": 5.0,
        "max_price": 50.0,
        "min_price": 1.0,
        "supply_capacity": 100.0,
        "fixed_costs": 2.0,  # Reduced from 10.0 to make environment profitable
        "price_elasticity": -2.0,
        "shock_probability": 0.1,  # 10% chance of shock per timestep
        "shock_magnitude": 0.5,  # 50% magnitude change
    }
    
    # RL agent configuration
    rl_config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99
    }
    
    # RARL agent configuration
    rarl_config = {
        "embedding_dim": 384,
        "use_gpu": False,
        "similarity_metric": "cosine",
        "retrieval_weight": 0.3,  # 30% weight on retrieved actions
        "k_retrieval": 5  # Retrieve 5 most similar experiences
    }
    
    all_rl_metrics = []
    all_rarl_metrics = []
    all_comparison_results = []
    all_phase1_results = []
    all_phase2_results = []
    
    # Training phase: train once or multiple times depending on flag
    if args.train_multiple_agents:
        # Train separate agents for each run (more rigorous)
        trained_agents = []
        for run in range(args.num_runs):
            print(f"\n{'=' * 60}")
            print(f"TRAINING RUN {run + 1}/{args.num_runs}")
            print(f"{'=' * 60}")
            
            # Initialize trainer with different seed for each run
            trainer = Trainer(
                env_config=env_config,
                rl_config=rl_config,
                rarl_config=rarl_config,
                seed=args.seed + run
            )
            
            # Phase 1: Stable training
            phase1_results = trainer.train_phase1_stable(args.phase1_timesteps)
            all_phase1_results.append(phase1_results)
            
            # Phase 2: Mixed training with shocks
            phase2_results = trainer.train_phase2_mixed(args.phase2_timesteps)
            all_phase2_results.append(phase2_results)
            
            # Get trained agents
            rl_agent, rarl_agent = trainer.get_agents()
            memory_bank = trainer.get_memory_bank()
            trained_agents.append((rl_agent, rarl_agent, memory_bank))
            
            # Print memory bank statistics
            if run == 0:
                print("\n" + "=" * 60)
                print("MEMORY BANK STATISTICS (Run 1)")
                print("=" * 60)
                stats = memory_bank.get_statistics()
                for key, value in stats.items():
                    print(f"{key}: {value}")
    else:
        # Train once and evaluate multiple times (faster, still valid)
        print("\n" + "=" * 60)
        print("TRAINING PHASE (Single Training)")
        print("=" * 60)
        
        # Initialize trainer
        trainer = Trainer(
            env_config=env_config,
            rl_config=rl_config,
            rarl_config=rarl_config,
            seed=args.seed
        )
        
        # Phase 1: Stable training
        phase1_results = trainer.train_phase1_stable(args.phase1_timesteps)
        all_phase1_results.append(phase1_results)
        
        # Phase 2: Mixed training with shocks
        phase2_results = trainer.train_phase2_mixed(args.phase2_timesteps)
        all_phase2_results.append(phase2_results)
        
        # Get trained agents
        rl_agent, rarl_agent = trainer.get_agents()
        memory_bank = trainer.get_memory_bank()
        
        # Print memory bank statistics
        print("\n" + "=" * 60)
        print("MEMORY BANK STATISTICS")
        print("=" * 60)
        stats = memory_bank.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save agents
        print("\nSaving trained agents...")
        rl_agent.save(f"{args.output_dir}/rl_agent")
        rarl_agent.save(f"{args.output_dir}/rarl_agent")
        memory_bank.save(f"{args.output_dir}/memory_bank")
        
        # Store for evaluation loop
        trained_agents = [(rl_agent, rarl_agent, memory_bank)] * args.num_runs
    
    # Evaluation phase: evaluate with different environment seeds
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    
    for run in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\nEvaluation Run {run + 1}/{args.num_runs}")
        
        rl_agent, rarl_agent, _ = trained_agents[run]
        
        # Create evaluation environment with different seed for each evaluation
        eval_env = MarketEnvironment(**env_config, seed=args.seed + 1000 + run)
        
        # Create evaluator
        evaluator = Evaluator(eval_env)
        
        # Compare agents
        comparison_results = evaluator.compare_agents(
            rl_agent=rl_agent,
            rarl_agent=rarl_agent,
            num_episodes=args.eval_episodes
        )
        
        all_comparison_results.append(comparison_results)
        all_rl_metrics.append(comparison_results["rl_metrics"])
        all_rarl_metrics.append(comparison_results["rarl_metrics"])
    
    # Use first run's results for visualization
    comparison_results = all_comparison_results[0]
    
    # Print summary
    print_comparison_summary(comparison_results)
    
    # Statistical analysis if multiple runs
    if args.num_runs > 1:
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS (Multiple Runs)")
        print("=" * 60)
        statistical_results = analyze_multiple_runs(all_rl_metrics, all_rarl_metrics)
        print_statistical_summary(statistical_results)
        
        # Save statistical results
        import json
        with open(f"{args.output_dir}/statistical_analysis.json", "w") as f:
            # Convert numpy types to native Python types for JSON
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, np.bool8)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (bool, int, float, str)) or obj is None:
                    return obj
                return obj
            
            json.dump(convert_to_serializable(statistical_results), f, indent=2)
        print(f"\nSaved statistical analysis to {args.output_dir}/statistical_analysis.json")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Training curves (use first run's results)
    plot_training_curves(
        all_phase1_results[0]["rl_episode_rewards"] + all_phase2_results[0]["rl_episode_rewards"],
        all_phase1_results[0]["rarl_episode_rewards"] + all_phase2_results[0]["rarl_episode_rewards"],
        save_path=f"{args.output_dir}/training_curves.png"
    )
    
    # Price comparison
    rl_prices = [m["prices"] for m in comparison_results["rl_metrics"]["episode_metrics"]]
    rarl_prices = [m["prices"] for m in comparison_results["rarl_metrics"]["episode_metrics"]]
    plot_price_comparison(
        rl_prices,
        rarl_prices,
        save_path=f"{args.output_dir}/price_comparison.png"
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        comparison_results,
        save_path=f"{args.output_dir}/metrics_comparison.png"
    )
    
    # Improvement summary
    plot_improvement_summary(
        comparison_results,
        save_path=f"{args.output_dir}/improvement_summary.png"
    )
    
    # Save results to CSV
    save_results_to_csv(
        comparison_results,
        save_path=f"{args.output_dir}/results.csv"
    )
    
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
