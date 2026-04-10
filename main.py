import argparse
from mlp.controller import LearningBasedController
from difficulty.static_budget import StaticBudgetController
from consistency.controller import ConsistencyController

def main():
    parser = argparse.ArgumentParser(description="Adaptive Early-Exit Strategies for LLMs")
    parser.add_argument("--strategy", type=str, choices=["learning", "static", "consistency"], required=True, help="Early-exit strategy to use")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "strategyqa"], required=True, help="Dataset to evaluate on")
    
    args = parser.parse_args()
    
    print(f"Running experiment with strategy: {args.strategy} on dataset: {args.dataset}")
    
    if args.strategy == "learning":
        strategy = LearningBasedController()
    elif args.strategy == "static":
        strategy = StaticBudgetController()
    elif args.strategy == "consistency":
        strategy = ConsistencyController()
        
    # TODO: Load dataset
    # TODO: Initialize LLM Wrapper
    # TODO: Run evaluation loop
    
    print(" detailed implementation to be added.")

if __name__ == "__main__":
    main()
