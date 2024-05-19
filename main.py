import argparse
import yaml
from experiments import run_experiment

def main():
    parser = argparse.ArgumentParser(description="High-Dimensional Sampling Experiments")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    run_experiment(config)

if __name__ == "__main__":
    main()