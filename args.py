import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse training arguments for HuggingFace TrainingArguments.")

    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for model and logs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--workers", type=bool, default=False,
                        help="Persistent workers for dataloader.")
    parser.add_argument("--prefetch_factor", type=int, default=0, help="Prefetch factor for dataloader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or evaluate.")
    parser.add_argument("--cfg_file", type=str, default="", help="Path to config file.")

    args = parser.parse_args()
    return args
