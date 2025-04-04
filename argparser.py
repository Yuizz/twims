import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["torch", "cpp"], help="Engine to use (only in development)")

    return parser.parse_args()

