"""
Command-line interface for Curve2Flood.
"""

import argparse
from .core import Curve2Flood_MainFunction

def main():
    parser = argparse.ArgumentParser(description="Run Curve2Flood flood mapping tool.")
    parser.add_argument("input_file", help="Path to the input configuration file.")
    args = parser.parse_args()

    Curve2Flood_MainFunction(args.input_file)

if __name__ == "__main__":
    main()
