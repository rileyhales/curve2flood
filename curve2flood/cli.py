"""
Command-line interface for Curve2Flood.
"""

import argparse
import logging

from curve2flood import Curve2Flood_MainFunction, LOG

def set_log_level(log_level: str):
    handler = LOG.handlers[0]
    if log_level == 'debug':
        LOG.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    elif log_level == 'info':
        LOG.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
    elif log_level == 'warn':
        LOG.setLevel(logging.WARNING)
        handler.setLevel(logging.WARNING)
    elif log_level == 'error':
        LOG.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)
    else:
        LOG.setLevel(logging.WARNING)
        handler.setLevel(logging.WARNING)
        LOG.warning('Invalid log level. Defaulting to warning.')
        return
        
    LOG.info('Log Level set to ' + log_level)

def main():
    parser = argparse.ArgumentParser(description="Run Curve2Flood flood mapping tool.")
    parser.add_argument("input_file", help="Path to the input configuration file.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output except errors.")
    parser.add_argument('-l', '--log', type=str, help='Log Level', 
                        default='warn', choices=['debug', 'info', 'warn', 'error'])
    args = parser.parse_args()

    set_log_level(args.log)
    Curve2Flood_MainFunction(args.input_file, args.quiet)

if __name__ == "__main__":
    main()
