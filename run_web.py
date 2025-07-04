#!/usr/bin/env python3
"""
Run script for RDB web interface.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rdb.web.app import create_app


def main():
    parser = argparse.ArgumentParser(description='RDB Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--data-dir', help='Data directory path')
    
    args = parser.parse_args()
    
    # Create Flask app
    app = create_app(data_dir=args.data_dir, debug=args.debug)
    
    print(f"Starting RDB Web Interface...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    if args.data_dir:
        print(f"Data directory: {args.data_dir}")
    print(f"\nAccess the web interface at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
