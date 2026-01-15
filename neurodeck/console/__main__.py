"""
NeuroDeck Console entry point.
"""

import os
import sys
from pathlib import Path

# Add neurodeck to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neurodeck.console.ui import run_console

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("NEURODECK_TOKEN"):
        print("❌ Error: NEURODECK_TOKEN environment variable not set")
        print("Please source config/.env file:")
        print("  source config/.env")
        sys.exit(1)
    
    print("🚀 Starting NeuroDeck Console...")
    run_console()