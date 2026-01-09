#!/usr/bin/env python
"""
Aadhaar Sentinel Dashboard Launcher

Launches the Streamlit dashboard for ALRI monitoring.
Loads precomputed ALRI results from storage if available,
otherwise uses sample data for demonstration.

Usage:
    streamlit run run_dashboard.py

Requirements: 11.1-11.5
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.app import main

if __name__ == "__main__":
    main()
