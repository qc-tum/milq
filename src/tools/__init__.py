"""Module for circuit operations at runtime:
- Assembling: Build one circuit object to run on the same device
- Cutting: Make circuits fit if on available space
- Mapping: Rewrite circuits for hardware connectivity
"""
from .cutting import cut_circuit
