"""
mcp-paperstream: Distributed BERTScore Server for Scientific Paper Review

Ein MCP-Server für verteilte BERTScore-Berechnung auf IoT-Geräten,
kombiniert mit Stable Diffusion für wissenschaftliche Visualisierungen.
"""

from .server import mcp

__version__ = "0.1.0"
__all__ = ["mcp"]
