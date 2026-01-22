"""
Vision-to-Action: Edge-first AI system for perception → cognition → action

An end-to-end AI system combining:
- Hardware-constrained computer vision (YOLOv8)
- Structured ingestion
- LLM-based reasoning (Gemini)
- Agentic orchestration (n8n)

Under 6 GB VRAM constraint with imgshape v4 governance.
"""

__version__ = "0.1.0"
__author__ = "Vision-to-Action Team"

# Core modules
from . import cv
from . import ingestion
from . import cognition
from . import orchestration
from . import configs

__all__ = [
    'cv',
    'ingestion',
    'cognition',
    'orchestration',
    'configs'
]
