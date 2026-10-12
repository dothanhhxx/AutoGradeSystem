"""
Hybrid ASAG Grader - Web Application Module

A modular Python package for Automated Short Answer Grading (ASAG)
with customizable weights for web deployment.

Author: HybridASAG Team
Version: 1.0.0
"""

import sys

# Console output in this package contains emoji; make streams UTF-8 so printing
# never raises UnicodeEncodeError on Windows (default cp1252 console encoding).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from .models import MetricScores, LLMFeedback, GradingResult, DEFAULT_WEIGHTS
from .grader import HybridASAGGrader
from .config import GraderConfig

__all__ = [
    "MetricScores",
    "LLMFeedback", 
    "GradingResult",
    "DEFAULT_WEIGHTS",
    "HybridASAGGrader",
    "GraderConfig",
]

__version__ = "1.0.0"
