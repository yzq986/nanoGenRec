"""NTP (Next Token Prediction) package for Semantic ID prediction."""

from .baseline import NTPProbe, SIDSequenceDataset
from .model import NTPModel

__all__ = ['NTPProbe', 'NTPModel', 'SIDSequenceDataset']


def __getattr__(name):
    if name == 'SemanticIDPredictionMetric':
        from .eval import SemanticIDPredictionMetric
        return SemanticIDPredictionMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
