"""Tests for additional metrics and the report generator.

Covers:
  - EffectiveDimensionMetric: PCA-based effective dimension
  - ReportGenerator: JSON/Markdown/CSV output, add_result, summary
"""

import json
import os
import tempfile
import torch

from metrics.effective_dim import EffectiveDimensionMetric
from metrics.base import MetricResult
from metrics.report import ReportGenerator


# ── EffectiveDimensionMetric ──────────────────────────────────────────────────

def test_effective_dim_low_rank():
    """Rank-1 embeddings → very few effective dimensions (≈ 0% utilization)."""
    metric = EffectiveDimensionMetric()
    n, D = 200, 64
    # Rank-1: all samples are multiples of a single direction
    base = torch.randn(1, D)
    scales = torch.randn(n, 1)
    emb = scales @ base  # rank 1

    result = metric.compute(emb)
    # Utilization should be very low (1 effective dim / 64 total)
    assert result.value < 0.1, \
        f"Rank-1 embeddings utilization too high: {result.value:.4f}"
    assert torch.isfinite(torch.tensor(result.value))
    print(f"  [PASS] EffectiveDimensionMetric rank-1 → low utilization ({result.value:.4f})")


def test_effective_dim_full_rank():
    """Full-rank random embeddings → higher effective dimension utilization."""
    metric = EffectiveDimensionMetric()
    torch.manual_seed(42)
    n, D = 500, 32
    # Random embeddings are near-full-rank
    emb = torch.randn(n, D)

    result = metric.compute(emb)
    # Should use a substantial fraction of dimensions
    assert result.value > 0.3, \
        f"Full-rank embeddings utilization too low: {result.value:.4f}"
    assert result.details['total_dimension'] == D
    assert result.details['sample_size'] == n
    print(f"  [PASS] EffectiveDimensionMetric full-rank → utilization={result.value:.4f}")


def test_effective_dim_returns_details():
    """Result details contain expected keys."""
    metric = EffectiveDimensionMetric()
    emb = torch.randn(100, 16)
    result = metric.compute(emb)

    for key in ('total_dimension', 'effective_dimensions', 'utilization_ratio',
                'participation_ratio', 'top_10_variance_ratio', 'sample_size'):
        assert key in result.details, f"Missing key: {key}"
    assert 'dim_95' in result.details['effective_dimensions']
    print(f"  [PASS] EffectiveDimensionMetric details keys present")


def test_effective_dim_sample_size_cap():
    """With more samples than sample_size, downsamples correctly."""
    metric = EffectiveDimensionMetric()
    torch.manual_seed(0)
    n, D = 2000, 16
    emb = torch.randn(n, D)

    result = metric.compute(emb, sample_size=100)
    assert result.details['sample_size'] == 100
    assert torch.isfinite(torch.tensor(result.value))
    print(f"  [PASS] EffectiveDimensionMetric samples capped to 100")


def test_effective_dim_assess_quality():
    """assess_quality returns correct status based on utilization ratio."""
    metric = EffectiveDimensionMetric()
    assert metric.assess_quality(0.8) == 'excellent'
    assert metric.assess_quality(0.6) == 'good'
    assert metric.assess_quality(0.4) == 'acceptable'
    assert metric.assess_quality(0.1) == 'poor'
    print(f"  [PASS] EffectiveDimensionMetric assess_quality thresholds")


def test_effective_dim_variance_thresholds():
    """Custom variance_thresholds produce expected keys in effective_dimensions."""
    metric = EffectiveDimensionMetric()
    emb = torch.randn(200, 32)
    result = metric.compute(emb, variance_thresholds=[0.5, 0.75])

    assert 'dim_50' in result.details['effective_dimensions']
    assert 'dim_75' in result.details['effective_dimensions']
    # dim_50 ≤ dim_75 (need fewer dims for 50% than 75%)
    assert (result.details['effective_dimensions']['dim_50'] <=
            result.details['effective_dimensions']['dim_75'])
    print(f"  [PASS] EffectiveDimensionMetric custom variance thresholds")


# ── ReportGenerator ───────────────────────────────────────────────────────────

def _make_results():
    return {
        'collision': MetricResult('collision', 0.03, [0.02, 0.04], {'n_collisions': 3}, 'excellent'),
        'entropy': MetricResult('entropy', 0.88, [0.9, 0.85], {}, 'good'),
        'utilization': MetricResult('utilization', 0.55, [], {}, 'acceptable'),
    }


def test_report_generator_add_result():
    """add_result and add_results store results correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('test_model', output_dir=tmpdir)
        gen.add_result(MetricResult('collision', 0.05))
        gen.add_results({'entropy': MetricResult('entropy', 0.9)})
        assert 'collision' in gen.results
        assert 'entropy' in gen.results
        print(f"  [PASS] ReportGenerator add_result and add_results")


def test_report_generator_json():
    """generate_json() produces a valid JSON file with expected structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('my_model', output_dir=tmpdir, metadata={'n_clusters': 1024})
        for name, r in _make_results().items():
            gen.add_result(r)

        path = gen.generate_json('test.json')
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)

        assert data['metadata']['model'] == 'my_model'
        assert data['metadata']['n_clusters'] == 1024
        assert 'collision' in data['metrics']
        assert 'entropy' in data['metrics']
        assert data['summary']['total_metrics'] == 3
        print(f"  [PASS] ReportGenerator JSON output valid")


def test_report_generator_markdown():
    """generate_markdown() produces a .md file with model name and table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('test_model_md', output_dir=tmpdir)
        for name, r in _make_results().items():
            gen.add_result(r)

        path = gen.generate_markdown('test.md')
        assert os.path.exists(path)

        content = open(path).read()
        assert 'test_model_md' in content
        assert 'collision' in content.lower() or 'Collision' in content
        assert '| Metric |' in content  # table header
        print(f"  [PASS] ReportGenerator Markdown output valid")


def test_report_generator_csv():
    """generate_csv() produces a CSV with header and data rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('test_model_csv', output_dir=tmpdir)
        for name, r in _make_results().items():
            gen.add_result(r)

        path = gen.generate_csv('test.csv')
        assert os.path.exists(path)

        lines = open(path).readlines()
        assert len(lines) == 4  # header + 3 metrics
        assert 'metric' in lines[0].lower()
        print(f"  [PASS] ReportGenerator CSV output valid ({len(lines)-1} data rows)")


def test_report_generator_generate_all():
    """generate_all() produces json, markdown, and csv files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('all_formats', output_dir=tmpdir)
        gen.add_result(MetricResult('entropy', 0.9))

        paths = gen.generate_all(prefix='report')
        assert set(paths.keys()) == {'json', 'markdown', 'csv'}
        for fmt, path in paths.items():
            assert os.path.exists(path), f"{fmt} file not created: {path}"
        print(f"  [PASS] ReportGenerator generate_all() produces all formats")


def test_report_generator_summary_counts():
    """Summary counts match actual result statuses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('summary_test', output_dir=tmpdir)
        gen.add_result(MetricResult('a', 1.0, status='excellent'))
        gen.add_result(MetricResult('b', 0.5, status='good'))
        gen.add_result(MetricResult('c', 0.2, status='poor'))
        gen.add_result(MetricResult('d', 0.3, status='poor'))

        path = gen.generate_json('s.json')
        data = json.load(open(path))
        summary = data['summary']

        assert summary['excellent_count'] == 1
        assert summary['good_count'] == 1
        assert summary['poor_count'] == 2
        assert summary['total_metrics'] == 4
        print(f"  [PASS] ReportGenerator summary status counts correct")


def test_report_generator_empty():
    """Empty result set produces valid (but empty) reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = ReportGenerator('empty_model', output_dir=tmpdir)
        paths = gen.generate_all(prefix='empty')
        for fmt, path in paths.items():
            assert os.path.exists(path), f"Empty {fmt} file not created"
        print(f"  [PASS] ReportGenerator handles empty results")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("EffectiveDimension + ReportGenerator Tests")
    print("=" * 50)

    print("\n1. EffectiveDimensionMetric")
    test_effective_dim_low_rank()
    test_effective_dim_full_rank()
    test_effective_dim_returns_details()
    test_effective_dim_sample_size_cap()
    test_effective_dim_assess_quality()
    test_effective_dim_variance_thresholds()

    print("\n2. ReportGenerator")
    test_report_generator_add_result()
    test_report_generator_json()
    test_report_generator_markdown()
    test_report_generator_csv()
    test_report_generator_generate_all()
    test_report_generator_summary_counts()
    test_report_generator_empty()

    print("\n" + "=" * 50)
    print("All tests passed!")
