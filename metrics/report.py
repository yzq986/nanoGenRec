"""
Report Generator

Generates evaluation reports in JSON, Markdown, and CSV formats.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import MetricResult


class ReportGenerator:
    """Generate evaluation reports in multiple formats"""

    def __init__(
        self,
        model_name: str,
        output_dir: str = 'eval_results',
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize report generator

        Args:
            model_name: Name of the evaluated model
            output_dir: Directory for output files
            metadata: Additional metadata to include
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.metadata = metadata or {}
        self.results: Dict[str, MetricResult] = {}

        os.makedirs(output_dir, exist_ok=True)

    def add_result(self, result: MetricResult):
        """Add a metric result"""
        self.results[result.name] = result

    def add_results(self, results: Dict[str, MetricResult]):
        """Add multiple metric results"""
        self.results.update(results)

    def generate_all(self, prefix: str = 'report') -> Dict[str, str]:
        """Generate all report formats

        Returns:
            Dict of format -> filepath
        """
        paths = {}
        paths['json'] = self.generate_json(f'{prefix}.json')
        paths['markdown'] = self.generate_markdown(f'{prefix}.md')
        paths['csv'] = self.generate_csv(f'{prefix}.csv')
        return paths

    def generate_json(self, filename: str = 'report.json') -> str:
        """Generate JSON report

        Returns:
            Path to generated file
        """
        filepath = os.path.join(self.output_dir, filename)

        report = {
            'metadata': {
                'model': self.model_name,
                'timestamp': datetime.now().isoformat(),
                **self.metadata,
            },
            'metrics': {
                name: result.to_dict()
                for name, result in self.results.items()
            },
            'summary': self._generate_summary(),
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"JSON report saved to: {filepath}")
        return filepath

    def generate_markdown(self, filename: str = 'REPORT.md') -> str:
        """Generate Markdown report

        Returns:
            Path to generated file
        """
        filepath = os.path.join(self.output_dir, filename)

        lines = []

        # Header
        lines.append(f'# Evaluation Report: {self.model_name}')
        lines.append('')
        lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append('')

        # Metadata
        if self.metadata:
            lines.append('## Metadata')
            lines.append('')
            for key, value in self.metadata.items():
                lines.append(f'- **{key}**: {value}')
            lines.append('')

        # Summary Table
        lines.append('## Summary')
        lines.append('')
        lines.append('| Metric | Value | Status |')
        lines.append('|--------|-------|--------|')

        for name, result in self.results.items():
            status_emoji = self._status_emoji(result.status)
            value_str = self._format_value(name, result.value)
            lines.append(f'| {self._format_name(name)} | {value_str} | {status_emoji} {result.status.capitalize()} |')

        lines.append('')

        # Detailed Results
        lines.append('## Detailed Results')
        lines.append('')

        for name, result in self.results.items():
            lines.append(f'### {self._format_name(name)}')
            lines.append('')

            # Main value
            lines.append(f'**Value**: {self._format_value(name, result.value)}')
            lines.append(f'**Status**: {self._status_emoji(result.status)} {result.status.capitalize()}')
            lines.append('')

            # Layer values
            if result.layer_values:
                lines.append('**Per-Layer Values**:')
                for i, lv in enumerate(result.layer_values):
                    lines.append(f'- Layer {i+1}: {lv:.4f}')
                lines.append('')

            # Details
            if result.details:
                lines.append('**Details**:')
                lines.append('')
                lines.append('| Key | Value |')
                lines.append('|-----|-------|')
                for key, value in result.details.items():
                    if isinstance(value, list):
                        # 格式化 list
                        if all(isinstance(v, float) for v in value):
                            val_str = ', '.join(f'{v:.4f}' for v in value)
                        else:
                            val_str = ', '.join(str(v) for v in value)
                        lines.append(f'| {key} | [{val_str}] |')
                    elif isinstance(value, float):
                        lines.append(f'| {key} | {value:.4f} |')
                    else:
                        lines.append(f'| {key} | {value} |')
                lines.append('')

        # Interpretation Guide
        lines.append('## Interpretation Guide')
        lines.append('')
        lines.append('| Metric | Ideal | Interpretation |')
        lines.append('|--------|-------|----------------|')
        lines.append('| Reconstruction Loss | < 0.05 | Lower = better quantization precision |')
        lines.append('| Codebook Utilization | 100% | Higher = better capacity usage |')
        lines.append('| Entropy (normalized) | > 95% | Higher = more uniform distribution |')
        lines.append('| Cosine Similarity Std | > 0.25 | Higher = better discrimination |')
        lines.append('| Effective Dim Ratio | > 70% | Higher = better info utilization |')
        lines.append('| Collision Rate | < 1% | Lower = more unique IDs |')
        lines.append('| Cluster Gini | < 0.15 | Lower = more balanced clusters |')
        lines.append('')

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to: {filepath}")
        return filepath

    def generate_csv(self, filename: str = 'report.csv') -> str:
        """Generate CSV report for easy comparison

        Returns:
            Path to generated file
        """
        filepath = os.path.join(self.output_dir, filename)

        import csv

        rows = []
        # Header row
        header = ['metric', 'value', 'status']
        # Add layer columns if any metric has layer values
        max_layers = max((len(r.layer_values) for r in self.results.values()), default=0)
        for i in range(max_layers):
            header.append(f'layer_{i+1}')

        rows.append(header)

        # Data rows
        for name, result in self.results.items():
            row = [name, result.value, result.status]
            for i in range(max_layers):
                if i < len(result.layer_values):
                    row.append(result.layer_values[i])
                else:
                    row.append('')
            rows.append(row)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"CSV report saved to: {filepath}")
        return filepath

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        statuses = [r.status for r in self.results.values()]
        return {
            'total_metrics': len(self.results),
            'excellent_count': statuses.count('excellent'),
            'good_count': statuses.count('good'),
            'acceptable_count': statuses.count('acceptable'),
            'poor_count': statuses.count('poor'),
            'unknown_count': statuses.count('unknown'),
        }

    def _format_name(self, name: str) -> str:
        """Format metric name for display"""
        return name.replace('_', ' ').title()

    def _format_value(self, name: str, value: float) -> str:
        """Format value based on metric type"""
        if 'utilization' in name or 'entropy' in name or 'dimension' in name:
            return f'{value:.1%}'
        elif 'collision' in name or 'gini' in name:
            return f'{value:.2%}'
        else:
            return f'{value:.4f}'

    def _status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        return {
            'excellent': '🟢',
            'good': '🟡',
            'acceptable': '🟠',
            'poor': '🔴',
            'unknown': '⚪',
        }.get(status, '⚪')
