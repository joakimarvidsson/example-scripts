---
name: numerai-performance-tracker
description: Track and analyze Numerai model performance. Use when checking live performance, calculating rolling metrics, identifying performance issues, or generating performance reports. Provides comprehensive monitoring via MCP and local analysis.
---

# Numerai Performance Tracker

## Overview

This skill handles monitoring and analyzing the performance of deployed Numerai models. Use it to track live scores, identify performance degradation, and generate insights for model improvement.

## Trigger Conditions

Use this skill when:
- Checking how models performed in resolved rounds
- User requests "check performance", "how is my model doing", "performance report", or "analyze scores"
- Investigating why a model is underperforming
- Comparing multiple models' live performance
- Generating periodic performance summaries

## Required Tools/Capabilities

- **Numerai MCP Server**: For fetching live scores and round data
- **Python Environment**: With `pandas`, `numpy` for analysis
- **File System Access**: For caching and storing reports (optional)

## MCP Configuration

The Numerai MCP server is configured in `~/.claude/mcp.json`:

```json
{
  "servers": {
    "numerai": {
      "transport": "sse",
      "url": "https://api-tournament.numer.ai/mcp/sse",
      "headers": {
        "Authorization": "Token ${NUMERAI_MCP_AUTH}"
      }
    }
  }
}
```

**Required Environment Variable**: `NUMERAI_MCP_AUTH` must be set to your Numerai API token.

**Python Wrapper**: For programmatic access, use the `NumeraiMCP` class from `numerai_2026_pipeline/numerai_mcp.py`:

```python
from numerai_mcp import NumeraiMCP

client = NumeraiMCP()  # Reads NUMERAI_MCP_AUTH from environment
performance = client.get_model_performance("my_model")
```

## Workflow

### 0) Verify MCP Configuration

Before starting, verify MCP access is properly configured:

```python
import os

# Check environment variable is set
assert os.environ.get("NUMERAI_MCP_AUTH"), (
    "NUMERAI_MCP_AUTH environment variable not set. "
    "Export your Numerai API token: export NUMERAI_MCP_AUTH='your-token'"
)

# Test MCP connection (optional - via Python wrapper)
from numerai_mcp import NumeraiMCP, NumeraiMCPAuthError

try:
    client = NumeraiMCP()
    print("MCP client initialized successfully")
except NumeraiMCPAuthError as e:
    print(f"MCP authentication failed: {e}")
```

### 1) Fetch Model Performance Data (MCP Required)

Query performance metrics for your models. Send GraphQL queries to the MCP endpoint at `https://api-tournament.numer.ai/mcp/sse`:

```graphql
query {
  account {
    models {
      id
      username
      stake
      nmrStaked
      return1d
      return3m
      roundModelPerformances(limit: 100) {
        roundNumber
        correlation
        correlationWithMetamodel
        tc
        fncV3
        apy
        roundResolved
        roundOpenTime
        roundResolveTime
      }
    }
  }
}
```

**Via Python wrapper**:

```python
from numerai_mcp import NumeraiMCP

client = NumeraiMCP()
performance = client.get_model_performance("my_model_name")

# performance contains:
# - corr: Correlation scores by round
# - mmc: MMC scores by round
# - fnc: Feature-neutral correlation scores
# - rank: Tournament rankings
# - payout: Payout history
```

### 2) Extract Key Metrics

Organize performance data for analysis:

```python
import pandas as pd
from datetime import datetime

# Parse MCP response into DataFrame
def parse_performances(model_data):
    """Convert MCP response to analysis-ready DataFrame."""
    records = []
    for perf in model_data["roundModelPerformances"]:
        if perf["roundResolved"]:
            records.append({
                "round": perf["roundNumber"],
                "corr": perf["correlation"],
                "mmc": perf["correlationWithMetamodel"],
                "tc": perf["tc"],
                "fnc": perf["fncV3"],
                "apy": perf["apy"],
                "resolved_at": perf["roundResolveTime"],
            })

    df = pd.DataFrame(records)
    df = df.sort_values("round", ascending=False)
    return df

# Calculate summary statistics
def summarize_performance(df, window=20):
    """Calculate rolling and overall statistics."""
    recent = df.head(window)

    return {
        "total_rounds": len(df),
        "recent_rounds": len(recent),
        "corr_mean": df["corr"].mean(),
        "corr_std": df["corr"].std(),
        "corr_recent_mean": recent["corr"].mean(),
        "mmc_mean": df["mmc"].mean() if "mmc" in df else None,
        "mmc_recent_mean": recent["mmc"].mean() if "mmc" in recent else None,
        "tc_mean": df["tc"].mean() if "tc" in df else None,
        "positive_corr_pct": (df["corr"] > 0).mean() * 100,
        "recent_positive_pct": (recent["corr"] > 0).mean() * 100,
    }
```

### 3) Calculate Rolling Statistics

Compute rolling metrics to identify trends:

```python
import numpy as np

def calculate_rolling_stats(df, windows=[20, 50, 100]):
    """Calculate rolling statistics for different windows."""
    stats = {}

    for window in windows:
        if len(df) >= window:
            recent = df.head(window)
            stats[f"corr_mean_{window}"] = recent["corr"].mean()
            stats[f"corr_std_{window}"] = recent["corr"].std()
            stats[f"sharpe_{window}"] = recent["corr"].mean() / recent["corr"].std() if recent["corr"].std() > 0 else 0
            stats[f"max_drawdown_{window}"] = calculate_drawdown(recent["corr"])

    return stats

def calculate_drawdown(returns):
    """Calculate maximum drawdown from cumulative returns."""
    cumulative = returns.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    return drawdown.min()
```

### 4) Identify Performance Issues

Check for common performance problems:

```python
def check_performance_issues(df, summary):
    """Identify potential performance issues."""
    issues = []

    # Check for negative recent performance
    if summary["corr_recent_mean"] < 0:
        issues.append({
            "severity": "high",
            "issue": "Negative recent correlation",
            "detail": f"Last 20 rounds avg: {summary['corr_recent_mean']:.4f}",
            "action": "Review model for data drift or overfitting"
        })

    # Check for declining performance
    if summary["corr_recent_mean"] < summary["corr_mean"] * 0.5:
        issues.append({
            "severity": "medium",
            "issue": "Declining performance",
            "detail": f"Recent ({summary['corr_recent_mean']:.4f}) vs Overall ({summary['corr_mean']:.4f})",
            "action": "Consider retraining on recent data"
        })

    # Check for high variance
    if summary["corr_std"] > 0.05:
        issues.append({
            "severity": "low",
            "issue": "High variance in predictions",
            "detail": f"Std dev: {summary['corr_std']:.4f}",
            "action": "Consider ensemble or regularization"
        })

    # Check for low hit rate
    if summary["positive_corr_pct"] < 50:
        issues.append({
            "severity": "medium",
            "issue": "Low positive correlation rate",
            "detail": f"Only {summary['positive_corr_pct']:.1f}% rounds positive",
            "action": "Review feature selection and model architecture"
        })

    return issues
```

### 5) Compare Multiple Models

For users with multiple models:

```python
def compare_models(models_data):
    """Compare performance across multiple models."""
    comparison = []

    for model in models_data:
        df = parse_performances(model)
        summary = summarize_performance(df)

        comparison.append({
            "model": model["username"],
            "stake": model["nmrStaked"],
            "rounds": summary["total_rounds"],
            "corr_mean": summary["corr_mean"],
            "corr_recent": summary["corr_recent_mean"],
            "positive_pct": summary["positive_corr_pct"],
            "return_3m": model["return3m"],
        })

    return pd.DataFrame(comparison).sort_values("corr_recent", ascending=False)
```

### 6) Generate Performance Report

Create a comprehensive performance report:

```python
def generate_report(model_name, df, summary, issues):
    """Generate markdown performance report."""
    report = f"""# Performance Report: {model_name}

## Summary
- **Total Rounds**: {summary['total_rounds']}
- **Overall Correlation**: {summary['corr_mean']:.4f} (+/- {summary['corr_std']:.4f})
- **Recent Correlation (20 rounds)**: {summary['corr_recent_mean']:.4f}
- **Positive Round Rate**: {summary['positive_corr_pct']:.1f}%

## Rolling Metrics

| Window | Mean Corr | Std | Sharpe |
|--------|-----------|-----|--------|
| 20 rounds | {summary.get('corr_mean_20', 'N/A')} | {summary.get('corr_std_20', 'N/A')} | {summary.get('sharpe_20', 'N/A')} |
| 50 rounds | {summary.get('corr_mean_50', 'N/A')} | {summary.get('corr_std_50', 'N/A')} | {summary.get('sharpe_50', 'N/A')} |

## Recent Performance (Last 10 Rounds)

| Round | Corr | MMC | TC |
|-------|------|-----|-----|
"""

    for _, row in df.head(10).iterrows():
        report += f"| {row['round']} | {row['corr']:.4f} | {row.get('mmc', 'N/A')} | {row.get('tc', 'N/A')} |\n"

    if issues:
        report += "\n## Issues Identified\n\n"
        for issue in issues:
            report += f"- **[{issue['severity'].upper()}]** {issue['issue']}: {issue['detail']}\n"
            report += f"  - Recommended action: {issue['action']}\n"

    report += f"\n---\n*Generated: {datetime.now().isoformat()}*\n"

    return report
```

### 7) Track Stake and Returns

Monitor staking performance:

```graphql
query {
  account {
    models {
      username
      stake
      nmrStaked
      return1d
      return1w
      return1m
      return3m
      roundModelPerformances(limit: 10) {
        roundNumber
        payout
        selectedStakeValue
      }
    }
  }
}
```

### 8) Set Up Alerts (Optional)

Define alerting thresholds:

```python
ALERT_THRESHOLDS = {
    "corr_min": -0.02,  # Alert if correlation drops below this
    "consecutive_negative": 3,  # Alert after N consecutive negative rounds
    "drawdown_max": -0.10,  # Alert if cumulative drawdown exceeds this
}

def check_alerts(df, thresholds=ALERT_THRESHOLDS):
    """Check if any alert conditions are triggered."""
    alerts = []

    # Check latest correlation
    if df.iloc[0]["corr"] < thresholds["corr_min"]:
        alerts.append(f"Latest correlation ({df.iloc[0]['corr']:.4f}) below threshold")

    # Check consecutive negatives
    consecutive = 0
    for _, row in df.iterrows():
        if row["corr"] < 0:
            consecutive += 1
        else:
            break

    if consecutive >= thresholds["consecutive_negative"]:
        alerts.append(f"{consecutive} consecutive negative rounds")

    return alerts
```

## Success Criteria

The skill is complete when:
- Performance data fetched for all requested models
- Rolling statistics calculated and presented
- Any performance issues identified and explained
- Performance report generated (if requested)
- User understands current model status and any recommended actions

## Metric Definitions

- **Correlation (CORR)**: Correlation between predictions and target
- **MMC**: Meta Model Contribution - orthogonal alpha vs the meta model
- **TC**: True Contribution - feature-neutral correlation
- **FNC**: Feature Neutral Correlation
- **APY**: Annualized percentage yield on stake

## Common Issues

- **Data lag**: Scores are only available after round resolution (~4 weeks)
- **API limits**: Large queries may need pagination
- **Model not found**: Verify model name/UUID is correct

## Next Steps

Based on performance analysis:
1. If underperforming: Use `numerai-experiment-design` to iterate on model
2. If stable and positive: Consider increasing stake
3. If highly variable: Consider ensemble approaches
4. If declining: Check for data version updates or market regime changes
