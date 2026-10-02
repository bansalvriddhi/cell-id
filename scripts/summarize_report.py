#!/usr/bin/env python3
"""Extract the report's numerical claims from completed, verified result files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/pilot/main'


def main():
    data = pd.read_csv(RUN / 'per_context.csv')
    raw = pd.read_csv(RUN / 'metrics.csv')
    selections = pd.read_csv(RUN / 'selections.csv')
    summary = json.loads((RUN / 'summary.json').read_text())
    primary = {}
    for policy in ['unadapted', 'random', 'trace-uncertainty', 'D-optimal', 'V-optimal']:
        a = data[data.policy.eq(policy)]
        five = a[a.budget.eq(5)]
        twenty = a[a.budget.eq(20)]
        primary[policy] = {
            'normalized_mse_k5': float(five.normalized_mse.mean()),
            'improvement_over_unadapted_k5_percent': float(100*(1-five.normalized_mse.mean())),
            'pearson_k5': float(five.pearson_mean.mean()),
            'normalized_mse_k20': float(twenty.normalized_mse.mean()),
            'normalized_auc': next(v['normalized_auc']['mean'] for v in summary['auc'] if v['policy']==policy)
        }
    paired = next(x for x in summary['paired_comparisons'] if x['policy']=='V-optimal' and x['reference']=='random' and x['budget']==5)['gain_reference_minus_policy']
    same_sets = []
    for context in sorted(data.context.unique()):
        s = selections[selections.context.eq(context) & selections.step.le(5)]
        panels = [set(s.loc[s.policy.eq(p), 'perturbation']) for p in ['trace-uncertainty','D-optimal','V-optimal']]
        assert all(len(v)==5 for v in panels)
        same_sets.append(panels[0]==panels[1]==panels[2])
    models = [json.loads(p.read_text()) for p in sorted((RUN/'models').glob('*.json'))]
    exploratory = json.loads((ROOT/'results/pilot/context_shift/summary.json').read_text())
    exploratory_scores = {str(s['budget']): s['normalized_mse'] for s in exploratory['scores']}
    output = {
        'source_script':'scripts/run_pilot.py',
        'claim_extraction_script':'scripts/summarize_report.py',
        'primary':primary,
        'voptimal_gain_over_random_k5':paired,
        'same_five_probe_sets_all_designs_context_count':int(sum(same_sets)),
        'target_coverage_min':float(raw.target_coverage.min()),
        'target_coverage_max':float(raw.target_coverage.max()),
        'metric_rows':len(raw),
        'model_rank_values':sorted(set(m['rank'] for m in models)),
        'model_noise_multiplier_values':sorted(set(m['noise_multiplier'] for m in models)),
        'exploratory_context_shift':exploratory_scores,
        'exploratory_k5_gain_over_primary_random':exploratory['main_k5_comparisons'][0]['gain_primary_minus_context_shift']
    }
    (ROOT/'report/key_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
