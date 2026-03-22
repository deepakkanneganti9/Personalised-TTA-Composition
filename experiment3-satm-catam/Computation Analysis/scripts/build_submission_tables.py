from pathlib import Path
import csv, json
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / 'generated_tables'
GENERATED.mkdir(parents=True, exist_ok=True)

METHODS = {
    'Semantic': {
        'results': ROOT / 'methods' / 'semantic' / 'outputs' / 'semantic_substitution_results.csv',
        'meta': ROOT / 'methods' / 'semantic' / 'outputs' / 'semantic_substitution_metadata.json',
    },
    'Context': {
        'results': ROOT / 'methods' / 'context' / 'outputs' / 'contextual_substitution_results.csv',
        'meta': ROOT / 'methods' / 'context' / 'outputs' / 'contextual_substitution_metadata.json',
    },
    'MLaaS': {
        'results': ROOT / 'methods' / 'mlaas' / 'outputs' / 'mlaas_substitution_results.csv',
        'meta': ROOT / 'methods' / 'mlaas' / 'outputs' / 'mlaas_substitution_metadata.json',
    },
    'TTA': {
        'results': ROOT / 'methods' / 'tta' / 'outputs' / 'tta_substitution_results.csv',
        'meta': ROOT / 'methods' / 'tta' / 'outputs' / 'tta_substitution_metadata.json',
    },
}

FAIR_SUMMARY = ROOT / 'outputs' / 'fair_mixed' / '20clean_80corrupt_overall_summary.csv'
FAIR_CASES = ROOT / 'outputs' / 'fair_mixed' / '20clean_80corrupt_case_results.csv'
OFFLINE = ROOT / 'outputs' / 'timing' / 'offline_runtime_summary.json'


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# final accuracy table from fair summary
fair_rows = read_csv(FAIR_SUMMARY)
accuracy_rows = []
for row in fair_rows:
    accuracy_rows.append({
        'Method': row['Method'],
        'Healthy': round(to_float(row['Healthy']), 4),
        'Degraded': round(to_float(row['Degraded']), 4),
        'After': round(to_float(row['After Substitution']), 4),
        'Recovery Ratio': round(to_float(row['Recovery Ratio']), 4),
        'Requirement Satisfaction': round(to_float(row['Requirement Satisfaction']), 4),
        'Success': round(to_float(row['Substitution Success']), 4),
        'False No Improvement': round(to_float(row['False Accept: No Improvement']), 4),
        'False Below Requirement': round(to_float(row['False Accept: Below Requirement']), 4),
    })
write_csv(GENERATED / 'final_accuracy_table.csv', list(accuracy_rows[0].keys()), accuracy_rows)

# timing/similarity table
sim_rows = []
for method, info in METHODS.items():
    meta = read_json(info['meta'])
    sim_rows.append({
        'Method': method,
        'Avg Similarity': round(to_float(meta.get('avg_similarity_ratio', 0.0)), 2),
        'Avg Candidate Pool': round(to_float(meta.get('avg_candidate_pool_size', 1.0)), 2),
        'Avg Candidates Scanned': round(to_float(meta.get('avg_candidate_scan_count', 1.0)), 2),
        'Avg Case Time (s)': round(to_float(meta.get('avg_case_time_seconds', 0.0)), 4),
        'Avg Selection (s)': round(to_float(meta.get('avg_selection_time_seconds', 0.0)), 4),
        'Avg Aggregation (s)': round(to_float(meta.get('avg_aggregation_time_seconds', 0.0)), 4),
        'Avg Evaluation (s)': round(to_float(meta.get('avg_evaluation_time_seconds', 0.0)), 4),
        'Runtime Total (s)': round(to_float(meta.get('runtime_seconds', 0.0)), 4),
        'Runtime Per Case (s)': round(to_float(meta.get('runtime_seconds', 0.0)) / max(int(meta.get('num_selected_compositions', 1)), 1), 4),
    })
write_csv(GENERATED / 'timing_similarity_table.csv', list(sim_rows[0].keys()), sim_rows)

# by corruption from fair summary (translate only in this bundle)
by_rows = []
for row in fair_rows:
    by_rows.append({
        'Method': row['Method'],
        'Translate After': round(to_float(row['After Substitution']), 4),
        'Translate RR': round(to_float(row['Recovery Ratio']), 4),
    })
write_csv(GENERATED / 'by_corruption_table.csv', list(by_rows[0].keys()), by_rows)

# overall computation time table
offline = read_json(OFFLINE)
prep = to_float(offline['offline_total_runtime_seconds'])
overall_rows = []
per_case_rows = []
for method, info in METHODS.items():
    meta = read_json(info['meta'])
    runtime = to_float(meta.get('runtime_seconds', 0.0))
    overall = runtime if method == 'TTA' else prep + runtime
    overall_rows.append({
        'Method': method,
        'Preparation Time (s)': round(0.0 if method == 'TTA' else prep, 4),
        'Method Runtime (s)': round(runtime, 4),
        'Overall Computation Time (s)': round(overall, 4),
        'Overall Time (min)': round(overall / 60.0, 2),
    })
    n = max(int(meta.get('num_selected_compositions', 1)), 1)
    per_case_rows.append({
        'Method': method,
        'Overall Time Per Case (s)': round(overall / n, 4),
    })
write_csv(GENERATED / 'overall_computation_time_table.csv', list(overall_rows[0].keys()), overall_rows)
write_csv(GENERATED / 'overall_time_per_case_table.csv', list(per_case_rows[0].keys()), per_case_rows)

# composition-size timing from per-method results using fixed first-5 subset where available
size_summary_rows = []
all_in_one_rows = []
for method, info in METHODS.items():
    rows = [r for r in read_csv(info['results']) if r.get('corruption_category') == 'translate']
    groups = defaultdict(list)
    for row in rows:
        name = row.get('composition_name') or row.get('original_composition_name') or ''
        size = name.count('_') + 1 if name else 0
        groups[size].append(row)
    total_ns = 0
    total_cases = 0
    cases_used_tokens = []
    coverage = []
    per_method_breakdown = []
    for size in sorted(groups):
        grp = sorted(groups[size], key=lambda x: (x.get('composition_name') or x.get('original_composition_name') or ''))
        take = grp[:5]
        vals = [to_float(x.get('case_time_seconds', 0.0)) for x in take]
        total = sum(vals)
        avg = total / len(vals)
        total_ns_size = int(round(total * 1e9))
        avg_ns_size = int(round(avg * 1e9))
        total_ns += total_ns_size
        total_cases += len(vals)
        coverage.append(str(size))
        cases_used_tokens.append(str(len(vals)))
        per_method_breakdown.append({
            'Composition size': size,
            'Cases used': len(vals),
            'Total time (ns)': total_ns_size,
            'Avg. time/case (ns)': avg_ns_size,
        })
        all_in_one_rows.append({
            'table': 'composition_size_breakdown',
            'method': method,
            'composition_size': size,
            'cases_used': len(vals),
            'total_time_ns': total_ns_size,
            'avg_time_per_case_ns': avg_ns_size,
        })
    size_summary_rows.append({
        'Method': method,
        'Composition size coverage': ','.join(coverage),
        'Cases used': ','.join(cases_used_tokens),
        'Total time (ns)': total_ns,
        'Avg. time/case (ns)': int(round(total_ns / max(total_cases, 1))),
    })
    write_csv(GENERATED / f'{method.lower()}_composition_size_table.csv', list(per_method_breakdown[0].keys()), per_method_breakdown)
write_csv(GENERATED / 'composition_size_time_table.csv', list(size_summary_rows[0].keys()), size_summary_rows)

# one single all-results CSV
all_results = []
for r in accuracy_rows:
    row = {'table': 'final_accuracy_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
for r in sim_rows:
    row = {'table': 'timing_similarity_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
for r in by_rows:
    row = {'table': 'by_corruption_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
for r in overall_rows:
    row = {'table': 'overall_computation_time_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
for r in per_case_rows:
    row = {'table': 'overall_time_per_case_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
for r in size_summary_rows:
    row = {'table': 'composition_size_time_table', 'method': r['Method']}
    row.update({k: r[k] for k in r if k != 'Method'})
    all_results.append(row)
all_results.extend(all_in_one_rows)
# union of keys
keys = []
for row in all_results:
    for k in row.keys():
        if k not in keys:
            keys.append(k)
write_csv(GENERATED / 'final_submission_results.csv', keys, all_results)

with open(GENERATED / 'submission_tables_manifest.json', 'w') as f:
    json.dump({'tables': sorted([p.name for p in GENERATED.glob('*.csv')])}, f, indent=2)

print('wrote', GENERATED / 'final_submission_results.csv')
