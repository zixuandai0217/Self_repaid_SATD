import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 1. Read data
MERGED_DATA = '../../Dataset/data/merged_data_40501_updated.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# 2. Split into two groups and construct event indicators
sf  = df[df['is_self_fixed'] == 1]['satd_survival_days']
nsf = df[df['is_self_fixed'] == 0]['satd_survival_days']
events_sf  = pd.Series(1, index=sf.index)   # All considered as "event occurred"
events_nsf = pd.Series(1, index=nsf.index)

base = os.path.splitext(os.path.basename(__file__))[0]

# 4. Kaplan-Meier fitting
kmf_sf  = KaplanMeierFitter(label='Self-repaid')
kmf_nsf = KaplanMeierFitter(label='Non-self-repaid')
kmf_sf.fit(sf,  events_sf)
kmf_nsf.fit(nsf, events_nsf)

# Plot and save PDF (with 95% confidence intervals)
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(10, 8))  # Increase figure size to accommodate confidence intervals

# Plot self-repaid group survival curve (with confidence intervals)
ax = kmf_sf.plot_survival_function(
    color=plt.cm.coolwarm(0.8),
    ci_show=True,
    ci_alpha=0.15,
    linewidth=2.5
)

# Plot non-self-repaid group survival curve (with confidence intervals)
kmf_nsf.plot_survival_function(
    ax=ax,
    color=plt.cm.coolwarm(0.2),
    ci_show=True,
    ci_alpha=0.15,
    linestyle='--',
    linewidth=2.5
)

# Set chart properties
plt.xlabel('Survival days', fontsize=18)
plt.ylabel('Survival probability', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.title('Kaplan-Meier Survival Curves with 95% Confidence Intervals', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=20, loc='upper right')
plt.tight_layout()

# Save PDF
out_pdf = f"{base}_with_CI.pdf"
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved survival curve figure with CI to: {out_pdf}")

# 5. Log-Rank test and save results
results = logrank_test(
    sf, nsf,
    event_observed_A=events_sf,
    event_observed_B=events_nsf,
    alpha=0.05
)

out_txt = f'{base}.txt'
with open(out_txt, 'w', encoding='utf-8') as f:
    f.write("Log-Rank Test result:\n\n")
    f.write(f"Test statistic: {results.test_statistic:.4f}\n")
    f.write(f"p-value:         {results.p_value:.4e}\n")
    f.write("Significant difference: " + ("Yes" if results.p_value < 0.05 else "No") + "\n")
print(f"Saved log-rank test result to: {out_txt}")

def summarize(series, name):
    return {
        'Group': name,
        'Count':  int(series.count()),
        'Mean':   float(series.mean()),
        'Median': float(series.median()),
        'StdDev': float(series.std()),
        'Min':    float(series.min()),
        'Max':    float(series.max())
    }

stats = [
    summarize(sf,  'Self-repaid'),
    summarize(nsf, 'Non-self-repaid')
]
summary_df = pd.DataFrame(stats)
# Save to json
summary_df.to_json(f'{base}.json', orient='records', force_ascii=False, indent=2)
print(f"Saved summary stats to:{base}.json")
