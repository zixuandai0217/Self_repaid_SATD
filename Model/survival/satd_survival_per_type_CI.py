import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np  # Add missing numpy import

# 1. Read data
MERGED_DATA = '../../Dataset/data/merged_data_40501_updated.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)

df = pd.DataFrame(records)
# Keep required columns
df = df[['satd_type', 'satd_survival_days', 'is_self_fixed']]

# Script base name for output naming
base = os.path.splitext(os.path.basename(__file__))[0]

# 2. Statistical analysis function
def summarize(series, name):
    return {
        'satd_type': name,
        'Count':   int(series.count()),
        'Mean':    float(series.mean()),
        'Median':  float(series.median()),
        'StdDev':  float(series.std()),
        'Min':     float(series.min()),
        'Max':     float(series.max())
    }

# 3. Generate three statistical JSON files:
#   - All (original)
#   - Each type Self-Fixed
#   - Each type Non-Self-Fixed
stats_all = []
stats_sf  = []
stats_nsf = []
for t, group in df.groupby('satd_type'):
    # All
    stats_all.append(summarize(group['satd_survival_days'], t))
    # Self-repaid
    sf_series = group[group['is_self_fixed'] == 1]['satd_survival_days']
    stats_sf.append(summarize(sf_series, t))
    # Non-self-repaid
    nsf_series = group[group['is_self_fixed'] == 0]['satd_survival_days']
    stats_nsf.append(summarize(nsf_series, t))

# Output file names
out_all   = f'{base}.json'
out_sf    = f'{base}_self_fixed.json'
out_nsf   = f'{base}_non_self_fixed.json'

# Write JSON
with open(out_all, 'w', encoding='utf-8') as f:
    json.dump(stats_all, f, ensure_ascii=False, indent=2)
with open(out_sf, 'w', encoding='utf-8') as f:
    json.dump(stats_sf, f, ensure_ascii=False, indent=2)
with open(out_nsf, 'w', encoding='utf-8') as f:
    json.dump(stats_nsf, f, ensure_ascii=False, indent=2)

print(f"Saved all stats to: {out_all}")
print(f"Saved self-fixed stats to: {out_sf}")
print(f"Saved non-self-fixed stats to: {out_nsf}")

# 3. Plot combined survival curves (with confidence intervals)
kmf = KaplanMeierFitter()
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(10, 8))  # Increase canvas size to accommodate more information
ax = plt.gca()
handles_sf, labels_sf = [], []
handles_nsf, labels_nsf = [], []

self_color = plt.cm.coolwarm(0.8)
non_self_color = plt.cm.coolwarm(0.2)

for i, (satd_type, group) in enumerate(df.groupby('satd_type')):
    # Self-repaid curve (solid line)
    sf = group[group['is_self_fixed'] == 1]['satd_survival_days']
    events_sf = pd.Series(1, index=sf.index)
    
    kmf.fit(sf, events_sf, label=f"{satd_type} (Self-repaid)")
    ax = kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.15,
        color=self_color,
        linewidth=2.5
    )
    
    # Get the most recently added line (survival curve)
    h_sf = ax.get_lines()[-1]
    handles_sf.append(h_sf)
    labels_sf.append(f"{satd_type} (Self-repaid)")
    
    # Non-self-repaid curve (dashed line)
    nsf = group[group['is_self_fixed'] == 0]['satd_survival_days']
    events_nsf = pd.Series(1, index=nsf.index)
    
    kmf.fit(nsf, events_nsf, label=f"{satd_type} (Non-self-repaid)")
    ax = kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.15,
        color=non_self_color,
        linestyle='--',
        linewidth=2.5
    )
    
    # Get the most recently added line (survival curve)
    h_nsf = ax.get_lines()[-1]
    handles_nsf.append(h_nsf)
    labels_nsf.append(f"{satd_type} (Non-self-repaid)")

# Merge legend items
all_handles = handles_sf + handles_nsf
all_labels = labels_sf + labels_nsf

# Set chart properties
ax.set_xlabel('Survival days', fontsize=18)
ax.set_ylabel('Survival probability', fontsize=18)
ax.set_title('Kaplan-Meier survival curves with 95% confidence intervals', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# （）
ax.legend(all_handles, all_labels, loc='upper right', 
          fontsize=11, ncol=2, columnspacing=1.0, handlelength=2.0)

plt.tight_layout()

# Save combined plot
out_pdf = f"{base}_with_CI.pdf"
plt.savefig(out_pdf, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved combined survival plot with CI to: {out_pdf}")

# 4. Plot individual survival curves for each type in plots/ directory (with confidence intervals)
plots_dir = 'plots_with_CI'
os.makedirs(plots_dir, exist_ok=True)

for i, (satd_type, group) in enumerate(df.groupby('satd_type')):
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Self-repaid curve
    sf = group[group['is_self_fixed'] == 1]['satd_survival_days']
    events_sf = pd.Series(1, index=sf.index)
    
    kmf.fit(sf, events_sf, label='Self-repaid')
    # Add confidence interval band
    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.15,
        color=self_color,
        linewidth=2.5
    )
    
    # Non-self-repaid curve
    nsf = group[group['is_self_fixed'] == 0]['satd_survival_days']
    events_nsf = pd.Series(1, index=nsf.index)
    
    kmf.fit(nsf, events_nsf, label='Non-self-repaid')
    # Add confidence interval band
    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        ci_alpha=0.15,
        color=non_self_color,
        linestyle='--',
        linewidth=2.5
    )
    
    # Set chart properties
    ax.set_xlabel('Survival days', fontsize=18)
    ax.set_ylabel('Survival probability', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_title(f'Survival Analysis: {satd_type}', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=20)
    
    plt.tight_layout()
    
    # Save image
    safe_name = satd_type.replace(' ', '_')
    plt.savefig(os.path.join(plots_dir, f"{safe_name}_with_CI.pdf"), 
               bbox_inches='tight', dpi=300)
    plt.close()

print(f"Saved individual survival plots with CI to: {plots_dir}/")
