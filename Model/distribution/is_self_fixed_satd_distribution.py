import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import chisquare

# Read data
MERGED_DATA = '../../Dataset/data/merged_data_40501.json'
with open(MERGED_DATA, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

target_col = "is_self_fixed"
counts = df[target_col].value_counts().sort_index()  # index: 0, 1
non_self_fixed = counts.get(0, 0)
self_fixed = counts.get(1, 0)
total = non_self_fixed + self_fixed

print(f"Total samples: {total}")
print(f"Observed frequency: 0→{non_self_fixed}, 1→{self_fixed}")


# Construct expected frequency (assuming both classes account for 50% each)
expected = [total * 0.5, total * 0.5]

# χ² goodness-of-fit test
chi2_stat, p_value = chisquare(f_obs=[non_self_fixed, self_fixed], f_exp=expected)

# Configure style (use Arial font globally)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.unicode_minus': False
})

colors = [plt.cm.coolwarm(0.8), plt.cm.coolwarm(0.2)]

# Plot
fig, ax = plt.subplots(figsize=(7, 4))  # Adjust canvas ratio, more compact

# Only show text in legend, pie chart itself only shows percentages to avoid duplicate text
patches, texts, _ = ax.pie(
    [self_fixed, non_self_fixed],
    labels=None,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12, 'color': '#333333', 'fontfamily': 'Arial'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'alpha': 0.85}
)


ax.legend(
    patches,
    [f"Self-repaid SATD({self_fixed}, {self_fixed/total*100:.1f}%)",
     f"Non-self-repaid SATD({non_self_fixed}, {non_self_fixed/total*100:.1f}%)"],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),   # Move legend to bottom center
    ncol=2,                       # Two-column layout
    frameon=False,
    fontsize=14,
    prop={'family': 'Arial'}
)
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.12)
# Save
base = os.path.splitext(os.path.basename(__file__))[0]
# plt.savefig(f'{base}.pdf', bbox_inches='tight')
# plt.close()

# Remove tight_layout (it will leave whitespace around pie chart)
base = os.path.splitext(os.path.basename(__file__))[0]
plt.savefig(f"{base}.pdf", bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()

with open(f'{base}.txt', 'w', encoding='utf-8') as f:
    f.write("Chi-square goodness-of-fit test result:\n\n")
    f.write(f"χ²: {chi2_stat:.6f}\n")
    f.write(f"p_value: {p_value:.3e}\n")
    f.write(f"is_significant: {p_value < 0.05} significant difference\n")