import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


with open('../../Dataset/data/merged_data_40501.json', encoding='utf-8') as f:
    df = pd.DataFrame(json.load(f)).replace({'is_self_fixed': {1: 'Self-fixed', 0: 'Non Self-fixed'}})


type_counts = df[df['is_self_fixed'] == 'Non Self-fixed']['satd_type'].value_counts()


plt.rcParams.update({'font.family': 'Times New Roman', 'axes.unicode_minus': False})
colors = sns.color_palette("Set3", len(type_counts))


fig, ax = plt.subplots(figsize=(8, 6))
patches, _, _ = ax.pie(
    type_counts,
    colors=colors,
    autopct='%1.1f%%',
    startangle=150,
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.5, 'alpha': 0.85}
)

ax.axis('equal')  
total = type_counts.sum()
legend_labels = [
    f"{lbl} ({cnt}, {cnt / total * 100:.1f}%)"
    for lbl, cnt in zip(type_counts.index, type_counts)
]
ax.legend(patches, legend_labels,
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          prop={'size': 10})

base = os.path.splitext(os.path.basename(__file__))[0]
plt.savefig(f'{base}.pdf', bbox_inches='tight')
plt.close()