import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 
DATA_PATH = '../../Dataset/data/merged_data_40501.json'
OUT_PREFIX = os.path.splitext(os.path.basename(__file__))[0]

# 
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    records = json.load(f)
df = pd.DataFrame(records)

# 
labels = {0: 'Non Self-fixed', 1: 'Self-fixed'}

#  TXT（）
counts = (df.groupby(['satd_type', df['is_self_fixed'].map(labels)])
            .size()
            .unstack(fill_value=0))
counts.columns.name = None

counts['total'] = counts.sum(axis=1)

txt_file = f"{OUT_PREFIX}.txt"
with open(txt_file, 'w', encoding='utf-8') as f:
    f.write(counts.to_string())
print(f': {txt_file}')

# 
props = counts.div(counts.sum(axis=1), axis=0)

# 
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
colors = [plt.cm.coolwarm(0.2), plt.cm.coolwarm(0.4), plt.cm.coolwarm(0.8)]

# 
ax = props.plot.bar(
    stacked=True,
    color=colors,
    figsize=(8, 6),
    xlabel='SATD Type',
    ylabel='Proportion',
    rot=45
)
plt.setp(ax.get_xticklabels(), ha='right')
plt.tight_layout()

#  PDF
pdf_file = f"{OUT_PREFIX}.pdf"
plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
plt.close()
print(f': {pdf_file}')