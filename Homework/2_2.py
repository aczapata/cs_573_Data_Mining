import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dating.csv")

rating_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner',
                                   'ambition_partner', 'shared_interests_partner']

fig = plt.figure(figsize=(15, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for ix, col in enumerate(rating_partner_from_participant):
    total_count = data.groupby(col)['decision'].count()
    success_count = data[data['decision'] ==1].groupby(col)['decision'].count()
    success_ratio = (success_count/total_count).fillna(0)
    ax = fig.add_subplot(3, 2, ix+1)
    ax.scatter(success_ratio.index.tolist(), success_ratio.values.tolist())
    ax.set_title(col + ' success rate')
    ax.set_ylabel('success rate')
    ax.set_xlabel('rating')
plt.show()
