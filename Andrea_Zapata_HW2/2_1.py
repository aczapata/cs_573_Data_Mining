import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dating.csv")
data_female = data[data['gender'] == 0]
data_male = data[data['gender'] == 1]
preference_scores_participant = ['attractive_important', 'sincere_important', 'intelligence_important',
                                 'funny_important', 'ambition_important', 'shared_interests_important']
male_means = []
female_means = []
for col in preference_scores_participant:
    male_means.append(data_male[col].mean())
    female_means.append(data_female[col].mean())

labels = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'shared interests']
x = np.arange(len(labels))
width = 0.45

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, male_means, width, label='Men')
ax.bar(x + width/2, female_means, width, label='Women')

ax.set_ylabel('Scores')
ax.set_title('Scores by Gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()

