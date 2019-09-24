import numpy as np
import pandas as pd
import sys

data = pd.read_csv(sys.argv[1])
data.head()

# Removing quotes
quoted_columns = ['race', 'race_o', 'field']
count_of_changed_cells = 0
for col in quoted_columns:
    count_of_changed_cells += len(data[data[col].str.startswith("'")])
    data[col] = [c[1:-1] if c.startswith("'") else c for c in data[col]]
print(f"Quotes removed from {count_of_changed_cells} cells.")

# Lowercase
count_of_changed_cells = len(data[~data['field'].str.islower()])
data['field'] = [c.lower() if ~c.islower() else c for c in data['field']]
print(f"Standardized {count_of_changed_cells} cells to lower case.")

# Encoding Labels
label_columns = ['gender', 'race', 'race_o', 'field']
le_encoders = {}
for col in label_columns:
    keys = data[col].sort_values().unique()
    values = np.arange(len(keys))
    le_encoders[col] = dict(zip(keys, values))
    data[col] = [le_encoders[col][c] for c in data[col]]

print(f"Value assigned for male in column gender: {le_encoders['gender']['male']}.")
print(f"Value assigned for European/Caucasian-American in column race: {le_encoders['race']['European/Caucasian-American']}.")
print(f"Value assigned for Latino/Hispanic American in column race o: {le_encoders['race_o']['Latino/Hispanic American']}.")
print(f"Value assigned for law in column field: {le_encoders['field']['law']}.")

data['total'] = data['attractive_important'] + data['sincere_important'] + data['intelligence_important'] \
                + data['funny_important'] + data['ambition_important'] + data['shared_interests_important']
preference_scores_participant = ['attractive_important', 'sincere_important', 'intelligence_important',
                                 'funny_important', 'ambition_important', 'shared_interests_important']
for col in preference_scores_participant:
    data[col] = data[col] / data['total']
    print(f"Mean of {col}: {data[col].mean():.2f}.")

data['total_o'] = data['pref_o_attractive'] + data['pref_o_sincere'] + data['pref_o_intelligence'] + data['pref_o_funny'] \
                  + data['pref_o_ambitious'] + data['pref_o_shared_interests']
preference_scores_partner = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny',
                             "pref_o_ambitious", 'pref_o_shared_interests']
for col in preference_scores_partner:
    data[col] = data[col] / data['total_o']
    print(f"Mean of {col}: {data[col].mean():.2f}.")

data = data.drop(columns=['total', 'total_o'])
data.to_csv(index=False, path_or_buf=sys.argv[2])
