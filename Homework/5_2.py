import pandas as pd
from discretize import binning_data
import numpy as np
import matplotlib.pyplot as plt
imp_var = __import__('5_1')

range_min = {}
range_max = {}

# Initialize ranges from pdf file
range_min['age'], range_max['age'] = 18, 58
range_min['age_o'], range_max['age_o'] = 18, 58
range_min['importance_same_race'], range_max['importance_same_race'] = 0, 10
range_min['importance_same_religion'], range_max['importance_same_religion'] = 0, 10

range_min['pref_o_attractive'], range_max['pref_o_attractive'] = 0, 1
range_min['pref_o_sincere'], range_max['pref_o_sincere'] = 0, 1
range_min['pref_o_intelligence'], range_max['pref_o_intelligence'] = 0, 1
range_min['pref_o_funny'], range_max['pref_o_funny'] = 0, 1
range_min['pref_o_ambitious'], range_max['pref_o_ambitious'] = 0, 1
range_min['pref_o_shared_interests'], range_max['pref_o_shared_interests'] = 0, 1

range_min['attractive_important'], range_max['attractive_important'] = 0, 1
range_min['sincere_important'], range_max['sincere_important'] = 0, 1
range_min['intelligence_important'], range_max['intelligence_important'] = 0, 1
range_min['funny_important'], range_max['funny_important'] = 0, 1
range_min['ambition_important'], range_max['ambition_important'] = 0, 1
range_min['shared_interests_important'], range_max['shared_interests_important'] = 0, 1

range_min['attractive'], range_max['attractive'] = 0, 10
range_min['sincere'], range_max['sincere'] = 0, 10
range_min['intelligence'], range_max['intelligence'] = 0, 10
range_min['funny'], range_max['funny'] = 0, 10
range_min['ambition'], range_max['ambition'] = 0, 10

range_min['attractive_partner'], range_max['attractive_partner'] = 0, 10
range_min['sincere_partner'], range_max['sincere_partner'] = 0, 10
range_min['intelligence_parter'], range_max['intelligence_parter'] = 0, 10
range_min['funny_partner'], range_max['funny_partner'] = 0, 10
range_min['ambition_partner'], range_max['ambition_partner'] = 0, 10
range_min['shared_interests_partner'], range_max['shared_interests_partner'] = 0, 10

range_min['sports'], range_max['sports'] = 0, 10
range_min['tvsports'], range_max['tvsports'] = 0, 10
range_min['exercise'], range_max['exercise'] = 0, 10
range_min['dining'], range_max['dining'] = 0, 10
range_min['museums'], range_max['museums'] = 0, 10
range_min['art'], range_max['art'] = 0, 10
range_min['hiking'], range_max['hiking'] = 0, 10
range_min['gaming'], range_max['gaming'] = 0, 10
range_min['clubbing'], range_max['clubbing'] = 0, 10
range_min['reading'], range_max['reading'] = 0, 10
range_min['tv'], range_max['tv'] = 0, 10
range_min['theater'], range_max['theater'] = 0, 10
range_min['movies'], range_max['movies'] = 0, 10
range_min['concerts'], range_max['concerts'] = 0, 10
range_min['music'], range_max['music'] = 0, 10
range_min['shopping'], range_max['shopping'] = 0, 10
range_min['yoga'], range_max['yoga'] = 0, 10

range_min['interests_correlate'], range_max['interests_correlate'] = -1, 1
range_min['expected_happy_with_sd_people'], range_max['expected_happy_with_sd_people'] = 0, 10
range_min['like'], range_max['like'] = 0, 10

data = pd.read_csv("dating.csv")
number_bins = [2, 5, 10, 50, 100, 200]
accuracy_training = np.empty(len(number_bins))
accuracy_test = np.empty(len(number_bins))

for b, n_bins in enumerate(number_bins):
    data_binned = binning_data(data, range_min, range_max, n_bins)
    test_data = data_binned.sample(frac=0.2, random_state=47)
    training_data = data_binned.drop(test_data.index)

    features = [col for col in training_data.columns if col not in ['decision']]
    classes = training_data['decision'].unique()
    # Learn probabilities
    p_decision, p_attributes = imp_var.nbc(features, training_data, 'decision', classes, 1)

    # Test probabilities
    print(f"Bin size: {n_bins}")
    accuracy_training[b] = imp_var.get_accuracy(training_data, 'decision', classes, features, p_decision, p_attributes)
    print(f"Training Accuracy: {accuracy_training[b]}")

    accuracy_test[b] = imp_var.get_accuracy(test_data, 'decision', classes, features, p_decision, p_attributes)
    print(f"Testing Accuracy: {accuracy_test[b]}")

x = np.arange(len(number_bins))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, accuracy_training, label='Training Data')
ax.plot(x, accuracy_test, label='Test Data')

ax.set_ylabel('Accuracy')
ax.set_ylabel('Number of bins')
ax.set_title('Accuracy by # of bins')
ax.set_xticks(x)
ax.set_xticklabels(number_bins)
ax.legend()
plt.show()
