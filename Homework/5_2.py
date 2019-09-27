import pandas as pd
import discretize
import numpy as np
import matplotlib.pyplot as plt
imp_var = __import__('5_1')

data = pd.read_csv("dating.csv")
number_bins = [2, 5, 10, 50, 100, 200]
accuracy_training = np.empty(len(number_bins))
accuracy_test = np.empty(len(number_bins))

for b, n_bins in enumerate(number_bins):
    data_binned = discretize.binning_data(data, n_bins)
    test_data = data_binned.sample(frac=0.2, random_state=47)
    training_data = data_binned.drop(test_data.index)

    features = [col for col in training_data.columns if col not in ['decision']]
    classes = training_data['decision'].unique()
    # Learn probabilities
    p_decision, p_attributes = imp_var.nbc(features, training_data, 'decision', classes, 1)

    # Test probabilities
    print(f"Bin size: {n_bins}")
    accuracy_training[b] = imp_var.get_accuracy(training_data, 'decision', classes, features, p_decision, p_attributes)
    print(f"Training Accuracy: {accuracy_training[b]:.2f}")

    accuracy_test[b] = imp_var.get_accuracy(test_data, 'decision', classes, features, p_decision, p_attributes)
    print(f"Testing Accuracy: {accuracy_test[b]:.2f}")

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
