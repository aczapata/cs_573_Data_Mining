import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
imp_var = __import__('5_1')

F = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
training_data = pd.read_csv("trainingSet.csv")
test_data = pd.read_csv("testSet.csv")
features = [col for col in training_data.columns if col not in ['decision']]
classes = training_data['decision'].unique()

accuracy_training = np.empty(len(F))
accuracy_test = np.empty(len(F))

for ix, t in enumerate(F):

    # Learn probabilities
    p_decision, p_attributes = imp_var.nbc(features, training_data, 'decision', classes, t)

    # Test probabilities
    accuracy_training[ix] = imp_var.get_accuracy(training_data, 'decision', classes, features, p_decision, p_attributes)
    accuracy_test[ix] = imp_var.get_accuracy(test_data, 'decision', classes, features, p_decision, p_attributes)

x = np.arange(len(F))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, accuracy_training, label='Training Data')
ax.plot(x, accuracy_test, label='Test Data')

ax.set_ylabel('Accuracy')
ax.set_ylabel('Fraction of training data')
ax.set_title('Accuracy by fraction of training data')
ax.set_xticks(x)
ax.set_xticklabels(F)
ax.legend()
plt.show()