import pandas as pd
import numpy as np


def nbc(feature_cols, data, class_col, class_val, t_frac):
    training_sample = data.sample(frac=t_frac, random_state=47)
    prob_decision = training_sample.groupby([class_col])[class_col].count() / len(training_sample)
    prob_attributes = np.empty((len(class_val), len(feature_cols)), dtype=object)

    for ix, col in enumerate(feature_cols):
        prob = (training_sample.groupby([class_col, col])[class_col]
                                .count().unstack(fill_value=0).stack() + 1) / \
                               (training_sample.groupby([class_col])[class_col].count() +
                                len(training_sample[col].sort_values().unique()))

        for i_class in class_val:
            prob_attributes[i_class][ix] = prob[i_class]

    return prob_decision, prob_attributes


def get_accuracy(data, class_col, classes_val, feature_cols, prob_decision, prob_attributes):
    prob = np.ones((len(data), len(classes_val)))

    # Multiply all class probabilities
    for class_i, class_val in enumerate(classes_val):
        prob[:, class_i] *= prob[:, class_i] * prob_decision[class_i]

    # Multiply all feature probabilities
    for ix, col in enumerate(feature_cols):
        for i_class in classes_val:
            prob[:, i_class] *= data[col].map(prob_attributes[i_class][ix])

    data["prediction"] = np.argmax(prob, axis=1)
    correct_classified = data[data[class_col] == data['prediction']][class_col].count()
    return correct_classified/len(data)


training_data = pd.read_csv("trainingSet.csv")
features = [col for col in training_data.columns if col not in ['decision']]
classes = training_data['decision'].unique()
# Learn probabilities
p_decision, p_attributes = nbc(features, training_data, 'decision', classes, 1)
# Test probabilities
accuracy_training = get_accuracy(training_data, 'decision', classes, features, p_decision, p_attributes)
print(f"Training Accuracy: {accuracy_training:.2f}")

test_data = pd.read_csv("testSet.csv")
accuracy_test = get_accuracy(test_data, 'decision', classes, features, p_decision, p_attributes)
print(f"Testing Accuracy: {accuracy_test:.2f}")
