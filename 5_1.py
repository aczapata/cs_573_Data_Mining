
import pandas as pd
import time

def nbc(feature_cols, data, class_col, t_frac):
    training_sample = data.sample(frac=t_frac, random_state=47)
    prob_decision = training_sample.groupby([class_col])[class_col].count() / len(training_sample)
    prob_attributes = {}
    for col in feature_cols:
        prob_attributes[col] = (training_sample.groupby([class_col, col])[class_col]
                                .count().unstack(fill_value=0).stack() + 1) / \
                               (training_sample.groupby([class_col])[class_col].count() +
                                len(training_sample[col].sort_values().unique()))
    return prob_decision, prob_attributes


def get_class(d, classes_val, feature_cols, prob_decision, prob_attributes):
    prob_prediction = {}
    for val in classes_val:
        prob_prediction[val] = prob_decision[val]
        for col in feature_cols:
            if d[col] in prob_attributes[col][val]:
                prob_prediction[val] = prob_prediction[val] * prob_attributes[col][val][d[col]]
    return prob_prediction


def get_accuracy(data, classes_val, feature_cols, prob_decision, prob_attributes):
    correct_classified = 0
    for i, x in data.iterrows():
        prob_prediction = get_class(x, classes_val, feature_cols, prob_decision, prob_attributes)
        correct_classified += 1 if max(prob_prediction, key=prob_prediction.get) == x['decision'] else 0
    return correct_classified/len(data)


training_data = pd.read_csv("trainingSet.csv")
features = [col for col in training_data.columns if col not in ['decision']]
classes = training_data['decision'].unique()

start = time.time()
p_decision, p_attributes = nbc(features, training_data, 'decision', 1)
end = time.time()
elapsed_time = end - start
print("Training Model : ", elapsed_time)
start = time.time()
accuracy_training = get_accuracy(training_data, classes, features, p_decision, p_attributes)
print(f"Training Accuracy: {accuracy_training:.2f}")
end = time.time()
elapsed_time = end - start
print("Testing Model - Training Set : ", elapsed_time)

start = time.time()
test_data = pd.read_csv("testSet.csv")
accuracy_test = get_accuracy(test_data, classes, features, p_decision, p_attributes)
print(f"Testing Accuracy: {accuracy_test:.2f}")
end = time.time()
elapsed_time = end - start
print("Testing Model - Test Set : ", elapsed_time)