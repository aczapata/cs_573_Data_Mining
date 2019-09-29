import sys
import pandas as pd

data = pd.read_csv(sys.argv[1])

test_data = data.sample(frac=0.2, random_state=47)
training_data = data.drop(test_data.index)

test_data.to_csv(index=False, path_or_buf="testSet.csv")
training_data.to_csv(index=False, path_or_buf="trainingSet.csv")
