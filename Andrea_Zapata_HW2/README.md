# CS 573 Homework 2
Andrea Zapata - PUID : 0031827996

Implementation of a Naive Bayes algorithm on on speed dating data.

## Running everything
As the assignment stated, the solution contains preprocess.py, 2_1.py, 2_2.py, discretize.py, split.py and 5_1.py, 5_2.py and 5_3.py 

To run the entire homework, the following commands need to be run in the same order.

```
python preprocess.py dating-full.csv dating.csv
python 2_1.py
python 2_2.py
python discretize.py dating.csv dating-binned.csv
python split.py dating-binned.csv
python 5_1.py
python 5_2.py
python 5_3.py
```
For a detailed description of a given file or the proposed solution, see the section below.

## Description of the scripts


### Preprocess.py

This script takes two arguments, the first argument is the original csv file which will be preproccesed, and the second is the output file.


```
python preprocess.py dating-full.csv dating.csv
```

### 2_1.py and 2_2.py

These files plot the graphs for section 2 of the homework (scores by gender and success rate by score). They both read the data from dating.csv, then "dating.csv" should be used as the second argument in the previous step.

```
python 2_1.py
python 2_2.py
```

### discretize.py

This script takes two arguments, the first argument is the original csv file which will be discretized, and the second is the output file.

```
python discretize.py dating.csv dating-binned.csv
```

For this problem, I wrote the ranges given in field-meaning.pdf in two dictionaries range_min and range_max. The function contains the parameters data, nbins to be able to reuse it for 5_2, the attribute print_count is used for printing the required output for this problem.

```
def binning_data(data, nbins, print_count =False)
```

### split.py

This script takes one argument as the original csv file which will be split. It outputs the files trainingSet.csv and testSet.csv.
```
python split.py dating-binned.csv
```

### 5_1.py

This script implements the Naive Bayes algorithm. It takes the training data from trainingSet.csv and the test data from testSet.csv and outputs the accuracy for the training and test set.
```
python 5_1.py
```

The function nbc_core learns the probabilities for the Naive Bayes Algorithm. 
feature_cols, class_col represent the corresponding attributes for features and class, class_val contains all possible class values, t_frac is the fraction of data used for training.
This function returns prob_decision which is a dictionary with the probabilities for each class and prob_attributes which is a (c , n) numpy array where c is the number of classes and n the number of feature columns. Each element D(i, j) of the array is a dictionary containing the conditional probabilities of the j-st column values given the class ci. 
```
def nbc_core(feature_cols, data, class_col, class_val, t_frac)
```
To match what was stated in the homework, I also have nbc which only takes t_frac and uses the default values for nbc_core.

```
def nbc(t_frac)
```


The function get_accuracy applies the learned probabilities to the given data to calculate its accuracy. The inputs are similar to the previous function and prob_decision, prob_attributes are the outputs of nbc.
For this function, I created a m,c numpy array where m is the size of data and c is the number of classes which will contain the probabilities for each class. First, I multiply each column of the array for the probability of the corresponding class. After that, I iterate over the feature columns, and multiply each of the class probabilities times the conditional probability of the value of the column given that class.
Finally, I take the max probability for each row to set the prediction and then compare it with the true class value to calculate the accuracy.

```
def get_accuracy(data, class_col, classes_val, feature_cols, prob_decision, prob_attributes):
```

### 5_2.py

This script uses the Naive Bayes algorithm for different binning sizes. It takes the data from dating.csv and uses the method from discretize.py to calculate the new dataframe using the given number of bins. For each binned data, the script uses the algorithm in 5_1 to calculate the training and testing accuracy. Finally, it plots the training and testing accuracy for each bin size.
```
python 5_2.py
```

### 5_3.py

This script uses the Naive Bayes algorithm for different sizes of training data. It takes the data from trainingSet.csv and testSet.csv and uses the t_frac attribute of nbc to learn the probabilities using different fractions of the training data. For each learned model, the script uses the algorithm in 5_1 to calculate the training and testing accuracy. Finally, it plots the training and testing accuracy for each fraction of training data.
```
python 5_2.py
```



