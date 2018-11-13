Here's a short explanation of the files function:

-> training.py [learning_rate] [regularization_rate]
takes parameters to run training on the data available in data/training/
example:
    python3 training 0.1 0.001
where:
    0.1 is the learning rate and 0.001 is the regularization factor for L2 Regularization

-> evaluation.py
takes parameters to runs an evaluatio on the data available in data/validation/
example:
    python3 training 0.1 0.001
where:
    0.1 is the learning rate and 0.001 is the regularization factor for L2 Regularization

-> feature_sets.py
a helper class and methods to generate features for each file in the provided paths and their corresponding class name

-> model_parameter_estimator.py
generates candidates of learning rate and regularization rate to test them and returns the one with the best accuracy an validation set

-> annotate.py [file_name]
reads the weights from the latest training and annotates a file

Some additional piece of advise:
-> the data is split as follows:
 - 850 docs pos/neg training
 - 100 docs pos/neg dev
 - 50 docs pos/neg validation