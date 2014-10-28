# An Implementation on Venn-ABERS Predictor

This is an implementation of Venn-ABERS predictor.

Venn-ABERS predictor is a recently developed algorithm for multi-probability predictions. It is modified from Zadrozny and Elkan's procedure of probability forecasting, which can be poorly calibrated. The modification introduced Venn predictors into the procedure to overcome the problem of potential poor calibration due to that Venn predictors are always well calibrated and guaranteed to be well calibrated under the exchangeability assumption. The basic idea of Venn-ABERS predictor is that the training set is split into two parts: the proper training set, which is used to train the learning machine, and the calibration set, which is to calculate the probabilistic outputs. The calibration set will be transformed into a monotone increasing set in this algorithm according to the paper of M. Ayer, H.D. Brunk, G.M. Ewing, W.T. Reid and E. Silverman. The term _"ABERS"_ also comes from the acronyms of their surnames.

## Table of Contents

* [Installation and Data Format](#installation-and-data-format)
* ["va-offline" Usage](#va-offline-usage)
* ["va-online" Usage](#va-online-usage)
* ["va-cv" Usage](#va-cv-usage)
* [Tips on Practical Use](#tips-on-practical-use)
* [Examples](#examples)
* [Precomputed Kernels](#precomputed-kernels)
* [Additional Information](#additional-information)
* [Acknowledgments](#acknowledgments)

## Installation and Data Format[↩](#table-of-contents)

On Unix systems, type `make` to build the `va-offline`, `va-online` and `va-cv` programs. Run them without arguments to show the usages of them.

The format of training and testing data file is:
```
<label> <index1>:<value1> <index2>:<value2> ...
...
...
...
```
Each line contains an instance and is ended by a `'\n'` character. For Venn-ABERS predictor, `<label>` is an integer indicating the class label (only support binary case problems). The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 1 and `<value>` is a real number. The only exception is the precomputed kernel, where `<index>` starts from 0; see the section of precomputed kernels. Indices must be in **ASCENDING** order. Labels in the testing file are only used to calculate accuracy or errors. If they are unknown, just fill the first column with any numbers.

## "va-offline" Usage[↩](#table-of-contents)

```
Usage: va-offline [options] train_file test_file [output_file]
options:
  -s model_file_name : save model
  -l model_file_name : load model
  -a ratio : set ratio of proper training set takes of all training set in Venn-ABERS (default 0.7)
  -b probability estimates : whether to output probability estimates for all labels, 0 or 1 (default 0)
  -f calibrated prediction : whether to calibrate prediction based on the higher possibility, 0 or 1 (default 0)
  -t svm_type : set type of SVM (default 0)
    0 -- C-SVC    (multi-class classification)
    1 -- nu-SVC   (multi-class classification)
  -k kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
    4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of C-SVC (default 1)
  -n nu : set the parameter nu of nu-SVC (default 0.5)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -wi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
  -q : quiet mode (no outputs);
```
`train_file` is the data you want to train with.  
`test_file` is the data you want to predict.  
`va-offline` will produce outputs in the `output_file` by default.

## "va-online" Usage[↩](#table-of-contents)

```
Usage: va-online [options] data_file [output_file]
options:
  -a ratio : set ratio of proper training set takes of all training set in Venn-ABERS (default 0.7)
  -b probability estimates : whether to output probability estimates for all labels, 0 or 1 (default 0)
  -f calibrated prediction : whether to calibrate prediction based on the higher possibility, 0 or 1 (default 0)
  -t svm_type : set type of SVM (default 0)
    0 -- C-SVC    (multi-class classification)
    1 -- nu-SVC   (multi-class classification)
  -k kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
    4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of C-SVC (default 1)
  -n nu : set the parameter nu of nu-SVC (default 0.5)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -wi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
  -q : quiet mode (no outputs);```
```
`data_file` is the data you want to run the online prediction on.  
`va-online` will produce outputs in the `output_file` by default.


## "va-cv" Usage[↩](#table-of-contents)

```
Usage: va-cv [options] data_file [output_file]
options:
  -a ratio : set ratio of proper training set takes of all training set in Venn-ABERS (default 0.7)
  -b probability estimates : whether to output probability estimates for all labels, 0 or 1 (default 0)
  -v num_folds : set number of folders in cross validation (default 5)
  -f calibrated prediction : whether to calibrate prediction based on the higher possibility, 0 or 1 (default 0)
  -t svm_type : set type of SVM (default 0)
    0 -- C-SVC    (multi-class classification)
    1 -- nu-SVC   (multi-class classification)
  -k kernel_type : set type of kernel function (default 2)
    0 -- linear: u'*v
    1 -- polynomial: (gamma*u'*v + coef0)^degree
    2 -- radial basis function: exp(-gamma*|u-v|^2)
    3 -- sigmoid: tanh(gamma*u'*v + coef0)
    4 -- precomputed kernel (kernel values in training_set_file)
  -d degree : set degree in kernel function (default 3)
  -g gamma : set gamma in kernel function (default 1/num_features)
  -r coef0 : set coef0 in kernel function (default 0)
  -c cost : set the parameter C of C-SVC (default 1)
  -n nu : set the parameter nu of nu-SVC (default 0.5)
  -m cachesize : set cache memory size in MB (default 100)
  -e epsilon : set tolerance of termination criterion (default 0.001)
  -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
  -wi weights : set the parameter C of class i to weight*C, for C-SVC (default 1)
  -q : quiet mode (no outputs);
```
`data_file` is the data you want to run the cross validation on.  
`va-cv` will produce outputs in the `output_file` by default.

## Tips on Practical Use[↩](#table-of-contents)

* Scale your data. For example, scale each attribute to `[0,1]` or `[-1,+1]`.
* Before use, consider using grid search to find the best parameters for SVM.
* If data for classification are unbalanced (e.g. many positive and few negative), try different penalty parameters `C` by `-wi` (see examples below).
* Specify larger cache size (i.e., larger `-m`) for huge problems.

## Examples[↩](#table-of-contents)

```
> va-offline train_file test_file output_file
```

Train a venn-abers predictor by using default settings (C-SVC with RBF kernel) on `train_file`. Then conduct this classifier to `test_file` and output the results to `output_file`.

```
> va-offline -k 1 -s model_file train_file test_file
```

Train a venn-abers predictor using support vector machines with polynomial kernel on `train_file`. Then conduct this classifier to `test_file` and output the results to the default output file, also the model will be saved to file `model_file`.

```
> va-online -t 2 -g 0.5 -c 2 data_file
```

Train an online venn-abers predictor classifier using support vector machine with RBF kernel on `data_file`. Also the cost of support vector machines is set to 2 while the gamma in RBF kernel is set to 0.5. Then output the results to the default output file.

```
> va-cv -t 1 -v 10 data_file
```

Do a 10-fold cross validation venn-abers predictor using support vector machine with linear kernel on `data_file`. Then output the results to the default output file.

## Precomputed Kernels[↩](#table-of-contents)

Users may precompute kernel values and input them as training and testing files. Then predictor does not need the original training/testing sets.

Assume there are `L` training instances `x1, ..., xL` and Let `K(x, y)` be the kernel value of two instances `x` and `y`. The input formats are:

New training instance for `xi`:

    <label> 0:i 1:K(xi,x1) ... L:K(xi,xL) 

New testing instance for any `x`:

    <label> 0:? 1:K(x,x1) ... L:K(x,xL) 

That is, in the training file the first column must be the "ID" of `xi`. In testing, `?` can be any value.

All kernel values including __ZEROs__ must be explicitly provided. Any permutation or random subsets of the training/testing files are also valid (see examples below).

### Examples

Assume the original training data has three four-feature instances and testing data has one instance:

```
0  1:1 2:1 3:1 4:1
1      2:3     4:3
1          3:1

0  1:1     3:1
```

If the linear kernel is used, we have the following new training/testing sets:

```
0  0:1 1:4 2:6  3:1
1  0:2 1:6 2:18 3:0 
1  0:3 1:1 2:0  3:1

0  0:? 1:2 2:0  3:1
```

`?` can be any value.

Any subset of the above training file is also valid. For example,

```
1  0:3 1:1 2:0  3:1
1  0:2 1:6 2:18 3:0 
```

implies that the kernel matrix is

```
[K(2,2) K(2,3)] = [18 0]
[K(3,2) K(3,3)] = [0  1]
```


## Additional Information[↩](#table-of-contents)
For any questions and comments, please email [c.zhou@cs.rhul.ac.uk](mailto:c.zhou@cs.rhul.ac.uk)

## Acknowledgments[↩](#table-of-contents)
Special thanks to Chih-Chung Chang and Chih-Jen Lin, which are the authors of [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/).