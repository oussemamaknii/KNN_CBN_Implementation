Introduction to Artificial Intelligence - LAB 4
ISEP – May 4, 2022

Instructions: Prepare a report including the source code and the
results. Submit the report on moodle until May 11, 23:59.
Remark: This lab will be done using Python 3.6.

The objective of this lab is to program and test two classification algorithms,
very simple but very effective: the K-Nearest Neighbor (KNN) algorithm and the
Classifier Bayesian Naive (CBN). We are studying here only the simplest versions
of these algorithms. For this lab we will need to import sklearn and numpy. The
tests can be done on sklearn’s predefined data that comes with their class labels
(target), for example:

iris = datasets.load_iris ()
X = iris.data
Y = iris.target
A Nearest neighbor

The Nearest Neighbor algorithm is a very simple classification algorithm which is
based on the following principle: the class of each test data (to be classified) must
be the class of the closest (most similar) data among the training data. List of
useful functions:
- metrics.pairwise.euclidean_distances: calculates distances between data.
- argsort: returns the indices of the ordered vector
- argmin, argmax: returns the indices of the minimum/maximum values
- neighbors.KNeighborsClassifier: K Nearest Neighbors alg. of sklearn
1. Create a TNN function (X, Y) which takes X data and labels as input Y and
which returns a label, for each data, predicted from the nearest neighbor of
this data. Here we take each data, one by one, as data test and we consider
all others as learning data. That we allows to test the power of our algorithm
according to a validation method by cross validation of “leave one out” type.
2. The TNN function calculates a predicted label for each data. Change the
function to calculate and return the prediction error: i.e. the percentage
badly predicted labels
3. Test on Iris data.
4. Test the function of the K Closest Neighbors of sklearn (with here K = 1).
Are the results different? Test with other values of K.
5. BONUS: Modify the TNN function so that it takes as input a number K of
neighbors (instead of 1). The predicted class will then be the majority class
among the K neighbors.

B Naive Bayesian classifier
The Naive Bayesian Classifier algorithm is a classification algorithm based on
calculating the probability of belonging to each class. That is to say that the test
data (to be classified) will be assigned to the most likely class. The probability of
belonging to each class is calculated from the learning data as follows:
class(x) = arg max
ωk
{ΠiP(wi/ωk)P(ωk)} (1)
Here P(ωk) is the a priori probability of belonging to the class k. in other words
it’s the probability of obtaining a data of class k if we draw a data at random.
P(xi/ωk) is the probability that a data x has the value xi
for the variable i, if
we know its class ωk. Here we will calculate this probability by calculating the
distance between the data xi and each barycenter of the classes (i.e. the class
average), divided by the sum distances between this data and each barycenter.
List of useful functions:
- mean, sum: calculate the mean and the sum of a list of values.
- unique: returns the list of list values, without repeating values.
- asarray: transforms a list into a vector.
- vector.prod: makes the product of the values of a vector.
- naive_bayes.GaussianNB: the naive Bayesian Classifier from sklearn.
1. Create a CBN (X, Y) function that takes X data and labels as input Y
and which returns a label, for each data, predicted from class la more likely
according to equation (1). Here again, we take each data, one by one, as
test data and we consider all data as training data. It is advisable to first
calculate the barycentres and the a priori probabilities P(ωk) for each class,
then calculate the conditional probabilities P(xi/ωk) for each class and each
variable.
2. The CBN function calculates a predicted label for each data. Change the
function to calculate and return the prediction error: i.e. the percentage of
badly predicted labels. Test on Iris data.
3. Test the function of the Naive Bayesian Classifier included in sklearn. This
function uses a Gaussian distribution instead of the barycenter distances.
The results are they different?
4. BONUS: Modify the CBN function so that it uses a Gaussian distribution
for probability laws instead of simple distance to the barycenter.
