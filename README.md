The KNN algorithm for this project was implemented using the Euclidean distance approach. This was first done by retiling each sample in the test file, which had a feature vector of dimension 1x784, into a matrix with a dimension of 949x784 and then finding the element wise difference between the sample matrix and the retiled test-query matrix. Note that the retiled matrix contains 948 duplicates of the same test-query sample. This was done in order to make it easy to calculate the distance between the test-query and all samples in the training set. 

After, the dot product of each row in the difference matrix was calculated against its transpose to get us the sum of the squares of all the values in that given row. After calculating the dot product, the square root of every value in the newly generated 949x1 matrix was calculated. This whole process of taking the square of the dot product of the difference between the test-query and training sample essentially gives us the euclidean distance between the feature vector for the test_query and the feature vector of every sample in the training set. 

After finding the distance between the test-query and every sample in the training set this data was sorted in ascending order, where the smallest value implied the closest samples to the test-query from the training set. Then the Kth smallest samples were picked and based on the label of the majority of those samples, the given test-query was predicted to have the same label.

This whole process was repeated for every sample in the test set.

An interesting part about this assignment was having to figure out the proper way to do the vector multiplications. I did not want a loop based approach for each feature value so I attempted to apply my knowledge from linear algebra to perform the vector multiplication efficiently. 

The KNN algorithm is a machine learning algorithm that’s mostly used for classification but sometimes for regression tasks. The steps to how it works are as follows:
Choose the number of `k` neighbors: `k` is a parameter that refers to the number of nearest neighbors to include in the majority vote (for classification) or average (for regression). The choice of `k` affects the algorithm's performance and an optimal value for k is discovered through experimentation. In this project the optimal value seems to be 7-9 closest neighbors.

Calculate the distance: For a given data point, the algorithm calculates the distance between that point and every other point in the dataset. 
Find the nearest neighbors: The algorithm identifies the `k` nearest data points (neighbors) based on the distance calculations.

Classification: The algorithm then assigns the class based on the majority vote of these neighbors


Predictions for k values between 1-19:
![KNN RESULTS] (KNN_RESUTLS.png)