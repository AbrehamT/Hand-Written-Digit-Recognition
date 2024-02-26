import pandas as pd
import numpy as np


training = pd.read_csv('MNIST_training.csv', header=None, skiprows=1)
test = pd.read_csv('MNIST_test.csv', header=None, skiprows=1)


training_data = np.array(training.drop(0, axis=1))
testing_data = test.drop(0, axis=1)

"""
    Dictionary to store the results of different K-Values
"""
data = {'K-Values':[], 'Accuracy %':[], 'Properly Labeled':[], 'Improperly Labeled':[]}


k_max = 20

for k_val in range(1, k_max):

    """

        Creating a dictionary where the predicted labels are
            stored in the respective ground truth.
            For example, all values that should be predicted as zero are stored
            in predicted_label[0]. If an element in predicted_label[0] is not labeled as
            zero then that means that the model has incorrectly classified that image

    """

    predicted_label = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: []
    }

    for testing_sample in range(0, 50, 1):

        #Retiling testing_query to have same dimensions as all the labels in training_data
        test_query = np.tile(testing_data.iloc[testing_sample, :], (949, 1))

        # Performing element wise subtraction, essentially train_pixelvalue_k - test_pixelvalue_k for every pixel
        diff = training_data - test_query
        diff_transpose = np.transpose(diff)
        sum_of_squares = []

        # Finding the dot product between a row in the difference matrix and it's transpose
        #   This will give the Sum of the Squares of a pixel value from testing data and an instance in training data
        for i in range(949):
            sum_of_squares.append(np.dot(diff[i, :], diff_transpose[:, i]))

        # Finding the Square root of the Sum of Squares
        dist = np.sqrt(np.array(sum_of_squares))

        """
            Creating a set for the distance between the query instance
            and the training data. Each label represents the distance between that label
            and the query.
        """
        label_0 = set(dist[i] for i in range(0,  95))
        label_1 = set(dist[i] for i in range(95, 190))
        label_2 = set(dist[i] for i in range(190, 285))
        label_3 = set(dist[i] for i in range(285, 380))
        label_4 = set(dist[i] for i in range(380, 475))
        label_5 = set(dist[i] for i in range(475, 570))
        label_6 = set(dist[i] for i in range(570, 665))
        label_7 = set(dist[i] for i in range(665, 760))
        label_8 = set(dist[i] for i in range(760, 855))
        label_9 = set(dist[i] for i in range(855, 949))

        # Sorting the distance
        dist_sort = np.sort(dist)

        """
         Creating a list where each index corresponds to a label
          -and the value at that index corresponds to the majority vote for that label
        """

        majority_vote = [0] * 10

        for i in range(k_val):
            if (dist_sort[i] in label_0):
                majority_vote[0] += 1
            if (dist_sort[i] in label_1):
                majority_vote[1] += 1
            if (dist_sort[i] in label_2):
                majority_vote[2] += 1
            if (dist_sort[i] in label_3):
                majority_vote[3] += 1
            if (dist_sort[i] in label_4):
                majority_vote[4] += 1
            if (dist_sort[i] in label_5):
                majority_vote[5] += 1
            if (dist_sort[i] in label_6):
                majority_vote[6] += 1
            if (dist_sort[i] in label_7):
                majority_vote[7] += 1
            if (dist_sort[i] in label_8):
                majority_vote[8] += 1
            if (dist_sort[i] in label_9):
                majority_vote[9] += 1

        """
            Putting in the number with the most votes based on what the testing query is
        """
        if testing_sample < 5:
            predicted_label[0].append(majority_vote.index(max(majority_vote)))
        if 5 <= testing_sample < 10:
            predicted_label[1].append(majority_vote.index(max(majority_vote)))
        if 10 <= testing_sample < 15:
            predicted_label[2].append(majority_vote.index(max(majority_vote)))
        if 15 <= testing_sample < 20:
            predicted_label[3].append(majority_vote.index(max(majority_vote)))
        if 20 <= testing_sample < 25:
            predicted_label[4].append(majority_vote.index(max(majority_vote)))
        if 25 <= testing_sample < 30:
            predicted_label[5].append(majority_vote.index(max(majority_vote)))
        if 30 <= testing_sample < 35:
            predicted_label[6].append(majority_vote.index(max(majority_vote)))
        if 35 <= testing_sample < 40:
            predicted_label[7].append(majority_vote.index(max(majority_vote)))
        if 40 <= testing_sample < 45:
            predicted_label[8].append(majority_vote.index(max(majority_vote)))
        if 45 <= testing_sample < 50:
            predicted_label[9].append(majority_vote.index(max(majority_vote)))

    """
        Calculating the accuracy by finding which elements in a given class do not match the label
        For example, in the class of samples for 0, we count how many of those samples have a label 
        other than 0, meaning that the model improperly classified a given sample as some other number.
    """
    accuracy_count = 50
    for label in predicted_label:
        wrong = [x for x in predicted_label[label] if x != label]
        accuracy_count -= len(wrong)

    data['K-Values'].append(k_val)
    data['Accuracy %'].append((accuracy_count / 50)*100)
    data['Properly Labeled'].append(accuracy_count)
    data['Improperly Labeled'].append(50 - accuracy_count)

results = pd.DataFrame(data=data, index=None)
print(results.to_string(index=False))