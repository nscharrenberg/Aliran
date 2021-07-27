import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
from python_speech_features import mfcc

dataset = []
training_set = []
test_set = []


# Get the distance between feature vectors
def distance(instance1, instance2, k):
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]

    dist = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    dist += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1))
    dist += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    dist -= k

    return dist


# Find Neighbors
def get_neighbors(training_dataset, instance, k):
    distances = []

    for i in range(len(training_dataset)):
        dist = distance(training_dataset[i], instance, k) + distance(instance, training_dataset[i], k)
        distances.append((training_dataset[i][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for i in range(k):
        neighbors.append(distances[i][0])

    return neighbors


# Identify the Nearest Neighbor (Genres)
def nearest_genre(neighbors):
    class_vote = {}

    for i in range(len(neighbors)):
        res = neighbors[i]

        if res in class_vote:
            class_vote[res] += 1
        else:
            class_vote[res] = 1

    sorted_vote = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_vote[0][0]


# Model Evaluation to get the accuracy
def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1

    return 1.0 * correct / len(test_set)


# Extract features from the audio files and store them in a model file
def extract_features(filename):
    directory = "Data/genres_original/"
    f = open(filename, "wb")
    it = 0

    for tempDir in os.listdir(directory):
        it += 1
        if it == 11:
            break
        for file in os.listdir(directory + tempDir):
            try:
                print(file)
                (rate, sig) = wav.read(directory + tempDir + "/" + file)
                mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                covariance = np.cov(np.matrix.transpose(mfcc_feat))
                mean_matrix = mfcc_feat.mean(0)
                feature = (mean_matrix, covariance, it)
                pickle.dump(feature, f)
            except Exception:
                f.close()

    f.close()


# Load in the Dataset
def load_dataset(filename, split, tr_set, te_set):
    with open(filename, "rb") as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for i in range(len(dataset)):
        if random.random() < split:
            tr_set.append(dataset[i])
        else:
            te_set.append(dataset[i])


if __name__ == '__main__':
    print('Starting....')
    local_filename = "dataset.aliran"

    extracting = False

    if extracting:
        print('Extracting Features...')
        print('Building Model...')
        extract_features(local_filename)

    print('Loading Dataset...')
    load_dataset(local_filename, 0.66, training_set, test_set)

    print('Making a prediction...')
    print('(This may take a few minutes)')
    predictions = []

    for x in range(len(test_set)):
        predictions.append(nearest_genre(get_neighbors(training_set, test_set[x], 5)))

    accuracy = get_accuracy(test_set, predictions)
    print('Prediction Accuracy is:')
    print(accuracy)
