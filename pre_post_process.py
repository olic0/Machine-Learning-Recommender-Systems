# -*- coding: utf-8 -*-
import csv
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp

"""Pre process"""
def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]
    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_col, max_row))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    ids = np.zeros((len(data), 2), dtype=np.int)
    i = 0
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
        ids[i] = [row, col]
        i += 1
    ### ratings.T as we want dimension (movies x users)
    return ids, ratings.T

def split_data(ratings, p_test=0.1):
    """Split the ratings to training data and test data with the given proportion in p_test"""
    # set seed
    np.random.seed(988)
    
    valid_ratings = ratings
    
    # init
    num_rows, num_cols = valid_ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    print("the shape of original ratings. (# of row, # of col): {}".format(
        ratings.shape))
    print("the shape of valid ratings. (# of row, # of col): {}".format(
        (num_rows, num_cols)))

    nz_items, nz_users = valid_ratings.nonzero()
    
    # split the data
    for user in set(nz_users):
        # randomly select a subset of ratings
        row, col = valid_ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test))
        residual = list(set(row) - set(selects))

        # add to train set
        train[residual, user] = valid_ratings[residual, user]

        # add to test set
        test[selects, user] = valid_ratings[selects, user]

    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

"""Post process for submissions"""
def getWantedPredictions(predictions_matrix, ids):
    """Computes array of the wanted predictions given the list of ids of the form [user_id, movie_id]"""
    wanted_predictions = []
    for user, movie in ids:
        wanted_predictions.append(predictions_matrix[user - 1, movie - 1])
    return wanted_predictions

def convert_ids_for_submission(ids):
    """Convert the list of tuple ids expressed as [user_id = 44, movie_id = 1] into a correct list for the submission: 'r44_c1'"""
    result = []
    for id_ in ids:
        newId = 'r'+str(id_[0])+'_c'+str(id_[1])
        result.append(newId)
    return result

def create_csv_submission(ids, pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    ids_for_submission = convert_ids_for_submission(ids)
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for r1, r2 in zip(ids_for_submission, pred):
            writer.writerow({'Id':str(r1),'Prediction':float(r2)})
