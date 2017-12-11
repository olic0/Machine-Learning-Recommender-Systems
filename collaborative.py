# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def convert_ID(string):
    """Convert the string ID of the form 'r44_c1' into the user_id 44 and movie_id 1"""
    remove_chars = string.replace("r","").replace("c", "").replace("_", " ")
    split = remove_chars.split()
    user_id = int(split[0])
    movie_id = int(split[1])
    return user_id, movie_id

def load_csv_data(data_path, nb_users, nb_movies):
    """Loads data and returns the rating matrix"""
    X = np.genfromtxt(data_path, dtype= str, delimiter=',', skip_header = True)
    ratings_matrix = np.zeros((nb_users, nb_movies), dtype=np.int)
    ids = np.zeros((len(X), 2), dtype=np.int)
    
    for i in range(X.shape[0]):
        user_id, movie_id = convert_ID(X[i][0])
        ids[i] = [user_id, movie_id]
        ### Check for errors
        if(user_id > 0 and user_id <= nb_users and movie_id > 0 and movie_id <= nb_movies):
            ### Use -1 because IDs start at 1 and we want them to start at 0
            ratings_matrix[user_id - 1][movie_id - 1] = int(X[i][1])
        else:
            print("Error with user {} and movie {}".format(user_id, movie_id))
    return ratings_matrix, ids

def movie_user_predictions(sample_matrix):
    """Compute the matrix of indices we want to predict with movies as rows and users 
    for which we want to predict the rating for this movie"""
    result = []
    movie_user_matrix = sample_matrix.T
    for movie_ratings in movie_user_matrix:
        user_indices = [user for user, rating in enumerate(movie_ratings) if rating != 0]
        result.append(user_indices)
    return np.array(result)


def convert_ids_for_submission(ids):
    """Convert the list of ids expressed as [user_id = 44, movie_id = 1] into a correct list for the submission: 'r44_c1'"""
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
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for r1, r2 in zip(ids, pred):
            writer.writerow({'Id':str(r1),'Prediction':float(r2)})
