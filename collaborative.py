# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

### Check the README.md for more details, especially on parameters

def movie_user_predictions(tuple_ids):
    """Convert the list of tuples (tuple_ids) into a np.array containing the lists of users associated to the corresponding (movie)
    for which we want to make the prediction"""
    result = []
    movie_i = []
    count = 0
    for ids in tuple_ids:
        u, m = ids[0]-1, ids[1]-1
        if m == count:
            movie_i.append(u)
        else:
            result.append(movie_i)
            movie_i = [u]
            count += 1
    # Append last one
    result.append(movie_i)
    return np.array(result)

def getTopKNeighbors(similarities, K):
    """Given the similarities of a user as an array, only keep the K best (max) similarities
       by setting other similarities to 0"""
    copy = np.copy(similarities)
    index = len(similarities) - K
    # Get the similarity value of the K best neighbor
    bestNeighborsVal = np.partition(similarities, index)[index]
    low_values_indices = copy < bestNeighborsVal # Where values are below the threshold
    copy[low_values_indices] = 0.0
    return copy

def compute_predictions_movie(ratings_matrix, movie, similarity_matrix, users_to_predict, K_best):
    """Compute all the wanted predictions for the given movie and returns the predictions in an array:
       Parameters: ratings_matrix: The ratings matrix with dimension movies x users
                   movie: Index of movie we want to predict (from 0 to 9999)
                   similarity_matrix: The similarity matrix between all users
                   users_to_predict: The list of users (indices) for which we want to compute the predictions
                   K_best: The number of neighbors to keep for each prediction """
    ratings = ratings_matrix[movie]
    # Following steps: only keep the similarities (user) if they have seen the movie
    copiedRatings = np.tile(ratings, (len(users_to_predict), 1))
    copiedRatings[copiedRatings != 0] = 1
    # Get similarity matrix of the wanted users with dimension: all users (10000) x wanted users(< 10000)
    similarities = similarity_matrix[users_to_predict].T
    similarities = np.multiply(similarities, copiedRatings.T)
    
    # Apply function getTopKNeighbors to each column of similarities to keep K best similarities for each user
    bestNeighbors = np.apply_along_axis(getTopKNeighbors, 0, similarities, K_best)
    # Get a list of sums of each column of bestNeighbors
    sumSimilarities = np.sum(bestNeighbors, axis = 0)

    #Compute prediction: dot product of the items in ratings @ bestNeighbors divided by the corresponding sum of similarities
    predictions = np.dot(ratings, bestNeighbors)
    predictions = predictions / sumSimilarities
    predictions[np.isnan(predictions)] = 0.0
    return predictions

def compute_matrix_predictions(ratings_matrix, ratings_to_predict, similarity_matrix, K_best):
    """Compute the wanted predictions by running the compute_predictions_movie for each movie"""
    nb_users, nb_movies = ratings_matrix.shape
    movie_user_matrix = np.copy(ratings_matrix).T 
    predictions = []
    ### Compute wanted predictions for wanted users
    for movie in range(nb_movies):
        users_to_predict = ratings_to_predict[movie]
        if(len(users_to_predict) > 0):
            predictions_movie = compute_predictions_movie(movie_user_matrix, movie, similarity_matrix, users_to_predict, K_best)
            predictions.append(predictions_movie)
    return predictions