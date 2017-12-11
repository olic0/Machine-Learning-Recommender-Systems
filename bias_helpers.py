import numpy as np
import scipy.sparse as sp

# ratings is matrix of dimension: items x users
def computeBiasMatrix(ratings):
    """Compute the bias matrix of same dimension as the ratings matrix
    where each element is equal to r_ui - mu - bias_u - bias_i"""
    num_items, num_users = ratings.shape
    # find the non-zero ratings indices 
    nz_row, nz_col = ratings.nonzero()
    nz_ratings = list(zip(nz_row, nz_col))
    
    # mu is overall mean (of non zeros)
    mu = np.sum(ratings)/len(nz_ratings)

    nz_users = np.zeros((num_users,), dtype=int)
    nz_items = np.zeros((num_items,), dtype=int)
    # Compute number of non zero ratings for each user and item
    for item, user in nz_ratings:
        nz_users[user] += 1
        nz_items[item] += 1
    
    mean_users = np.sum(ratings, axis = 0)
    mean_users = [mean_users[0, i] for i in range(num_users)]/nz_users
    mean_items = np.sum(ratings, axis = 1)
    mean_items = [mean_items[i, 0] for i in range(num_items)]/nz_items
    
    bias_users = mean_users - np.ones(num_users) * mu
    bias_items = mean_items - np.ones(num_items) * mu
    new_ratings = sp.lil_matrix((num_items, num_users))
    for item, user in nz_ratings:
        new_ratings[item, user] = ratings[item, user] - (mu + bias_users[user] + bias_items[item])
    return new_ratings, mu, bias_users, bias_items

def predictionsWithBias(item_features, user_features, bias_u, bias_i, mean_rating):
    '''Computes the final predictions matrix using the formula: pred_u_i = mean + bias_u + bias_i + item_features.T @ user_features'''
    # item_features and user_features of shape: K x num_items/users
    preds_matrix = np.dot(item_features.T, user_features)
    num_items, num_users = preds_matrix.shape
    ### biasU: each elements of a column has same value (corresponding to the user bias)
    biasU = np.tile(bias_u, (num_items,1))
    ### biasI: each elements of a row has same value (corresponding to the movie bias)
    biasI = np.tile(bias_i, (num_users,1)).T
    mean_matrix = np.ones((num_items, num_users)) * mean_rating
    
    predictionsBiased = mean_matrix + biasU + biasI + preds_matrix
    print(np.shape(predictionsBiased))
    return predictionsBiased