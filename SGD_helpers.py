import numpy as np
import scipy.sparse as sp

### Check the README.md for more details, especially on parameters of the functions

def init_MF(train, num_features):
    """Create the 2 features matrix: user_features, item_features given the matrix of ratings and the number of features"""
    num_item, num_user = train.get_shape()
    # All elements are random and in the interval [0, 1/num_features] 
    user_features = np.random.rand(num_features, num_user)/num_features
    item_features = np.random.rand(num_features, num_item)/num_features

    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (RMSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
    return np.sqrt(1.0 * mse / len(nz))

def matrix_factorization_SGD(train, test, gamma, num_features, lambda_user, lambda_item, num_epochs,
                             user_feat, item_feat, include_test = True):
    """matrix factorization by SGD. include_test set to False if we want to train on the whole ratings matrix, thus we have no test"""
    # set seed
    np.random.seed(988)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    # same for tests only if include_test is true
    nz_test = []
    if(include_test):
        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        
    # make copy of user_features and item_features matrices and modify the copies    
    user_features = np.copy(user_feat)
    item_features = np.copy(item_feat)
    print("Learn the matrix factorization using SGD with K = {}, lambda_i = {}, lambda_u = {}, num_epochs = {}".format(num_features, lambda_item, lambda_user, num_epochs))
    
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            err = train[d, n] - user_info.T.dot(item_info)
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)
            
        rmse = compute_error(train, user_features, item_features, nz_train)
        if(it % 5 == 0 or it == num_epochs - 1):
            print("iter: {}, RMSE on training set: {}.".format(it, rmse))

    # evaluate the test error
    if include_test:
        rmse = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse))
    return item_features, user_features, rmse