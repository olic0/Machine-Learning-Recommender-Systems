from pre_post_process import *
from SGD_helpers import *
from bias_helpers import *

if __name__ == '__main__':
    
    # Load the data in sparse matrices
    print("Loading Train")
    _, ratings = load_data("data_train.csv")
    print("Loading Sample")
    sample_ids, _ = load_data("sample_submission.csv")
    
    print("Splitting data: 90% in training, 10% in testing")
    _, train, test = split_data(ratings, p_test = 0.1)
    
    # Compute biased training and testing matrices
    bias_train, mean, bias_u_train, bias_i_train = computeBiasMatrix(train)
    bias_test, _, _, _ = computeBiasMatrix(test)
    
    # Best parameters found with grid search
    gamma = 0.025
    lambdas_user = np.logspace(-3,0,4)[::-1] #From max to min
    lambdas_item = np.logspace(-3,0,4)[::-1] #From max to min
    num_features = 20
    num_epochs = 20
    min_loss = 1000
    
    best_user_feats = []
    best_item_feats = []

    # Initialise user features and item features:
    user_init, item_init = init_MF(ratings, num_features)
    
    # Run Grid Search on lambdas_user and lambdas_item 
    for x,lambda_u in enumerate(lambdas_user):
        for y,lambda_i in enumerate(lambdas_item):
            print("K = {}, lambda_u = {}, lambda_i = {}".format(num_features, lambda_u, lambda_i))
            item_features, user_features, rmse = matrix_factorization_SGD(bias_train, bias_test, gamma, num_features, lambda_u,
                                                                    lambda_i, num_epochs, user_init, item_init)
            ### For warm start, we keep the user_features and item_features that gave us the minimal rmse previously computed
            if rmse < min_loss:
                min_loss = rmse
                user_init = user_features
                item_init = item_features
                best_user_feats = np.copy(user_features)
                best_item_feats = np.copy(item_features)
                
    print("Computing predictions with min test loss being: {}".format(min_loss))
    # Compute predictions with best features matrices
    predictions =  predictionsWithBias(best_item_feats, best_user_feats, bias_u_train, bias_i_train, mean)
    
    # Set predictions above 5 (below 1) to 5 (1)
    predictions[ np.where(predictions > 5.0)] = 5.0
    predictions[ np.where(predictions < 1.0)] = 1.0
    
    # Convert matrix of predictions to array of the wanted ones
    preds = getWantedPredictions(predictions.T, sample_ids)
    
    create_csv_submission(sample_ids, preds, 'submission.csv')
