# Project Recommender System
### Olivier Couque (246212), Emma Lejal Glaude (250692), Matthieu Sauvé (246243)

This project contains the following elements :
- PDF Report of the Project
- 3 iPython Notebooks: Recommender_Collaborative.ipynb, Recommender_Factorization.ipynb, Recommender_Surprise.ipynb
- 5 python modules : pre_post_process.py, collaborative.py, SGD_helpers.py, bias_helpers.py run.py

_Note_ : in order to do the preprocessing you will need the files "data_train.csv" and "sample_submission.csv" of the Kaggle competition available at this link: https://www.kaggle.com/c/epfml17-rec-sys

Used Libraries: 
- Surprise: http://surpriselib.com/ All informations about installations are provided at this link (pip install scikit-surprise) and the documentation is really detailed. Our Recommender_Surprise.ipynb notebook, described below, shows a nice and not too complex use of the library.
- Pandas: https://pandas.pydata.org/ (pip install pandas or conda install pandas with Anaconda). We only used pandas to load the ratings in a dataframe to be able to use the Surprise.Dataset.load_from_df function that needed a pandas.DataFrame as input.
- Sklearn: http://scikit-learn.org/stable/install.html for the installation. We used this library only to compute the similarities matrices between users in the Recommender_Collaborative.ipynb notebook.
- Numpy / Scipy: https://www.scipy.org/install.html for the installation (already available with Anaconda).
- Seaborn: https://seaborn.pydata.org/installing.html#installing for the installation. Seaborn was only used to visualize the grid search (for the report).
				
1. Report : this is where you will discover how we tackled the competition. This document explains the preprocessing and every step we took to achieve this current version. It also explains which Machine Learning model we use for the predictions and specify the value of the parameters we used.

2. Notebooks : 

	- Recommender_Collaborative.ipynb: Notebook using the User - user collaborative filtering technique. Here are the following steps of the notebook:
		
		1. Load the data, divide the ratings matrix in training and testing set.
		2. Statistics on the dataset
		3. Presentation of the algorithm used: similarity metrics, how to compute the predictions
		4. Find best parameters (similarity matrix and number K of best neighbors to keep for each user) by running the algorithm numerous times with different parameters: train on training ratings and test on testing ratings with RMSE.
		5. Compute the wanted predictions on the whole ratings (no test), using the algorithm with the best found parameters.
		6. Creation of csv file for the submission.
	
	- Recommender_Factorization.ipynb:
	
		1. Load the data, Divide the ratings matrix in training and testing set.
		2. Basic Matrix Factorization: 
		- find best parameters, first the learning rate gamma, then the regularizers and number of features with grid search.	
		- compute the SGD on the whole ratings matrix with best found parameters
		- compute the predictions with the user features and item features found with SGD
		3. Biased Matrix Factorization: 
		- Convert training and testing ratings in the biased training ratings and biased testing ratings
		- Run same steps of Basic Matrix Factorization on the biased training ratings and biased testing ratings
		- compute the predictions with user features, item features found with SGD, overall mean, bias of users and bias of items
		4. Setting the predictions below 1.0 to 1.0 and those above 5.0 to 5.0 (just a few of them)
		5. Creation of csv file for the submission
	
	- Recommender_Surprise.ipynb: Notebook using the Surprise library for building and analyzing recommender systems. Here are the following steps of the notebook:
		
		1. Load the data in a pandas dataframe. Create the ratings matrix from the dataframe using Surprise.
		2. Initialize and run a grid search to find the best parameters with cross validation (5 folds) using the library.
		3. Run the matrix factorization algorithm (called SVD in Surprise which actually is the biased matrix factorization using SGD) on the full ratings with the best parameters giving the minimal previous testing loss. 
		4. Predict the wanted ratings given the user ids and item ids
		5. Creation of csv file for the submission

3. Python modules :

    - pre_post_process.py : This module allows us to transform the data of a csv file into a sparse ratings matrix and functions to convert the final made predictions in a csv file of the correct format expected by the Kaggle competition. It contains the following important functions:
							
			- preprocess_data(data):
				Parameters:
					data is the path to the csv file containing the data (data_train.csv or sample_submission.csv).
				Returns:
					the sparse matrix (from library scipy.sparse.lil_matrix) of the ratings associated to the correct user and item. 
					the list of tuples of the form [user id, movie id] for all the known ratings.
			
			- split_data(ratings, p_test = 0.1): 
				Parameters:
					ratings is the sparse ratings matrix.
					p_test is the proportion of ratings that will be in the testing set (the rest in the training set), set by default to 0.1.
				Returns:
					the ratings matrix 
					the corresponding training ratings matrix of type sp.sparse.lil_matrix.
					the correpsonding testing ratings matrix of type sp.sparse.lil_matrix.
			
			- getWantedPredictions(predictions_matrix, ids): 
				Parameters:
					predictions_matrix is the matrix of predictions of type np.matrix.
					ids is the list of tuple ids [user id, movie id] for which we want to extract the predictions.
				Returns:
					a list of all the predictions corresponding to the ids in the list, in the same order as indicated by the csv file: all predictions of movie id 1 first, then movie id 2, etc...
			
			- create_csv_submission(ids, pred, name): 
				Parameters:
					ids is the list of tuples [user id, movie id], typically the ids contained in sample_submission.csv.
					pred is the list of predictions that we obtained with getWantedPredictions(predictions_matrix, ids).
					name is the name to give to the csv file.
				It creates a csv file in the correct format for the submission with the given ids and predictions (they have to be of same lengths).
			
    - collaborative.py : This module contains all the useful methods for the Recommender_Collaborative.ipynb file, which are the following:
			
			- movie_user_predictions(tuple_ids): 
				Parameters:
					tuple_ids is a list of tuples [user id, movie id], typically the ids for which we want to compute the predictions.
				Returns:
					a np.array with the total number of movies as length. Each element of the array is a list of user ids for which we want to compute the prediction for this user and movie: [[user ids to predict for movie 1],[user ids to predict for movie 2],...,[user ids to predict for movie 999]].
			
			- getTopKNeighbors(similarities, K):
				Parameters:
					similarities: list of similarities between a user and all other users
					K: integer
				Returns:
					the list of similarities by only keeping the K biggest ones and setting the others to 0.
			
			- compute_predictions_movie(ratings_matrix, movie, similarity_matrix, users_to_predict, K_best):
				Parameters:
					ratings_matrix: The ratings matrix with dimension movies x users
					movie: Index of movie we want to predict (from 0 to 9999)
					similarity_matrix: The similarity matrix between all users
					users_to_predict: The list of users (indices) for which we want to compute the predictions
					K_best: The number of neighbors to keep for each prediction
				Returns:
					the array of predictions for all the users in users_to_predict and movie indicated in parameters.
				
			- compute_matrix_predictions(ratings_matrix, ratings_to_predict, similarity_matrix, K_best):
				Parameters:
					ratings_matrix: The ratings matrix with dimension movies x users
					ratings_to_predict: The np.array of movies and user ids for which we want to compute the predictions. This is the result of the function movie_user_predictions(tuple_ids).
					similarity_matrix: The similarity matrix between all users
					K_best: The number of neighbors to keep for each prediction
				Returns:
					the full array of predictions for each movie and user specified in ratings_to_predict
	
    - SGD_helpers : This module contains all the useful methods for the Recommender_Factorisation.ipynb file, which are the following:
	
			- init_MF(train, num_features):
				Parameters:
					train: a sparse matrix of ratings, typically the training matrix
					num_features: the number of features (int) for the features matrices
				Returns:
					the user features matrix (dimension num_features x users) with all elements as random numbers in the interval [0, 1.0/num_features].
					the item features matrix (dimension num_features x items) with all elements as random numbers in the interval [0, 1.0/num_features].
	
			- compute_error(data, user_features, item_features, nz):
				Parameters:
					data: a sparse matrix of ratings, typically the test ratings matrix 
					user_features: user_features matrix
					item_features: item_features matrix
					nz: the "non-zero" indices which is a list of tuples (user id, movie id) for which the rating is not zero in data 
				Returns:
					the RMSE between the real ratings of data and the predictions obtained with user_features and item_features.
			
			- matrix_factorization_SGD(train, test, gamma, num_features, lambda_user, lambda_item, num_epochs, user_feat, item_feat, include_test = True):
				Parameters:
					train: the matrix of ratings (scipy.sparse.lil_matrix) on which the SGD is training/learning, typically the training ratings or full ratings for the final submission
					test: the matrix of ratings (scipy.sparse.lil_matrix) on which we test our predictions by using RMSE
					gamma: the learning rate (float)
					num_features: the number of features of the user features and item features matrices (int)
					lambda_user: regression term for users (float)
					lambda_item: regression term for items (float)
					num_epochs: number of epochs to run the SGD algorithm (int)
					user_feat: the user features matrix, typically initialised with init_MF(train, num_features)
					item_feat: the item features matrix, typically initialised with init_MF(train, num_features)
					include_test: if True (default) compute the RMSE between the test ratings and the predictions, if False, don't consider the test ratings matrix. Typically False when we train on all the rating (no test).
				Returns:
					the item features matrix computed and modified with the SGD algorithm in order to minimize the RMSE loss.
					the user features matrix computed and modified with the SGD algorithm in order to minimize the RMSE loss. 
					the test RMSE.
				
	- bias_helpers.py : This module contains all the useful methods for the "Adding biases" part of the Recommender_Factorisation.ipynb file, which are the following:
	
			- computeBiasMatrix(ratings):
				Parameters:
					ratings: the matrix of ratings (scipy.sparse.lil_matrix)
				Returns:
					a lil_matrix where each non zero elements are now r'(u,i) = ratings(u,i) - overall average - bias(u) - bias(i). Thus we can run SGD on this new rating matrix. 
					the overall average (float).
					the list of bias for all users.
					the list of bias for all items.
				
			- predictionsWithBias(item_features, user_features, bias_u, bias_i, mean_rating):
				Parameters:
					item_features: the item features matrix (np.matrix)
					user_features: the user features matrix (np.matrix)
					bias_u: the list of bias for all users -> obtained when running computeBiasMatrix(ratings) 
					bias_i: the list of bias for all items -> obtained when running computeBiasMatrix(ratings) 
					mean_rating: the overall average of ratings -> obtained when running computeBiasMatrix(ratings)
				Returns:
					the matrix of predictions where each prediction is of the form p(u,i) = mean_rating + bias(u) + bias(i) + (item_features @ user_features.T)(u, i)
				
				
    - run.py : This module executes the program in order to get our best submission from the Kaggle competition -> it takes some time to compute (hours) because we are running a grid search by keeping the features matrices giving the minimal loss at each new iteration (warm start) but you will see prints on the terminal as indications.
			   It is commented such that you understand each step we did to obtain the predictions for the best submission.
			   Do not worry about the RunTime warnings when running the run.py.
			   You may change the path of the "data_train.csv" and "sample_submission.csv" files in the load_data(path) function.

To run the run.py module :
- On Mac : open the Terminal, enter your path to the folder where the Python modules are, enter the following command : chmod +x run.py. To execute enter : python run.py
- On Windows: open the Terminal, enter your path to the folder where the Python modules are. To execute enter : python run.py