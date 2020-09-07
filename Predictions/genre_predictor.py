'''
Script to perform song genre classification based on their lyrics
'''
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix

''' Function to call data cleaning, data preparing and prediction functions '''
def main(csv_data):
	# Remove classes we don't want to use
	classes_to_remove = ['indie', 'edm']
	new_class_df = remove_classes(csv_data, classes_to_remove)
	# Clean the data
	print("Cleaning data...")
	cleaned_df = clean_data(new_class_df)
	# Gather testing and training sets
	print("Preparing Data...")
	train_lyrics, train_genres, test_lyrics, test_genres = prepare_data(cleaned_df)
	# Perform all model predictions and cross-validate using voting classifier
	print("Performing Classifications...")
	voting_classifier(train_lyrics, train_genres, test_lyrics, test_genres)

''' This function removes any classes we don't want to use for predictions '''
def remove_classes(csv_data, classes_to_remove):
	removed = csv_data['genre'].isin(classes_to_remove)
	new_data = csv_data[~removed]
	return new_data

''' This function performs additional data cleansing to existing data set and saves it to a CSV '''
def clean_data(csv_data):
	invalid_songs = []
	# Checks if the song only contains ascii characters. 
	for index, row in csv_data.iterrows():
		if len(row['cleaned_lyrics']) != len(row['cleaned_lyrics'].encode()):
			invalid_songs.append(index) # Store index of invalid song

	# Remove invalid lines based on index
	valid_df = csv_data.drop(invalid_songs)
	# Remove any numbers that may be in dataset
	valid_df['cleaned_lyrics'] = valid_df['cleaned_lyrics'].str.replace('\d+', '')
	# Make all songs the same length
	shortest_song = min(valid_df['cleaned_lyrics'].str.len())
	valid_df['cleaned_lyrics'] = valid_df['cleaned_lyrics'].str[0:shortest_song]
	# Write cleaned data to csv
	cleaned_headers = ['genre', 'song', 'group', 'cleaned_lyrics']
	valid_df.to_csv('Data/CleanedGenres.csv', columns=cleaned_headers)
	return valid_df

''' This function gets testing and training sets for predicitons '''
def prepare_data(csv_data):
	# Make sure same number of songs from each sub-genre is used
	genre_cnts = csv_data.groupby("genre")["song"].count()
	min_genre_cnt = min(genre_cnts)
	genre_splits = csv_data.groupby("genre").head(min_genre_cnt)
	genre_splits = genre_splits.sample(frac=1)
	# Split data: 80% training, 20% testing
	train_idxs = np.random.rand(len(genre_splits)) < 0.80
	train_data = genre_splits[train_idxs]
	test_data = genre_splits[~train_idxs]
	# Get cleaned lyrics and associated GROUP
	train_lyrics = train_data["cleaned_lyrics"]
	train_genres = train_data["group"]
	test_lyrics = test_data["cleaned_lyrics"]
	test_genres = test_data["group"]
	return(train_lyrics, train_genres, test_lyrics, test_genres)

''' Multinomial naive Bayes classification model '''
def multinomNB_model(train_lyrics, train_genres, test_lyrics, test_genres):
	# Define pipeline
	multinom_pipe = Pipeline([
		('tfidfv', TfidfVectorizer(ngram_range=(1,2), smooth_idf=True, use_idf=False, max_df=0.5)),
		('mnb', MultinomialNB(alpha=0.02864, fit_prior=False))])
	# Fit an predict model
	multinom_pipe.fit(train_lyrics, train_genres)
	multinom_preds = multinom_pipe.predict(test_lyrics)

	# Get accuracies for this model
	print("Multinomial Naive Bayes Results: ")
	get_acc(multinom_preds, multinom_pipe, test_lyrics, test_genres, "Multinomial Naive Bayes Confusion Matrix", 'MNB')
	# Return for voting classifier
	return multinom_pipe

''' Bernoulli naive Bayes classification model '''
def bernoulliNB_model(train_lyrics, train_genres, test_lyrics, test_genres):
	# Define pipelie
	bern_nb_pipe = Pipeline([
		('tfidfv', TfidfVectorizer(ngram_range=(1,2), smooth_idf=True, use_idf=False, max_df=0.5, binary=True)),
		('bnb', BernoulliNB(alpha=0.02864, fit_prior=False))])
	# Fit and predict model
	bern_nb_pipe.fit(train_lyrics, train_genres)
	bern_preds = bern_nb_pipe.predict(test_lyrics)
	# Below is an example of finding the best parameters for various features of this model
	# Depending on how many you are looking for, this can take a very long time
	'''
	grid_params = {
	  	'bnb__alpha': np.linspace(0.0001, 0.1, 8),
	}
	get_best_params(bern_nb_pipe, grid_params, train_lyrics, train_genres)
	'''
	# Get accuracies
	print("Bernoulli Naive Bayes Results: ")
	get_acc(bern_preds, bern_nb_pipe, test_lyrics, test_genres, "Bernoulli Naive Bayes Confusion Matrix", 'BNB')
	return bern_nb_pipe

''' Stochastic gradient descent model '''
def SGDC_model(train_lyrics, train_genres, test_lyrics, test_genres):
	# Define pipelie
	sgdc_lin_pipe = Pipeline([
		('tfidfv', TfidfVectorizer(ngram_range=(1,2), smooth_idf=True, use_idf=False, max_df=0.5)),
		('sclf', SGDClassifier(loss='modified_huber', penalty='l2',
							alpha=0.0001, random_state=42,
							max_iter=400, tol=None, class_weight='balanced'))
	])
	# Fit and predict model
	sgdc_lin_pipe.fit(train_lyrics, train_genres)
	sgdc_preds = sgdc_lin_pipe.predict(test_lyrics)
	# Get accuracies
	print("Stochastic Gradient Descent Results: ")
	get_acc(sgdc_preds, sgdc_lin_pipe, test_lyrics, test_genres, "Stochastic Gradient Descent Confusion Matrix", 'SGD')
	return sgdc_lin_pipe

''' Voting classifier is defined here to create all models, perform individual predictions, and voting predictions '''
def voting_classifier(train_lyrics, train_genres, test_lyrics, test_genres):
	# Stores each model
	estimators = []
	# Create stochastic gradient descent model
	sgdc_lin_pipe = SGDC_model(train_lyrics, train_genres, test_lyrics, test_genres)
	estimators.append(('sdcglin', sgdc_lin_pipe))
	# Create multinomial naive Bayes model
	multinom_pipe = multinomNB_model(train_lyrics, train_genres, test_lyrics, test_genres)
	estimators.append(('multinom', multinom_pipe))
	# Create Bernoulli naive Bayes model
	bern_pipe = bernoulliNB_model(train_lyrics, train_genres, test_lyrics, test_genres)
	estimators.append(('bernoulli', bern_pipe))
	# Create ensemble of models (voting classifier) and fit it to training data
	print("Making Voting Predictions...")
	ensemble = VotingClassifier(estimators, voting='hard', verbose=True)
	ensemble.fit(train_lyrics, train_genres)
	# Voting prediciton
	vote_predicted = ensemble.predict(test_lyrics)
	# Get voting accuracies
	print("Results using cross validation of both models: ")
	get_acc(vote_predicted, ensemble, test_lyrics, test_genres, "Voting Classifier Confusion Matrix", 'VC')

''' Based on a model's pipeline and grid parameters, find the optimal parameters'''
def get_best_params(model_pipeline, grid_params, train_lyrics, train_genres):
	clf = GridSearchCV(model_pipeline, grid_params)
	clf.fit(train_lyrics, train_genres)

	print("Model: ", model_pipeline)
	print("Best Score: ", clf.best_score_)
	print("Best Parameters: ", clf.best_params_)

''' 
Gets accuracies given predicted model, a fitted classifier, test data, and title for confusion matrix
Creates a plot of the confusion matrix and a classification report for overall and class accuracies
'''
def get_acc(predicted, classifier, test_lyrics, test_genres, conf_mat_title, fname):
	# Class names
	class_names = ['Blues', 'Rock', 'Rap', 'Country']
	# Plot confusion matrix
	disp = plot_confusion_matrix(classifier, test_lyrics, test_genres,
							display_labels=class_names,
							cmap=plt.cm.Blues,
							normalize=None)

	disp.ax_.set_title(conf_mat_title)
	# plt.show()
	save_name = 'Plots/' + fname + '.png'
	plt.savefig(save_name)
	# Classification report to get accuracies
	class_report = classification_report(test_genres, predicted, target_names=class_names)
	print(class_report)

''' Runs the script given a path to data file '''
if __name__ == "__main__":
	# Load data in to Pandas DF
	data_location = "Data/Genres.csv"
	data = pd.read_csv(data_location)
	main(data)

