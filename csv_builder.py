'''
This module creates a CSV to later be read by spark as a DataFrame
Ths CSV is stored in the script directory, and searches each genre
directory for lyrics from songs in that genre
'''

import random
import csv
from os import listdir, getcwd, chdir, mkdir, walk
from os.path import isfile, join, dirname, basename, isdir
from math import ceil
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import desc, col


# Sets up the CSV for data to be added
def main():
	genre_dirs = [x[0] for x in walk(main_dir)][2:] # Get all directories for each genre
	csv_headers = ['genre', 'song', 'group', 'lyric_lines', 'cleaned_lyrics', 'line_cnt', 'verse_chorus_cnt', 
				'pos_sentiment', 'neut_sentiment', 'neg_sentiment', 'total_sentiment'] # Headers for csv

	create_lyric_csv(csv_file, csv_headers) # Create empty csv with headers

	# Get the genre with least songs, this is how many songs from each genre will be added
	genre_sizes = sc.parallelize(genre_dirs).map(lambda x: least_files(x)).sortBy(lambda x: x)
	max_rows = genre_sizes.collect()[0]

	# Add songs from each genre to the csv
	created_rows = sc.parallelize(genre_dirs).map(lambda x: read_genre_songs(x, max_rows))
	total_rows = created_rows.reduce(lambda x, y: x+y) 
	# Add training/testing column to csv with random values
	create_training_col(total_rows)

# Creates empty csv file
def create_lyric_csv(csv_fname, csv_headers):
	with open(csv_fname, 'w', newline='') as lyric_csv:
		writer = csv.writer(lyric_csv)
		writer.writerow(csv_headers)


# Gets number of files in directory
def least_files(g_dir):
	return len([s_file for s_file in listdir(g_dir) if isfile(join(g_dir, s_file))])

# Loops through each song in a genre and cleans the song
def read_genre_songs(genre, max_rows):
	chdir(main_dir)
	chdir(genre)
	# Only want text files
	genre_songs = [f for f in listdir(genre) if isfile(join(genre, f)) and f.endswith(".txt")]
	genre_name = basename(genre)
	# Loop through each song and clean it
	genre_row = 0
	for song_path in genre_songs:
		genre_row = clean_song_data(song_path, genre_name, genre_row, max_rows)
	# Number of rows added for each genre (used to get total rows)
	return genre_row

# Cleans/validates each song before adding to csv
def clean_song_data(song, genre_name, genre_row, max_rows):
	# Make sure adding correct number of songs for this genre
	if genre_row <= max_rows:
		# Total verse/chrous counts and line counts
		total_vc = 0
		valid_lines = []
		# Open song and loop through lines
		with open(song, 'r') as opened_song:
			song_lines = opened_song.read().splitlines()
			for i, orig_line in enumerate(song_lines):
				# Increment total number of verses and choruses
				if is_verse_or_chorus(song_lines, orig_line, i):
					total_vc += 1

				if is_valid_line(orig_line) == True:
					valid_lines.append(orig_line)

				'''
				# Remove invalid lines
				if (orig_line == '') or ('[' and ']' in orig_line) or ('(' and ')' in orig_line):
					del song_lines[i]
				else:
					valid_lines.append(orig_line)
				'''

			total_lines = len(valid_lines) # Total valid lines in song
			combined_lines = ' '.join(valid_lines).lower() # String with all song lyrics
			all_lyrics = prepare_data(combined_lines) # Lemmatized words

		# Open csv
		with open(csv_file, 'a', newline='', encoding='UTF-8') as opened_csv:
			group = {
				"blues" : '1',
				"rnb" : '1',
				"rock" : '2',
				"heavy_metal" : '2',
				"punk_rock" : '2',
				"rap" : '3',
				"old_school_rap" : '3',
				"indie" : '4',
				"edm" : '5',
				"country" : '6'}
			# Create group ID
			group_id = group.get(genre_name)
			writer = csv.writer(opened_csv)
			# Make sure song is at least 400 chars before adding
			if len(all_lyrics) >= 400:
				# Get sentiment values
				pos, neu, neg, tot = get_sentiment(valid_lines)
				# Add row to CSV
				writer.writerow([genre_name, song, group_id, ' '.join(valid_lines), all_lyrics.replace(r'[^\w\s]', ''), total_lines, total_vc, pos, neu, neg, tot])
				genre_row += 1
	# Keep track of how many records added
	return genre_row

# Checks if line is valid to be analyzed
def is_valid_line(orig_line):
	lower_line = orig_line.lower()
	invalid_words = ['verse', 'chorus', 'hook', 'couplet', 'pont', 'intro', 'outro', 
					'refrain', 'bridge', 'interlude', 'pre-hook', 'post-hook', 'instrumental',
					'guitar solo']

	if (orig_line.isspace()) or ('[' and ']' in orig_line) or ('(' and ')' in orig_line):
		return False
	elif any(x in lower_line for x in invalid_words) and len(lower_line.split()) <= 5:
		return False
	else: 
		return True

# Determines if new section is verse, chorus, etc
def is_verse_or_chorus(song_file, song_line, idx):
	# Break in song file, potenitally verse/chrous
	if song_line == '':
		# If next line says verse or chorus search one line further
		if '[' and ']' in song_file[idx+1]:
			idx += 1
		# If next 3 lines are not empty, classify as verse/chorus
		next_lines = [l for l in song_file[idx+1:idx+4] if l != '']
		if len(next_lines) >= 3:
			return True
	return False

# Removes stopwords, tokenizes, and lemmatizes lyrics
def prepare_data(song_lines):
	from nltk.corpus import stopwords
	stop_words = set(stopwords.words("english"))
	
	lyrics = word_tokenize(song_lines) # list of individual words from lyrics
	clean_lyrics = [word for word in lyrics if word.isalnum()] # list with words only containing alphanumeric characters
	no_stop_words = [w for w in clean_lyrics if not w in stop_words] # new lyrics list without stop words

	lemmatizer = WordNetLemmatizer() # var to lemmatize words
	lemmatized_lyrics = [lemmatizer.lemmatize(w) for w in no_stop_words] # lemmatizing lyrics
	all_lyrics = ' '.join(lemmatized_lyrics) # Combine valid lines back to one song
	return all_lyrics

# Calculates sentiment values
def get_sentiment(song_lines):
	num_pos = num_neu = num_neg = comp_val = 0
	comp_cnt = 0
	sid = SentimentIntensityAnalyzer()
	# Loop through song lines
	for line in song_lines:
		line = line.replace(r'[^\s]', '')
		if len(line.split(' ')) >= 3:
			# Compute polarity
			comp_cnt += 1
			comp = sid.polarity_scores(line)
			comp = comp['compound']
			comp_val += comp
			# Increment associated polarity count
			if comp >= 0.5:
				num_pos += 1
			elif comp > -0.5 and comp < 0.5:
				num_neu += 1
			else:
				num_neg += 1

	# Compute average sentiment for each category
	num_total = num_pos + num_neu + num_neg
	if num_total == 0:
		return (0.0, 0.0, 0.0, 0.0)
	else:
		percent_pos = (num_pos/float(num_total))*100
		percent_neu = (num_neu/float(num_total))*100
		percent_neg = (num_neg/float(num_total))*100
		percent_total = (comp_val/float(comp_cnt))*100
		return (percent_pos, percent_neu, percent_neg, percent_total)

# Creates csv column to keep track of training and testing: Train=1, Test=0
def create_training_col(num_rows):
	# 80% train, 80% test
	train_size = [1]*(ceil(num_rows * 0.8))
	test_size = [0]*(num_rows - len(train_size))
	is_train_data = train_size + test_size
	random.shuffle(is_train_data) # Randomize row it is in
	# Add training column to csv
	new_csv = pd.read_csv(csv_file)
	new_col = pd.DataFrame({"isTraining": is_train_data})
	new_csv = new_csv.merge(new_col, left_index=True, right_index=True)
	new_csv.to_csv(csv_file, index=False)

# Runs program
if __name__ == "__main__":
	conf = SparkConf().setAppName("FinalProjCSV").setMaster("local[*]")#("spark://192.168.1.109:7077")
	sc = SparkContext(conf = conf)
	spark = SparkSession(sparkContext=sc)
	# Main dir is where all genres are located
	main_dir = getcwd()
	csv_fname = "Genres.csv"
	csv_file = join(main_dir, csv_fname)
	main()
	sc.stop()

