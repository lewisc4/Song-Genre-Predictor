'''
This module scrapes song lyrics from genius.com based
on files containting artist names for each genre.
Songs are stored as individual files in their respective
genre directories
'''

from os import listdir, getcwd, chdir, mkdir
from os.path import isfile, join, dirname, basename, isdir
from itertools import zip_longest
import re
import lyricsgenius
import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import desc, col

# 1.)
# Initializes genius with specified search parameters, given an access token
def initialize_genius(access_token):
	genius = lyricsgenius.Genius(access_token)
	genius.verbose = True
	genius.skip_non_songs = True
	genius.excluded_terms = ["Freestyle", "(Remix)", "(Live)"]
	return genius

# 2.)
# Sets the genre data and gathers information based on it
def main():
	main_path = getcwd()
	genre_data = get_genre_artists(main_path)
	get_genre_data(genre_data, main_path)

# 3.)
# Returns dictionary with genres as keys and list of artists in that genre as values
def get_genre_artists(main_path):
	artists_dir = join(main_path, "all_artists")
	genre_artists = [join(artists_dir, f) for f in listdir(artists_dir) if isfile(join(artists_dir, f)) and f.endswith(".txt")]
	genre_data = {}

	for genre in genre_artists:
		genre_name = basename(genre).strip(".txt")
		with open(genre, 'r') as artists_file:
			genre_data[genre_name] = artists_file.read().split(',')
	return genre_data

# 4.) 
# Loops through each genre to extract artist data
def get_genre_data(genre_data, data_location):
	for genre, artist_list in genre_data.items():
		chdir(data_location)
		create_genre_dir(genre)
		get_artist_data(artist_list)

# 5.)
# Creates a directory to store each song in specified genre
def create_genre_dir(genre):
	save_dir = join(getcwd(), genre)
	if isdir(save_dir) == False:
		mkdir(save_dir)
	chdir(save_dir)

# 6.)
# Loops through each artist in a genre and gets specified number of their songs
def get_artist_data(artist_list):
	split_artists = list(grouper(artist_list, 8))
	for artist_group in split_artists:
		artist_rdd = sc.parallelize(artist_group, 16)

		searched_rdd = artist_rdd.map(lambda x: genius_search(x))
		completed_searches = searched_rdd.collect()

		for artist_data in completed_searches:
			save_song_lyrics(artist_data)

# 7.)
# Splits a list into even chunks, used for rdd
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

# 8.)
# Function called by rdd to perform parallel searching
def genius_search(artist):
	access_token = "_n7LD7-iQnVEAAmBv7LPR6XlC211L2a3R2lQMtxpOeticv2z6_TuWc1xhLY9tuJJ"
	genius = initialize_genius(access_token)
	if artist is not None:
		try:
			artist_data = genius.search_artist(artist, max_songs=30, sort="popularity")
		except:
			print("Error for: " + artist)
		else:
			return artist_data

# 9.)
# Saves the lyrics of a song to a text file to be parsed
def save_song_lyrics(artist_data):
	if artist_data is not None:
		for song in artist_data.songs:
			try:
				song_name_line = str(song).split("\n")[0]
				song_file_name = re.findall(r'"([^"]*)"', song_name_line)[0]
				song.save_lyrics(extension="txt", filename=song_file_name, overwrite=True)
			except:
				print("Cannont save song")

# Runs the script
if __name__ == "__main__":
	# Initialize spark context
	conf = SparkConf().setAppName("FinalProj").setMaster("local[*]")#("spark://192.168.1.109:7077")
	sc = SparkContext(conf = conf)
	spark = SparkSession(sparkContext=sc)
	main()
	sc.stop()

