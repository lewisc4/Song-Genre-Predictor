def special_case:
    #special case
    rdd = sc.textFile("cleanup.txt").flatMap(lambda line: line.split("\n")).filter(lambda x: len(x) > 3 and len(x) < 40 and not any(c.isdigit() for c in x) and not "The " in x)
    rdd1 = rdd.filter(lambda x: not "added" in x and not "listed on" in x).collect()
    print(rdd1)

    with open("./{}".format("edm.txt"), "w") as outfile:
    outfile.write("\n".join(rdd1))

    #for i in range(0,len(rdd)):
    #print(rdd[i] + ',', end='')

    print("")
    print(len(rdd1))

def contains_paragraphs:
    rdd = sc.textFile("cleanup.txt").flatMap(lambda line: line.split("\n")).filter(lambda x: len(x) > 0 and len(x) < 40).collect()
    for i in range(0,len(rdd)):
    print(rdd[i] + ',', end='')

    print("")
    print(len(rdd))

def newline_separated:
    rdd = sc.textFile("cleanup.txt").flatMap(lambda line: line.split("\n")).collect()

    for i in range(0,len(rdd)):
    rdd[i].replace("\"", "")
    rdd[i].replace(".", "")
    print(rdd[i] + ',', end='')

def number_separated:
    rdd = sc.textFile("cleanup.txt").flatMap(lambda line: line.split("\n")).collect()

    for i in range(1,len(rdd) + 1, 2):
    rdd[i].replace("\"", "")
    rdd[i].replace(".", "")
    print(rdd[i] + ',', end='')

def numbered_names:
    #have to go in file first and make the first 10 numbers 2 digits
    rdd = sc.textFile("cleanup.txt").flatMap(lambda line: line.split(",")).collect()
    for i in range(0, len(rdd)):
    l = len(rdd[i]) - 5
    rdd[i] = rdd[i][-l:]
    print(rdd[i] + ',', end='')
    print("")
    print(len(rdd))

#function to print number of duplicates removed
#f is filename
#o is original item count
#n is new item count
def print_num_of_dupes(f, o, n):
  #print file name
  print(f)
  #print old count
  print(o)
  #print new count
  print(n)

#function to create an rdd for each file in fs
#fs is list containing genre file names as strings
def create_genre_rdds(fs):
  genre_rdds = dict()  #empty dictionary
  #for file in filelist(fs)
  for f1 in fs:
    #rdd containing content of f1
    fn1 = sc.textFile(f1).flatMap(lambda line: line.lower().split(",")).filter(lambda x: len(x) > 0) 
    #switch to/from array to get rid of special characters
    s = fn1.collect()
    for t in range(0, len(s)):
      s[t] = s[t].replace("\"", "").replace(".", "").replace("'", "").replace("-", "")
    fn = sc.parallelize(s)

    nd = fn.distinct() #new rdd with distinct values from fn
    genre_rdds[f1] = nd #add rdd of distinct values to dictionary
    print_num_of_dupes(f1, fn.count(), nd.count()) #print no of duplicates removed

  #return list of rdds
  return genre_rdds

#function to print all artists in each text file
#grdds: dict
#key: filename
#value: rdd of artists
def print_artists(grdds):
  #iterating dictionary
  for (fn, rdd) in grdds.items():
    gs = rdd.collect()
    print(fn + "    Count: {}".format(len(gs)))
    #overwriting file with list containing no dupes
    with open("./{}".format("test.txt"), "w") as outfile:
      outfile.write(",".join(gs))
    #printing all artists in an rdd, separated by commas
    for g in gs:
      print(g + ',', end = '')
    print("")


#list containing all genre file names
#genre_files = []
genre_files = ["punk_rock.txt", "heavy_metal.txt", "old_school_rap.txt", "rap.txt", "rock.txt", "country.txt", "rnb.txt", "blues.txt", "edm.txt", "indie.txt"]
#dictionary containing genre file names and an rdd of artists in that genre
genre_dict = create_genre_rdds(genre_files)
print_artists(genre_dict)