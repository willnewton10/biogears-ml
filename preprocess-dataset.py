

# clear asthma-dataset-1-augmented-1 of its folders

# for each folders in asthma-dataset-1
#   create a folder of the same name in asthma-dataset-1-augmented-1
#   for each csv file in this folder,
#     load the data from the csv file
#     augment the data:
#       - perhaps add small gaussian noise,
#           mean 0
#           variance should be small
#               maybe, for a given column, get the average distance between consecutive rows
#               OR some fraction of the difference between the min and max val in the col
#       - perhaps offset the event by removing a uniform random number of
#           seconds of rows from the beginning of the dataset
#       - perhaps remove rows altogether!
#       - perhaps make multiple copies of same csv with different augmentations. let's say 5


