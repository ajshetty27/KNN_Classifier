import argparse
import csv
import os
from PIL import Image
import math
import numpy as np
from scipy.stats import mode
from scipy.spatial import distance
from numpy import dot
from numpy.linalg import norm
import cv2
from sewar.full_ref import mse, rmse

# This is the classification scheme you should use for kNN
classification_scheme = ['Female','Male','Primate','Rodent','Food']

def kNN(training_data, k, sim_id, data_to_classify):
    processed = [data_to_classify[0] + [student_id]]
    k=k

    #Reads image for image_simality_measures
    def image_reader(path_str):
        img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        width = 256
        height = 256
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized
    #Converts image into np array and flattens to use for comparison
    def convert_image_distances(path_str):
        image = Image.open(path_str)
        arr = np.asarray(image.resize((256,256)))
        arr_main = arr.flatten()
        return arr_main

    # Method to calculate distance using euclidean
    def euclidean(p1,p2):
        dist = np.sqrt(np.sum((p1-p2)**2))
        return dist
    # Method to calculate distance using mse formula implemented by myself
    def mse_myself(p1,p2):
        #Subtract differences from two numpy arrays
        difference_array = np.subtract(p1, p2)
        #Square the differences of the  two numpy arrays
        squared_array = np.square(difference_array)
        #Find the mean of the squared differences
        mse = squared_array.mean()
        return mse
    # Method to calculate distance using rmse formula implemented by myself
    def rmse_myself(p1,p2):
        #Subtract differences from two numpy arrays
        difference_array = np.subtract(p1, p2)
        #Square the differences of two numpy arrays
        squared_array = np.square(difference_array)
        #Find the mean of the squard differences
        mse = squared_array.mean()
        #Find the square root of the mean
        rmse = math.sqrt(mse)
        return rmse

    #loop through each item in data_classify
    for item in data_to_classify[1:]:
        #Reading in path for image
        path_str = item[0]
        #Distances list
        point_dist  = []
        for items in training_data[1:]:
            #Reading in path for image
            path_str_2 =  items[0]
            image_arr1 = convert_image_distances(path_str)
            image_arr2 = convert_image_distances(path_str_2)
            if sim_id == 1:
                distances = mse(image_reader(path_str_2), image_reader(path_str))
            elif sim_id == 2:
                distances = euclidean(image_arr2, image_arr1)
            elif sim_id == 3:
                distances = rmse(image_reader(path_str_2), image_reader(path_str))
            elif sim_id == 4:
                distances  = mse_myself(image_arr2, image_arr1)
            elif sim_id == 5:
                distances  = rmse_myself(image_arr2, image_arr1)


            point_dist.append(distances)
        point_dist_new = np.array(point_dist)
        #Sorting distance list in increasing order with k as the value of neighbours
        dist_check_location = np.argsort(point_dist_new)[:k]
        #holds class of the neighbhours
        classification  = []

        # Iterate through sorted distances and find class of neighbours
        for i in range (len(dist_check_location)):
            classification.append(training_data[dist_check_location[i]][1])

        #Find highest occuring class in the list of classes
        labels = classification
        class_found = mode(labels)
        class_found = class_found.mode[0]
        #Append to processed
        processed.append([item[0], item[1], class_found])
        print([item[0], item[1], class_found])



    return processed

def main():
    opts = parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = read_csv_file(opts['training_data'])
    data_to_classify = read_csv_file(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    result = kNN(training_data, opts['k'], opts['sim_id'], data_to_classify)
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        write_csv_file(out, result)


# Straightforward function to read the data contained in the file "filename"
def read_csv_file(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def write_csv_file(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)
#
def parse_arguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-s', '--sim_id', nargs='?', type=int)
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('training_data')
    parser.add_argument('data_to_classify')
    params = parser.parse_args()

    if params.sim_id < 0 or params.sim_id > 5:
        print('Argument sim_id must be a number from 1 to 5')
        return None

    opt = {'k': params.k,
           'f': params.f,
           'sim_id': params.sim_id,
           'training_data': params.training_data,
           'data_to_classify': params.data_to_classify,
           'mode': params.unseen
           }
    return opt


if __name__ == '__main__':
    main()
