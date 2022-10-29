'''
A helper script to create a test and train split and move ERDs accordingly
'''
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import os

def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def get_data(path):
    png_list = []
    xml_list = []
    for file in get_files(path):
        if 'xml' in file:
            xml_list.append(file)
        elif 'png' or 'jpg' in file:
            png_list.append(file)
    png_list = np.array(sorted(png_list))
    xml_list = np.array(sorted(xml_list))
    data = np.array([png_list, xml_list]).T
    #print(len(data[0]))
    data = data.reshape(len(data), 2)
    return data

def get_split(data):
    #print(data)
    X_train, X_test = train_test_split(data, test_size=.2) # Used 80-20 split on the data
    print(X_train)
    print('---------------------------')
    print(X_test)
    return X_train, X_test

def move_files(X_train, X_test, path, train_path, test_path):
    for png_file, xml_file in X_train: 
        src_path_png = path + '/' + png_file
        src_path_xml = path + '/' + xml_file
        shutil.copy2(src_path_png, train_path)
        shutil.copy2(src_path_xml, train_path)
    
    for png_file, xml_file in X_test: 
        src_path_png = path + '/' + png_file
        src_path_xml = path + '/' + xml_file
        shutil.copy2(src_path_png, test_path)
        shutil.copy2(src_path_xml, test_path)
    

def main():
    PATH = "/home/andrusha/Desktop/Projects/ERD-Clustering/ERDs/Collection_1"
    TRAIN_PATH = "/home/andrusha/Desktop/Projects/ERD-Clustering/ERDs/train"
    TEST_PATH = "/home/andrusha/Desktop/Projects/ERD-Clustering/ERDs/test"
    data = get_data(PATH)
    X_train, X_test = get_split(data)
    move_files(X_train, X_test, PATH, TRAIN_PATH, TEST_PATH)
    


if __name__ == '__main__':
    main()