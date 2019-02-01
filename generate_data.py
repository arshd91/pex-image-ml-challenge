import cv2
import h5py
import sys
import glob
import os
import random
import numpy as np


if __name__=='__main__':

    if len(sys.argv) < 3:
        print("usage: python3 generate_data.py [path-to-images] [path-to-data-dump] [extension]")
        exit()

    img_dir_path = sys.argv[1]
    indoor_path = img_dir_path+"/indoor"
    outdoor_path = img_dir_path+"/outdoor"

    dataset_path = sys.argv[2]

    ext = sys.argv[3]

    #extract all files from data folder
    indoor_files = sorted(glob.glob(os.path.join(indoor_path, "*."+ext)))
    outdoor_files = sorted(glob.glob(os.path.join(outdoor_path, "*."+ext)))

    indoor_tuples = [ (f,1) for f in indoor_files]
    outdoor_tuples = [ (f,0) for f in outdoor_files]
    total_tuples = indoor_tuples + outdoor_tuples

    n_indoor = len(indoor_files)
    n_outdoor = len(outdoor_files)
    total = n_indoor + n_outdoor

    # train/dev/test splits
    train_split = 0.9
    dev_split = 0.1
    test_split = 0.0

    random.shuffle(total_tuples)

    train_size = int(len(total_tuples) * train_split)
    train_cut = train_size

    dev_size = int(len(total_tuples) * dev_split)
    dev_cut = train_cut + dev_size

    test_size = total - train_size + dev_size
    test_cut = dev_cut + test_size

    r,g,b = [],[],[]

    for i, record in enumerate(total_tuples):
        print("processing file {} with label {}".format(record[0],record[1]))

        img = cv2.imread(record[0])
        small_img = cv2.resize(img, (320, 240))

        # cv2.imshow('image', small_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if i < train_cut:
            datatype = 'train'
            idx = i
            r.append(small_img[:,:,0])
            g.append(small_img[:,:,1])
            b.append(small_img[:,:,2])

        elif i >= train_cut and i < dev_cut:
            datatype = 'dev'
            idx = i - train_size

        else :
            datatype = 'test'
            idx = i - train_size - dev_size

        if not os.path.exists(dataset_path + '/' + datatype + '/0'):
            os.makedirs(dataset_path + '/' + datatype + '/0')

        if not os.path.exists(dataset_path + '/' + datatype + '/1'):
            os.makedirs(dataset_path + '/' + datatype + '/1')

        img_write_path = dataset_path + '/' + datatype + '/' + str(record[1]) + '/img_' + str(idx) + '_' + str(record[1]) + '.jpg'
        cv2.imwrite(img_write_path, small_img)

    print(np.mean(r)/255.0, np.std(r)/255.0)
    print(np.mean(g)/255.0, np.std(g)/255.0)
    print(np.mean(b)/255.0, np.std(b)/255.0)

















