import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import isfile, join


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder,)
        except:
            print(f)
            assert False


if __name__ == "__main__":
    df = pd.read_csv("data/_annotations.csv")
    filenames = df["filename"].unique()

    images = [join('/home/milad/Downloads/self_driving_pics', 'export', x) for x in filenames]
    annotations = [join('data', 'labels', x) for x in os.listdir('data/labels') if isfile(join('data', 'labels', x))]

    images.sort()
    annotations.sort()
    k = 0
    for i in range(len(images)):
        if images[i][:-3] != annotations[i].replace('labels', 'images')[:-3]:
            print(images[i])
            k +=1

    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
                                                                                    random_state=1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5, random_state=1)
    move_files_to_folder(train_images, 'data/images/train')
    move_files_to_folder(val_images, 'data/images/val/')
    move_files_to_folder(test_images, 'data/images/test/')
    move_files_to_folder(train_annotations, 'data/labels/train/')
    move_files_to_folder(val_annotations, 'data/labels/val/')
    move_files_to_folder(test_annotations, 'data/labels/test/')
