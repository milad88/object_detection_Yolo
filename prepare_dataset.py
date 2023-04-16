import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


if __name__ == "__main__":
    df = pd.read_csv("data/_annotations.csv")
    filenames = df["filename"].unique()

    images = [os.path.join('data', 'images', x) for x in filenames]
    annotations = [os.path.join('data', 'annotations', x) for x in os.listdir('data/annotations') if x[-3:] == "txt"]

    images.sort()
    annotations.sort()
    for f in images:
        if f.replace("jpg", "txt") in annotations:
            print(f)
    print("finsdd")
    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2,
                                                                                    random_state=1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations,
                                                                                  test_size=0.5, random_state=1)
    move_files_to_folder(train_images, 'data/images/train')
    move_files_to_folder(val_images, 'data/images/val/')
    move_files_to_folder(test_images, 'data/images/test/')
    move_files_to_folder(train_annotations, 'data/annotations/train/')
    move_files_to_folder(val_annotations, 'data/annotations/val/')
    move_files_to_folder(test_annotations, 'data/annotations/test/')
