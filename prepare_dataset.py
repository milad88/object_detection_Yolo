import os

import pandas as pd

from config import class_name_to_id_mapping


# Dictionary that maps class names to IDs


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5_from_dict(info_dict):
    print_buffer = []

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = (b["xmax"] - b["xmin"])
        b_height = (b["ymax"] - b["ymin"])

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save
    save_file_name = os.path.join("data", "annotations", info_dict["filename"].replace("png", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


def convert_to_yolov5_from_df(df: pd.DataFrame, filename: str):
    print_buffer = []

    # For each bounding box
    for i, row in df.iterrows():  # for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[row["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (row["xmin"] + row["xmax"]) / 2
        b_center_y = (row["ymin"] + row["ymax"]) / 2
        b_width = (row["xmax"] - row["xmin"])
        b_height = (row["ymax"] - row["ymin"])

        # Normalise the co-ordinates by the dimensions of the image

        image_h = row["width"]
        image_w = row["height"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save
    save_file_name = os.path.join("data", "annotations", filename.replace("jpg", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


if __name__ == "__main__":
    df = pd.read_csv("~/Downloads/self_driving_pics/export/_annotations.csv")
    filenames = df["filename"].unique()
    for f in filenames:
        convert_to_yolov5_from_df(df[df['filename']==f], f)
    print("something")
