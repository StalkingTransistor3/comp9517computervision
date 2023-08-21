import sys
from ultralytics import YOLO
import json
import os
import shutil
import math 
from functools import reduce
TEST_FRACTION = .1


def format_dataset(dataset_path, remake=False):
    """formats the dataset for YOLO. returns the path to the formatted dataset"""
    if not remake:
        return
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    if os.path.exists(script_path + "\\yolo_formatted_dataset"):
        shutil.rmtree(script_path + "\\yolo_formatted_dataset")
    
    os.mkdir(script_path + "\\yolo_formatted_dataset")
    # Formatting training data
    train_path = script_path + "\\yolo_formatted_dataset\\train"
    os.mkdir(train_path)
    os.mkdir(train_path + "\\images")
    os.mkdir(train_path + "\\labels")
    with open(f"{dataset_path}\\train_annotations") as f:
        data = f.read()
        train_data = json.loads(data)
        format_yolo_data(dataset_path + "\\train\\train", train_path, train_data)

    # formatting val data
    val_path = script_path + "\\yolo_formatted_dataset\\val"
    os.mkdir(val_path)
    os.mkdir(val_path + "\\images")
    os.mkdir(val_path + "\\labels")
    with open(f"{dataset_path}\\valid_annotations") as f:
        data = f.read()
        val_data = json.loads(data)
        format_yolo_data(dataset_path + "\\valid\\valid", val_path, val_data)

    with open(script_path + "\\yolo_formatted_dataset\\dataset.yaml", 'w') as yaml_file:
        yaml_file.write(f"path: {script_path}\\yolo_formatted_dataset\\\n")
        yaml_file.write("train: train\n")
        yaml_file.write("val: val\n")
        # yaml_file.write("test: test\n")
        yaml_file.write("nc: 3\n")
        yaml_file.write("names: \n  0: background\n  1: penguin\n  2: turtle\n")
    

def format_yolo_data(images_path, dest, dict, split_start=0, split_end=-1, imgsz=640):
    for image in os.listdir(images_path)[split_start:split_end]:
        img_path = images_path + "\\" + image
        shutil.copy(img_path, f"{dest}\\images\\{image}")
        file_data = next(filter(lambda x: x["image_id"] == int(image[9:12]), dict))
        with open(f"{dest}\\labels\\{image.split('.')[0]}.txt", 'w') as label_file:
            xmin, ymin, width, height = file_data['bbox']
            xCenter = xmin + width/2
            yCenter = ymin + height/2
            label_file.write(f"{file_data['category_id']} {xCenter/imgsz} {yCenter/imgsz} {width/imgsz} {height/imgsz}")


def test_model(model, test_path, file=None):
    images_list = list(map(lambda x: test_path+"\\images\\"+x, os.listdir(test_path + "\\images")))
    res_list = model.predict(images_list, max_det=1, save_conf=True)
    distance_list = []
    iou_list = []
    for res in res_list:
        img_name = res.path.split('\\')[-1]
        f = open(f"{test_path}\\labels\\{img_name.split('.')[0]}.txt", 'r')
        data = list(map(lambda x: float(x), f.read().split()))
        box = res.boxes.xyxy
        if (len(box) < 0):
            continue
        box = box[0]
        distance = math.sqrt(
            (data[1] * res.orig_shape[0] - (box[0] + box[2]) / 2) ** 2 + 
            (data[2] * res.orig_shape[1] - (box[1] + box[3]) / 2) ** 2
        )
        distance_list.append(distance)

        data_xyxy = [
            (data[1] - data[3]/2) * res.orig_shape[0], 
            (data[2] - data[4]/2) * res.orig_shape[1], 
            (data[1] + data[3]/2) * res.orig_shape[0], 
            (data[2] + data[4]/2) * res.orig_shape[1],
        ]
        iou_list.append(get_i_o_u(data_xyxy, box))
        # print(f"data: {data_xyxy}, box: {box}, dx: {data[1] * res.orig_shape[0]}, iou: {get_i_o_u(data_xyxy, box)}")
    
    if file == None:
        print(f"Average bbox distance: {get_mean(distance_list)}")
        print(f"Bbox distance standard deviation: {get_standard_deviation(distance_list)}")
        print(f"Average bbox iou: {get_mean(iou_list)}")
        print(f"Bbox iou standard deviation: {get_standard_deviation(iou_list)}")
    else:
        file.write(f"Average bbox distance: {get_mean(distance_list)}\n")
        file.write(f"Bbox distance standard deviation: {get_standard_deviation(distance_list)}\n")
        file.write(f"Average bbox iou: {get_mean(iou_list)}\n")
        file.write(f"Bbox iou standard deviation: {get_standard_deviation(iou_list)}\n")


def get_standard_deviation(array):
    mean = get_mean(array)
    return math.sqrt(get_mean(list(map(lambda x: (x - mean) ** 2, array))))


def get_mean(array):
    return sum(array) / len(array)


def get_i_o_u(bbox1, bbox2):
    """takes two bounding boxes of the form x1, y1, x2, y2 (the corners)
    and calculates the intersection over union"""
    return get_intersection(bbox1, bbox2) / get_union(bbox1, bbox2)


def get_intersection(bbox1, bbox2):
    x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    return x_overlap * y_overlap


def get_union(bbox1, bbox2):
    return (
        (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
      + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
      - get_intersection(bbox1, bbox2)
    )


def get_accuracy(confusion_matrix):
    """assuming positive is penguin and negative is turtle"""
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[2][2]
    fp = confusion_matrix[1][2]
    fn = confusion_matrix[2][1]
    return (tp + tn) / (tp + tn + fp + fn)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Incorrect format. to run, try \"py project_yolo.py [path_to_dataset]\" or \"py project_yolo.py [path_to_weights]\"")
    weights = 'yolov8n.pt'
    train = True
    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    if sys.argv[1][-3:] == '.pt':
        weights = sys.argv[1]
        train = False
    else: 
        dataset_path = sys.argv[1]
        format_dataset(dataset_path, True)
        if len(sys.argv) == 3 and sys.argv[2] == "-f":
            exit()

    model = YOLO(weights)
    if train:
        model.train(data=f'{script_path}\\yolo_formatted_dataset\\dataset.yaml', epochs=150, imgsz=640, device=0, patience=50)
    metrics = model.val(max_det=1, dnn=True)
    results = metrics.results_dict
    precision, recall = results['metrics/precision(B)'], results['metrics/recall(B)']
    with open(f"{str(metrics.save_dir)}\\results.txt", "w") as f:
        test_model(model, f"{script_path}\\yolo_formatted_dataset\\val", f)
        f.write(f"Accuracy: {get_accuracy(metrics.confusion_matrix.matrix)}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {2 * precision * recall / (precision + recall)}\n")