import os
import shutil
import xml.etree.ElementTree as ET
import argparse
import sys
import yaml

parser = argparse.ArgumentParser(description="Convert VOC dataset (with correct directory structure) to YOLO dataset")
parser.add_argument("dataset_dir", metavar="dataset-dir", help="Directory of VOC dataset")
parser.add_argument("--output", help="Output directory (otherwise it will overwrite dataset)")
args = parser.parse_args()

args.dataset_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.dataset_dir)))

if not args.output:
    args.output = args.dataset_dir
else:
    args.output = os.path.abspath(os.path.expanduser(os.path.expandvars(args.output)))
    if not os.path.exists(args.output):
        os.mkdir(args.output)

if not os.path.exists(args.dataset_dir):
    print("Please supply a valid directory", file=sys.stderr)
    exit = True
if not os.path.exists(os.path.join(args.dataset_dir, "Annotations")):
    print("Missing Annotations directory", file=sys.stderr)
    exit = True
if not os.path.exists(os.path.join(args.dataset_dir, "ImageSets")):
    print("Missing ImageSets directory", file=sys.stderr)
    exit = True
if not os.path.exists(os.path.join(args.dataset_dir, "JPEGImages")):
    print("Missing JPEGImages directory", file=sys.stderr)
    exit = True
if not os.path.exists(os.path.join(args.dataset_dir, "ImageSets", "Main")):
    print(f"Missing {os.path.join(args.dataset_dir, 'ImageSets', 'Main')} directory", file=sys.stderr)
    exit = True
if not os.path.exists(os.path.join(args.dataset_dir, "labels.txt")):
    print("Missing labels.txt file", file=sys.stderr)
    exit = True
if exit == True:
    sys.exit(1)


if os.path.exists(os.path.join(args.output, 'train')):
    shutil.rmtree(os.path.join(args.output, 'train'))
os.mkdir(os.path.join(args.output, 'train'))
if os.path.exists(os.path.join(args.output, 'test')):
    shutil.rmtree(os.path.join(args.output, 'test'))
os.mkdir(os.path.join(args.output, 'test'))
if os.path.exists(os.path.join(args.output, 'val')):
    shutil.rmtree(os.path.join(args.output, 'val'))
os.mkdir(os.path.join(args.output, 'val'))


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(annotation_file, output_file, classes):
    with open(output_file, 'w') as out_file:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def populateDir(dirname):
    with open(os.path.join(args.dataset_dir, "ImageSets", "Main", f"{dirname}.txt"),'r') as f:
        for base_name in [x.strip() for x in f.readlines()]:
            annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
            image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
            if not os.path.exists(annotation_path):
                print(f"Could not find {annotation_path}")
                continue
            if not os.path.exists(image_path):
                image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                if not os.path.exists(image_path):
                    print(f"Could not find {image_path}")
                    continue
            shutil.copy(image_path, os.path.join(args.output, dirname))
            convert_annotation(annotation_path, os.path.join(args.output, dirname, base_name + '.txt'), classes)

classes = []
with open(os.path.join(args.dataset_dir, "labels.txt"), 'r') as f:
    classes = [x.strip() for x in f.readlines()]

categories = os.listdir(os.path.join(args.dataset_dir, "ImageSets", "Main"))

if len(categories) > 0:
    if "test.txt" in categories and "train.txt" in categories and "val.txt" in categories:
        print("Found test, train and val")
        populateDir('test')
        populateDir('train')
        populateDir('val')
    elif "test.txt" in categories and "trainval.txt" in categories:
        print("Found test and trainval")
        print("Splitting up trainval in train and val with 90/10")
        populateDir('test')
        with open(os.path.join(args.dataset_dir, "ImageSets", "Main", "trainval.txt"),'r') as f:
            file_list = [x.strip() for x in f.readlines()]
            for i,base_name in enumerate(file_list):
                if i <= 0.9 * len(file_list):
                    annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
                    image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
                    if not os.path.exists(annotation_path):
                        print(f"Could not find {annotation_path}")
                        continue
                    if not os.path.exists(image_path):
                        image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                        if not os.path.exists(image_path):
                            print(f"Could not find {image_path}")
                            continue
                    shutil.copy(image_path, os.path.join(args.output, "train"))
                    convert_annotation(annotation_path, os.path.join(args.output, "train", base_name + '.txt'), classes)
                else:
                    annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
                    image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
                    if not os.path.exists(annotation_path):
                        print(f"Could not find {annotation_path}")
                        continue
                    if not os.path.exists(image_path):
                        image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                        if not os.path.exists(image_path):
                            print(f"Could not find {image_path}")
                            continue
                    shutil.copy(image_path, os.path.join(args.output, "val"))
                    convert_annotation(annotation_path, os.path.join(args.output, "val", base_name + '.txt'), classes)
    else:
        print("Did not find the right image sets, creating own division 75/18/7")
        print(f'Using {os.path.join(args.dataset_dir, "ImageSets", "Main", categories[0])} as image set')
        with open(os.path.join(args.dataset_dir, "ImageSets", "Main", categories[0]),'r') as f:
            file_list = [x.strip() for x in f.readlines()]
            for i,base_name in enumerate(file_list):
                if i <= 0.75 * len(file_list):
                    annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
                    image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
                    if not os.path.exists(annotation_path):
                        print(f"Could not find {annotation_path}")
                        continue
                    if not os.path.exists(image_path):
                        image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                        if not os.path.exists(image_path):
                            print(f"Could not find {image_path}")
                            continue
                    shutil.copy(image_path, os.path.join(args.output, "train"))
                    convert_annotation(annotation_path, os.path.join(args.output, "train", base_name + '.txt'), classes)
                elif i <= 0.93:
                    annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
                    image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
                    if not os.path.exists(annotation_path):
                        print(f"Could not find {annotation_path}")
                        continue
                    if not os.path.exists(image_path):
                        image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                        if not os.path.exists(image_path):
                            print(f"Could not find {image_path}")
                            continue
                    shutil.copy(image_path, os.path.join(args.output, "test"))
                    convert_annotation(annotation_path, os.path.join(args.output, "test", base_name + '.txt'), classes)
                else:
                    annotation_path = os.path.join(args.dataset_dir, "Annotations", base_name + ".xml")
                    image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".jpg")
                    if not os.path.exists(annotation_path):
                        print(f"Could not find {annotation_path}")
                        continue
                    if not os.path.exists(image_path):
                        image_path = os.path.join(args.dataset_dir, "JPEGImages", base_name + ".JPG")
                        if not os.path.exists(image_path):
                            print(f"Could not find {image_path}")
                            continue
                    shutil.copy(image_path, os.path.join(args.output, "val"))
                    convert_annotation(annotation_path, os.path.join(args.output, "val", base_name + '.txt'), classes)
    with open(os.path.join(args.output, os.path.basename(args.output) + ".yaml"), 'w') as f:
        yamlcontent = [
            {'path': args.output},
            {'train': 'train'},
            {'val': 'val'},
            {'test': 'test'},
            {'nc': len(classes)},
            {'names': classes}
        ]
        yaml.safe_dump(yamlcontent, f)

    print(f"Finished converting dataset, make sure to update the {os.path.basename(args.output) + '.yaml'} file if you change the dataset location!")
else:
    print(f"Need at least one file in {os.path.join(args.dataset_dir, 'ImageSets', 'Main')}", file=sys.stderr)
    sys.exit(1)




