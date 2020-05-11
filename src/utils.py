import os
import random
import shutil
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from skimage import feature, transform
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from itertools import * 
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator,\
        array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical

def get_data(folder):
	"""
	Load the data and labels from the given folder.
	"""
	X = []
	y = []
	for wbc_type in os.listdir(folder):
		if not wbc_type.startswith('.'):
			if wbc_type in ['NEUTROPHIL']:
				label = 0
			elif wbc_type in ['EOSINOPHIL']:
				label = 1
			elif wbc_type in ['MONOCYTE']:
				label = 2
			elif wbc_type in ['LYMPHOCYTE']:
				label = 3
			else:
				label=4
			for image_filename in tqdm(os.listdir(folder + wbc_type)):
				img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
				if img_file is not None:
					img_file = np.array(Image.fromarray(img_file).resize((80, 60)))
					img_arr = np.asarray(img_file)
					X.append(img_arr)
					y.append(label)
	X = np.asarray(X)
	y = np.asarray(y)
	y_Hot = to_categorical(y , num_classes = 5)
	return X,y_Hot

def get_annotation(image, tree):
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "R":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (0, 255, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
                    if name[0] == "W":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (0, 0, 255), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
                    if name[0] == "P":
                        cv2.rectangle(image, (xmin, ymin),
                                (xmax, ymax), (255, 0, 0), 1)
                        cv2.putText(image, name, (xmin + 10, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)
    plt.figure(figsize=(16,16))
    plt.imshow(image)
    plt.show()

def crop_wbc(image, tree):
    for elem in tree.iter():
        if 'object' in elem.tag or 'part' in elem.tag:
            for attr in list(elem):
                if 'name' in attr.tag:
                    name = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                    if name[0] == "W":
                        return xmin, ymin, xmax, ymax

def augment_image(inpath, des, nums):
    data_gen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range=90,
            shear_range=0.1,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest')
    for root, dirs, files in os.walk(inpath):
        classes = dirs
        break

    fold = [36, 91, 150, 15]
    j = 0
    for c in classes:
        times = int(fold[j]*1.2)
        j += 1
        for root, dirs, files in os.walk("".join([inpath, c])):
            for f in files:
                fname = "/".join([root, f])
                print(f"Processing {fname}")
                img = load_img(fname)
                img = img_to_array(img)
                img = img.reshape((1, ) + img.shape)
                i = 0
                for batch in data_gen.flow(x=img, save_to_dir=des, batch_size=16,
                        shuffle=False, save_format="jpeg", save_prefix=c):
                    i += 1
                    if i > times:
                        break

def train_test_split(path, test_ratio=0.2):
    for root, dirs, files in os.walk(path):
        first = dirs
        break

    for c in first:
        #dest = shutil.copytree(path+c, path+"TRAIN/"+c)
        os.makedirs(path+"TEST/"+c)
        for root, dirs, files in os.walk(path+"TRAIN/"+c):
            print(root)
            ntot = len(files)
            test = random.sample(files, k=int(test_ratio*ntot))
            for f in test:
                shutil.move(root+"/"+f, path+"TEST/"+c)

def gen_masked_img():
    fname = "../dataset-master/labels.csv"
    df = pd.read_csv(fname)
    df = df[["Image", "Category"]]
    df.dropna(inplace=True)
    labels = pd.Series(df.Category.values, index=df.Image).to_dict()

    # Note that the function below is adapted from https://github.com/Shenggan/BCCD_Dataset
    for i in range(0, 500):
        image_index = "BloodImage_00{:03d}".format(i)
        try :
            lab = labels[i]
            if "," in lab:
                print(f"Image {i}, multi-labels, skipped.")
                continue
            image = cv2.imread(f"../BCCD_Dataset/BCCD/JPEGImages/{image_index}.jpg")
            tree = ET.parse(f"../BCCD_Dataset/BCCD/Annotations/{image_index}.xml")
            new_image = np.zeros_like(image)
            wbc = crop_wbc(image, tree)
            new_image[wbc[1]:wbc[3], wbc[0]:wbc[2], :] = 1
            new_image = np.multiply(new_image, image)
            cv2.imwrite(f"../dataset-master/masked/{lab}/BloodImage_{i}.jpg", new_image)
        except :
            continue
from skimage import feature, transform
import numpy as np
import matplotlib.pyplot as plt

def plot_DeepExplain(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    inp = "../dataset-master/masked/"
    des = "../dataset-master/augmented/"
    #augment_image(inp, des, 500)
    train_test_split(des)

if __name__ == "__main__":
    main()
