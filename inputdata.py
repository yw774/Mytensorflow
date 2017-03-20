from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from PIL import Image
from scipy import ndimage, misc
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def savepictures(filename, ims):
  for i,im in enumerate(ims):
    new_im = Image.fromarray(im.astype(np.uint8))
    new_im.save(filename+str(i)+".png")


def find_label(folder,x,start):  
  for i in range(start,10201):
    a=folder[i][0:3]
    if a!=x:
      return i

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]
  #print('Found %d folders' %len(data_folders))
  #print(data_folders)
  return data_folders

pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images, image_size):
  """Load the data for a single letter label."""
  image_files = sorted(os.listdir(folder))
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for i,image in enumerate(image_files):
    if i>=min_num_images: continue
    image_file = os.path.join(folder, image)
    try:
      #image_data = (ndimage.imread(image_file).astype(float) - 
      #             pixel_depth / 2) / pixel_depth
      image_data = (ndimage.imread(image_file)).astype(float)     
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  #print('Full dataset tensor:', dataset.shape)
  #print('Mean:', np.mean(dataset))
  #print('Standard deviation:', np.std(dataset))
  return dataset
 
def file_pickle(pickle_files, save, force):
  if force or not os.path.exists(pickle_files):
    try:
  	  with open(pickle_files,'wb') as f:
  	    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_files, ':', e)
    return pickle_files

def maybe_pickle(data_folders, min_num_images_per_class, image_size, force):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      #print('%s already present - Skipping pickling.' % set_filename)
      pass
    else:
      #print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class, image_size)
      file_pickle(set_filename, dataset, force)
  return dataset_names

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, image_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        #print(train_letter.shape)
        #print(train_dataset.shape)
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  #print(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels



