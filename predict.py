import glob
import importlib
import json
import subprocess
import os
from json import JSONEncoder
import shutil
import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import arwjpg
from set_directory import set_directory
import dedupe
from cnn import CNN


# Scoring Helper Functions
def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, target_file):
    with open(target_file, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def random_crop(img, crop_dims):
    h, w = img.shape[0], img.shape[1]
    ch, cw = crop_dims[0], crop_dims[1]
    assert h >= ch, 'image height is less than crop height'
    assert w >= cw, 'image width is less than crop width'
    x = np.random.randint(0, w - cw + 1)
    y = np.random.randint(0, h - ch + 1)
    return img[y:(y + ch), x:(x + cw), :]


def random_horizontal_flip(img):
    assert len(img.shape) == 3, 'input tensor must have 3 dimensions (height, width, channels)'
    assert img.shape[2] == 3, 'image not in channels last format'
    if np.random.random() < 0.5:
        img = img.swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)
    return img


def load_image(img_file, target_size):
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]
    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))
    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})
    return samples


def predict(model, data_generator):
    return model.predict(data_generator, workers=16, use_multiprocessing=True, verbose=1)


def get_predictions(base_model_name, weights_file, img_dir, img_format='jpg'):
    weights_type = 'technical' if 'technical' in weights_file else 'aesthetic'

    # load samples
    if os.path.isfile(img_dir):
        image_dir, samples = image_file_to_json(img_dir)
    else:
        image_dir = img_dir
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(), img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample[f'score_{weights_type}'] = calc_mean_score(predictions[i])

    return samples


# Classes
class Nima:
    def __init__(self, base_model_name, n_classes=10, learning_rate=0.001, dropout_rate=0, loss=earth_movers_distance,
                 decay=0, weights='imagenet'):
        self.base_model = None
        self.nima_model = None
        self.n_classes = n_classes
        self.base_model_name = base_model_name
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.decay = decay
        self.weights = weights
        self._get_base_module()

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model_name == 'InceptionV3':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_v3')
        elif self.base_model_name == 'InceptionResNetV2':
            self.base_module = importlib.import_module('tensorflow.keras.applications.inception_resnet_v2')
        else:
            self.base_module = importlib.import_module('tensorflow.keras.applications.' + self.base_model_name.lower())

    def build(self):
        # get base model class
        BaseCnn = getattr(self.base_module, self.base_model_name)

        # load pre-trained model
        self.base_model = BaseCnn(input_shape=(224, 224, 3), weights=self.weights, include_top=False, pooling='avg')

        # add dropout and dense layer
        x = Dropout(self.dropout_rate)(self.base_model.output)
        x = Dense(units=self.n_classes, activation='softmax')(x)

        self.nima_model = Model(self.base_model.inputs, x)

    def compile(self):
        self.nima_model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.decay), loss=self.loss)

    def preprocessing_function(self):
        return self.base_module.preprocess_input


class TrainDataGenerator(tf.keras.utils.Sequence):
    """inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator"""

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(256, 256), img_crop_dims=(224, 224), shuffle=True):
        self.indexes = None
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_crop_dims = img_crop_dims  # dimensions that images get randomly cropped to
        self.shuffle = shuffle
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_crop_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                img = random_crop(img, self.img_crop_dims)
                img = random_horizontal_flip(img)
                X[i, ] = img

            # normalize labels
            y[i, ] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


class TestDataGenerator(tf.keras.utils.Sequence):
    """inherits from Keras Sequence base object, allows to use multiprocessing in .fit_generator"""

    def __init__(self, samples, img_dir, batch_size, n_classes, basenet_preprocess, img_format,
                 img_load_dims=(224, 224)):
        self.indexes = None
        self.samples = samples
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.basenet_preprocess = basenet_preprocess  # Keras basenet specific preprocessing function
        self.img_load_dims = img_load_dims  # dimensions that images get resized into when loaded
        self.img_format = img_format
        self.on_epoch_end()  # call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))  # number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.samples[i] for i in batch_indexes]  # get batch samples
        X, y = self.__data_generator(batch_samples)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        X = np.empty((len(batch_samples), *self.img_load_dims, 3))
        y = np.empty((len(batch_samples), self.n_classes))

        for i, sample in enumerate(batch_samples):
            # load and randomly augment image
            img_file = os.path.join(self.img_dir, '{}.{}'.format(sample['image_id'], self.img_format))
            img = load_image(img_file, self.img_load_dims)
            if img is not None:
                X[i, ] = img

            # normalize labels
            if sample.get('label') is not None:
                y[i, ] = normalize_labels(sample['label'])

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = self.basenet_preprocess(X)

        return X, y


# Utils
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def demangle_filenames(img_dir, ext):
    with set_directory(img_dir):
        dirty_files = glob.glob(f'*.{ext}')
        clean_files = [re.sub(rf'[.\-\s](?!{ext})', '', f) for f in dirty_files]
        dict_dirty_clean = dict(zip(dirty_files, clean_files))
        [os.rename(dirty, dict_dirty_clean[dirty]) for dirty in dict_dirty_clean]


def get_file_paths(img_dir, ext):
    with set_directory(img_dir):
        return glob.glob(f'*.{ext}')


def delete_jpgs(jpg_dir):
    """
    Deletes jpg files without having to change working directory
    :param jpg_dir: Directory containing jpg files
    :return: None. Empties the specified file directory
    """
    duplicates_path = os.path.join(jpg_dir, 'duplicates')
    for full_file_path in glob.iglob(os.path.join(jpg_dir, '*.jpg')):
        os.remove(full_file_path)
    if os.path.exists(duplicates_path):
        for full_file_path in glob.iglob(os.path.join(duplicates_path, '*.jpg')):
            os.remove(full_file_path)


def rank(filename):
    return int(filename.split('(')[0])


def get_final_winners(dup_dict):
    d = dup_dict

    losers = set()
    winners = set()

    for i, key in enumerate(list(d.keys())):
        group = d[key] + [key]
        group_ranks = [rank(el) for el in group]
        row_dict = dict(zip(group_ranks, group))
        min_rank = [min(row_dict.keys())][0]
        winner = row_dict[min_rank]
        winners.add(winner)
        del row_dict[min_rank]
        rest = list(row_dict.values())
        losers.update(rest)

    final_winners = {el for el in winners if el not in losers}
    return final_winners


def remove_duplicates(img_dir):
    with set_directory(img_dir):
        filenames = glob.glob('*')
        if not os.path.exists('duplicates'):
            os.mkdir('duplicates')
        [shutil.move(f, "duplicates/") for f in filenames if '.jpg' not in f]
        filenames = glob.glob('*.jpg')
        for f in filenames:
            if f[0] not in "0123456789":
                shutil.move(f, "duplicates/")
    cnn = CNN()
    encodings = cnn.encode_images(image_dir=img_dir)
    duplicates_cnn = cnn.find_duplicates(encoding_map=encodings)
    winners = get_final_winners(duplicates_cnn)

    with set_directory(img_dir):
        files_to_move = [f for f in filenames if f not in winners]
        if not os.path.exists('duplicates'):
            os.mkdir('duplicates')
        [shutil.move(img, "duplicates/") for img in files_to_move]



# Powershell
def powershell_undo_string(full_file_paths):
    filenames = [os.path.basename(path) for path in full_file_paths]
    filesnames_unranked = [f.split('-')[-1] for f in filenames]
    # Combine all file names into a single list
    combo_list_files = [el for element in zip(filenames, filesnames_unranked) for el in element]
    combo_list_full_paths = [os.path.join(f) for f in combo_list_files]
    # Cast to string for Powershell injection
    combo_list_paths_as_string = " ".join(f"'{p}'," for p in combo_list_full_paths)[:-1]
    return combo_list_paths_as_string


def powershell_rename_list(file_dir, combo_list):
    cmd = f"""
        Function ConvertTo-Hashtable($list) {{
            $h = @{{}}
            while($list) {{
                $head, $next, $list = $list
                $h.$head = $next
            }}
            $h
        }}
        $hash = ConvertTo-Hashtable {combo_list}
        Set-Location -Path "{file_dir}";
        $hash.Keys | % {{ Rename-Item $_ $hash.Item($_) }}
        """
    subprocess.run(["powershell", "-Command", cmd])


# Rename
def undo_rename_img(img_dir, ext):
    with set_directory(img_dir):
        [os.rename(str(el), str(el).split('-')[-1]) for el in glob.glob(f'*.{ext}')]


def rename_img(img_dir, dict_id_score, ext):
    with set_directory(img_dir):
        for i, id in enumerate(dict_id_score):
            os.rename(f'{id}.{ext}', f'{i + 1}({round(float(dict_id_score[id]), 3)})-{id}.{ext}')


# def rename_arw(arw_dir, jpg_dir):
#     with set_directory(jpg_dir):
#         jpg_files = glob.glob('*.jpg')


# with set_directory(jpg_dir):
#     # Derive what the filenames would have been if they were ARW Files
#     jpg_files = glob.glob('*.jpg')
#     list_a = [fn.replace('.jpg', '.ARW') for fn in jpg_files]
#     list_b = [fn.split('-')[-1].replace('.jpg', '.ARW') for fn in jpg_files]
#     assert len(list_a) == len(list_b)
#     # Combine all file names into a single list
#     list_c = [x for y in zip(list_b, list_a) for x in y]
#     assert len(list_c) == len(list_a) + len(list_b)
#     # Cast to string for Powershell injection
#     c = " ".join(f"'{el}'," for el in list_c)[:-1]
# os.chdir(arw_dir)
# # Rename the ARW files according to the combined list
# powershell_rename_list(arw_dir, c)


def rename_arw_undo(arw_dir):
    start = os.getcwd()
    os.chdir(arw_dir)
    # Derive what the filenames would have been if they were ARW Files
    arw_files = glob.glob('*.ARW')
    print(f'arw_files = {arw_files}')
    list_a = [f for f in arw_files]
    print(f'list_a = {list_a}')
    list_b = [f.split('-')[-1] for f in arw_files]
    print(f'list_b = {list_b}')
    assert len(list_a) == len(list_b)
    print('Lists are equal length')

    # Combine all file names into a single list
    list_c = [x for y in zip(list_a, list_b) for x in y]
    print(f'list_c = {list_c}')
    assert len(list_c) == len(list_a) + len(list_b)

    # Cast to string for Powershell injection
    c = " ".join(f"'{el}'," for el in list_c)[:-1]
    print(f'c = {c}')

    os.chdir(arw_dir)

    # Rename the ARW files according to the combined list
    powershell_rename_list(arw_dir, c)

    os.chdir(start)


# Analysis
def jpg_combined(base_model_name, weights_aesthetic, weights_technical, jpg_dir, ext='jpg'):
    # Create a df of the average of technical and aesthetic scores
    technical_scores = get_predictions(base_model_name, weights_technical, jpg_dir, ext)
    aesthetic_scores = get_predictions(base_model_name, weights_aesthetic, jpg_dir, ext)
    df_t = pd.read_json(json.dumps(technical_scores))
    df_a = pd.read_json(json.dumps(aesthetic_scores))
    df = pd.merge(df_t, df_a, on='image_id')
    df['average_score'] = df[['score_technical', 'score_aesthetic']].mean(axis=1)
    df = df.sort_values('average_score', ascending=False)
    ids = df['image_id'].tolist()
    scores = df['average_score'].tolist()
    dict_id_score = dict(zip(ids, scores))
    rename_img(jpg_dir, dict_id_score, jpg)
    return json.loads(df.to_json(orient='records'))


def jpg_aesthetic(base_model_name, weights_aesthetic, jpg_dir, ext='jpg'):
    # Create df with aesthetic scores
    aesthetic_scores = get_predictions(base_model_name, weights_aesthetic, jpg_dir, ext)
    df = pd.read_json(json.dumps(aesthetic_scores))
    df = df.sort_values('score_aesthetic', ascending=False)
    keys = df['image_id'].tolist()
    values = df['score_aesthetic'].tolist()
    d = dict(zip(keys, values))
    rename_img(jpg_dir, d, 'jpg')
    remove_duplicates(jpg_dir)
    return json.loads(df.to_json(orient='records'))


def get_aesthetic_score(base_model_name, weights_aesthetic, jpg_dir, ext='jpg'):
    aesthetic_scores = get_predictions(base_model_name, weights_aesthetic, jpg_dir, ext)
    df = pd.read_json(json.dumps(aesthetic_scores))
    df = df.sort_values('score_aesthetic', ascending=False)
    keys = df['image_id'].tolist()
    values = df['score_aesthetic'].tolist()
    dict_id_score = dict(zip(keys, values))
    return dict_id_score, df


def rank_img_aesthetic(base_model_name, weights_aesthetic, img_dir, ext):
    # Get directory of jpg files
    jpg_dir = get_jpg_directory(img_dir, ext)
    # Create a df of the average of technical and aesthetic scores
    dict_id_score, df_id_score = get_aesthetic_score(base_model_name, weights_aesthetic, jpg_dir, "jpg")
    if ext == 'ARW':
        rename_img(jpg_dir, dict_id_score, 'jpg')
        rename_img(img_dir, dict_id_score, 'ARW')
        remove_duplicates(jpg_dir)
        with set_directory(os.path.join(jpg_dir, 'duplicates')):
            duplicate_files = glob.glob('*.jpg')
        with set_directory(img_dir):
            arw_files = glob.glob("*.ARW")
            arw_duplicates = [f.replace('.jpg', '.ARW') for f in duplicate_files]
            if not os.path.exists('duplicates'):
                os.mkdir('duplicates')
            [shutil.move(f, "duplicates/") for f in arw_files if f in arw_duplicates]
        delete_jpgs(jpg_dir)
        try:
            os.removedirs(os.path.join(jpg_dir, 'duplicates'))
        except:
            time.sleep(10)
            os.removedirs(os.path.join(jpg_dir, 'duplicates'))
    elif ext == 'jpg':
        remove_duplicates(img_dir)
        rename_img(jpg_dir, dict_id_score, 'jpg')
    else:
        raise ValueError('File extension not supported')

    return json.loads(df_id_score.to_json(orient='records'))


def rank_img_combo(base_model_name, weights_aesthetic, weights_technical, img_dir, ext):
    # Get directory of jpg files
    jpg_dir = get_jpg_directory(img_dir, ext)
    # Create a df of the average of technical and aesthetic scores
    dict_id_score, df_id_score = get_average_scores(base_model_name, jpg_dir, weights_aesthetic, weights_technical)
    rename_img(jpg_dir, dict_id_score, 'jpg')
    if ext == 'ARW':
        rename_img(img_dir, dict_id_score, 'ARW')
        delete_jpgs(jpg_dir)
    return json.loads(df_id_score.to_json(orient='records'))


def get_jpg_directory(img_dir, ext):
    if ext == 'arw' or ext == 'ARW':
        arwjpg.main(img_dir)  # converts arw to jpg in new jpg_dir
        jpg_dir = os.path.join(img_dir, 'jpg')
    else:
        jpg_dir = img_dir
    return jpg_dir


def get_average_scores(base_model_name, jpg_dir, weights_aesthetic, weights_technical):
    technical_scores = get_predictions(base_model_name, weights_technical, jpg_dir, 'jpg')
    aesthetic_scores = get_predictions(base_model_name, weights_aesthetic, jpg_dir, 'jpg')
    df_t = pd.read_json(json.dumps(technical_scores))
    df_a = pd.read_json(json.dumps(aesthetic_scores))
    df = pd.merge(df_t, df_a, on='image_id')
    df['average_score'] = df[['score_technical', 'score_aesthetic']].mean(axis=1)
    df = df.sort_values('average_score', ascending=False)
    keys = df['image_id'].tolist()
    values = df['average_score'].tolist()
    dict_id_score = dict(zip(keys, values))
    return dict_id_score, df


def get_arw_aesthetic(base_model_name, weights_aesthetic, arw_dir, ext='jpg'):
    # Save start path for later
    start_dir = os.getcwd()
    # Create temporary jpg directory for ranking
    arwjpg.main(arw_dir)
    # Get the jpg dir
    jpg_dir = os.path.join(arw_dir, ext)
    # Create a df of combined technical and aesthetic scores
    aesthetic_scores = get_predictions(base_model_name, weights_aesthetic, jpg_dir, ext)
    df = pd.read_json(json.dumps(aesthetic_scores))
    # Take the average of the aesthetic and technical scores
    df = df.sort_values('score_aesthetic', ascending=False)
    print('Renaming Started...\n')
    arr = df['image_id'].tolist()
    # rename the jpg files
    rename_img(jpg_dir, arr)
    # rename the arw files
    rename_arw(arw_dir, jpg_dir)
    # clean up jpgs directory
    delete_jpgs(jpg_dir)
    # restore working directory
    os.chdir(start_dir)
    return json.loads(df.to_json(orient='records'))


if __name__ == '__main__':
    jpg_aesthetic('MobileNet',
                  'weights_aesthetic.hdf5',
                  r'C:\src\Ranked',
                  'jpg')
