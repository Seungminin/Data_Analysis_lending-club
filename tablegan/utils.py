"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
"""
from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from scipy.spatial import cKDTree

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import pickle
import pandas as pd

import gc
#메모리 증가
import sys
#sys.setrecursionlimit(10**6)

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])

DATASETS = ('LACity', 'Health', 'Adult', 'Ticket')


def padding_duplicating(data, row_size):
    arr_data = np.array(data.values.tolist())

    col_num = arr_data.shape[1]

    npad = ((0, 0), (0, row_size - col_num))

    # PAdding with zero
    arr_data = np.pad(arr_data, pad_width=npad, mode='constant', constant_values=0.)

    # Duplicating Values 
    for i in range(1, arr_data.shape[1] // col_num):
        arr_data[:, col_num * i: col_num * (i + 1)] = arr_data[:, 0: col_num]

    return arr_data


def reshape(data, dim):
    data = data.reshape(data.shape[0], dim, -1)

    return data


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)

    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def save_data(data, data_file):
    with open(data_file, 'wb') as handle:
        return pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(data_file):
    with open(data_file + '.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    # import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def histogram(data_r, data_f, xlabel, ylabel, save_dir):
    if not os.path.exists(save_dir + '/histo'):
        os.makedirs(save_dir + '/histo')

    fig = plt.figure()
    plt.hist(data_r, bins='auto', label="Real Data")
    plt.hist(data_f, bins='auto', alpha=0.5, label=" Fake Data")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    plt.savefig(save_dir + "/histo/" + xlabel)

    plt.close(fig)

    plt.close()


def cdf(data_r, data_f, xlabel, ylabel, save_dir):
    if not os.path.exists(save_dir + '/cdf'):
        os.makedirs(save_dir + '/cdf')

    axis_font = {'fontname': 'Arial', 'size': '18'}

    # Cumulative Distribution
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    fig = plt.figure()

    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    plt.grid()
    plt.margins(0.02)

    plt.plot(x1, y, marker='o', linestyle='none', label='Real Data', ms=8)
    plt.plot(x2, y, marker='o', linestyle='none', label='Fake Data', alpha=0.5, ms=5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

    plt.savefig(save_dir + "/cdf/" + xlabel)

    plt.close(fig)

    gc.collect()


def nearest_value(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rounding(fake, real, batch_size=10000):
    """
    범주형 데이터를 Label Encoding 후 복원하고, 연속형 데이터를 가장 가까운 값으로 반올림하는 함수.

    Parameters:
    - fake (numpy.ndarray): 생성된 가짜 데이터
    - real (numpy.ndarray): 원본 데이터
    - batch_size (int): 배치 단위 처리 크기 (기본값: 10,000)

    Returns:
    - fake (numpy.ndarray): 범주형 데이터 복원 및 연속형 데이터 반올림된 가짜 데이터
    """

    # ✅ Categorical 컬럼 찾기 (문자열 또는 카테고리형 데이터)
    categorical_cols = real.select_dtypes(include=['object', 'category']).columns.tolist()

    # ✅ Continuous 컬럼 찾기 (숫자형 데이터)
    continuous_cols = real.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # ✅ Label Encoding 적용 (범주형 데이터 변환)
    encoders = {col: LabelEncoder().fit(real[col]) for col in categorical_cols}

    for col in categorical_cols:
        print(f"🔄 Label Encoding: {col}")
        fake[:, col] = encoders[col].inverse_transform(encoders[col].transform(fake[:, col].astype(str)))

    # ✅ 연속형 데이터에 대해 반올림 적용 (`searchsorted` 사용)
    for i, col in enumerate(continuous_cols):
        print(f"⚡ Fast rounding column: {col}")

        # ✅ 원본 데이터 정렬 (정렬 O(M log M))
        unique_values = np.sort(np.unique(real[col].values))

        # ✅ 이진 탐색을 통한 가장 가까운 값 찾기 (O(N log log M))
        indices = np.searchsorted(unique_values, fake[:, i], side="left")

        # ✅ 경계값 처리 (인덱스 범위 초과 방지)
        indices = np.clip(indices, 0, len(unique_values) - 1)

        # ✅ 가장 가까운 값으로 대체
        fake[:, i] = unique_values[indices]

    return fake

def compare(real, fake, save_dir, col_prefix, CDF=True, Hist=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        # Comparing Based on on mimumum number of columns and rows

    max_col = min(real.shape[1], fake.shape[1])
    max_row = min(fake.shape[0], real.shape[0])

    gap = np.zeros(max_col)

    for i in range(max_col):

        if Hist == True:
            histogram(real[: max_row, i], fake[: max_row, i], xlabel=col_prefix + ' : Column ' + str(i + 1), ylabel='',
                      save_dir=save_dir)

        if CDF == True:
            cdf(real[: max_row, i], fake[: max_row, i], xlabel=col_prefix + ' : Columns ' + str(i + 1),
                ylabel='Percentage', save_dir=save_dir)

        print(col_prefix + " : Cumulative Dist of Col " + str(i + 1))


def generate_data(sess, model, config, option, num_samples=1000000):

    print("🚀 Start Generating Data...")

    if option == 1:
        # ✅ 샘플 수 및 배치 계산
        input_size = num_samples
        dim = config.output_width
        batch_size = config.batch_size

        total_batches = math.ceil(input_size / batch_size)
        actual_samples = total_batches * batch_size

        print(f"🔢 요청 샘플 수: {input_size}, 실제 생성 샘플 수 (배치 맞춤): {actual_samples}")

        # ✅ 결과 배열 초기화
        merged_data = np.zeros((actual_samples, dim, dim), dtype=float)

        save_dir = f'./{config.sample_dir}/{config.dataset}'
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(total_batches):
            print(f"📦 Generating batch {idx + 1}/{total_batches}")

            # ⭐ 마지막 배치 샘플 수 조정
            samples_to_generate = batch_size if idx < total_batches - 1 else input_size - (idx * batch_size)

            z_sample = np.random.uniform(-1, 1, size=(samples_to_generate, model.z_dim))
            zero_labels = model.zero_one_ratio

            y = np.ones((samples_to_generate, 1))
            y[:int(zero_labels * samples_to_generate)] = 0
            np.random.shuffle(y)

            y = y.astype('int16')
            y_one_hot = np.zeros((samples_to_generate, model.y_dim))
            y_one_hot[np.arange(samples_to_generate), y.flatten()] = 1

            samples = sess.run(
                model.sampler,
                feed_dict={model.z: z_sample, model.y: y_one_hot, model.y_normal: y}
            )

            # ✅ 생성된 샘플 병합
            start_idx = idx * batch_size
            end_idx = start_idx + samples_to_generate
            merged_data[start_idx:end_idx] = samples.reshape(samples_to_generate, dim, dim)

        # ✅ 최종 데이터 변환 및 샘플 자르기
        fake_data = merged_data[:input_size].reshape(input_size, dim * dim)
        fake_data = fake_data[:, :model.attrib_num]  # (1000000, 65)
        print(f"✅ Fake Data shape: {fake_data.shape}")

        # ✅ 원본 데이터 로드
        origin_data_path = model.train_data_path
        if os.path.exists(origin_data_path + ".csv"):
            print(f"📥 Loading CSV input file: {origin_data_path}.csv")
            origin_data = pd.read_csv(origin_data_path + ".csv", sep=',')  # ✅ 수정됨
            origin_data = origin_data.apply(pd.to_numeric, errors='coerce').fillna(0)  # 숫자 변환 및 NaN 처리
        elif os.path.exists(origin_data_path + ".pickle"):
            with open(origin_data_path + '.pickle', 'rb') as handle:
                origin_data = pickle.load(handle)
        else:
            print("❌ Error: 원본 데이터 로드 실패")
            exit(1)

        # ✅ 데이터 스케일링
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        min_max_scaler.fit(origin_data.values)
        scaled_fake = min_max_scaler.inverse_transform(fake_data)

        # ✅ 데이터 반올림 및 저장
        round_scaled_fake = rounding(scaled_fake, origin_data.values)
        output_path = f'{save_dir}/{config.dataset}_{config.test_id}_fake.csv'
        print("fake 파일 만들어지는 중")
        pd.DataFrame(round_scaled_fake).to_csv(output_path, index=False, sep=',')

        print(f"✅ Generated Data shape: {round_scaled_fake.shape}")
        print(f"💾 파일 저장 완료: {output_path}")

    elif option == 5:  # Results for ShadowGAN (membership attack)
        save_dir = f'./{config.sample_dir}/{config.dataset}'
        os.makedirs(save_dir, exist_ok=True)

        if config.shgan_input_type == 1:
            with open(f'./samples/{config.dataset}/{config.test_id}/{config.test_id}_scaled_fake_tabular.pickle', 'rb') as handle:
                data_x = pickle.load(handle)
            output_file = f'{save_dir}/{config.dataset}_{config.test_id}_atk_fake_data.csv'
            discriminator_sampling(data_x, [], output_file, 'In', config, model, sess)

        elif config.shgan_input_type == 2:
            with open(f'./data/{config.dataset}/test_{config.dataset}_cleaned.pickle', 'rb') as handle:
                data_x = pickle.load(handle)
            with open(f'./data/{config.dataset}/test_{config.dataset}_labels.pickle', 'rb') as handle:
                data_y = pickle.load(handle)
            output_file = f'{save_dir}/{config.dataset}_{config.test_id}_atk_test_data.csv'
            discriminator_sampling(data_x, data_y.reshape(-1, 1), output_file, 'Out', config, model, sess)

        elif config.shgan_input_type == 3:
            with open(f'./data/{config.dataset}/train_{config.dataset}_cleaned.pickle', 'rb') as handle:
                data_x = pickle.load(handle)
            with open(f'./data/{config.dataset}/train_{config.dataset}_labels.pickle', 'rb') as handle:
                data_y = pickle.load(handle)
            output_file = f'{save_dir}/{config.dataset}_{config.test_id}_atk_train_data.csv'
            discriminator_sampling(data_x, data_y.reshape(-1, 1), output_file, '', config, model, sess)


def discriminator_sampling(input, lables, output_file, title, config, dcgan, sess):
    dim = config.output_width  # 8
    chunk = config.batch_size

    X = pd.DataFrame(input)

    padded_ar = padding_duplicating(X, dim * dim)

    X = reshape(padded_ar, dim)

    print("Final Real Data shape = " + str(input.shape))  # 15000 * 8 * 8

    # we need to generate lables from fake date to feed teh Discriminator Sampler
    input_size = len(input)
    print("input shape = " + str(input.shape))

    merged_data = np.ndarray([chunk * (input_size // chunk), 2], dtype=float)

    print(" Chunk Size = " + str(chunk))

    for idx in xrange(input_size // chunk):

        print(" [*] %d" % idx)
        # z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
        if len(lables) == 0:
            if config.dataset == "LACity":

                CLASSIFY_COL = 8  # ( 'Total Payments' = 8 starting from 0)
                CLASSIFY_VAL = 77636.37


            elif config.dataset == "Health":  # Correct
                CLASSIFY_COL = 31  # 'DIQ010' = 31 ( starting from 0)
                CLASSIFY_VAL = 1.0

            elif config.dataset == "Adult":  # Correct

                CLASSIFY_COL = 12  # 'hours-per-week' = 12 ( starting from 0)
                CLASSIFY_VAL = 40.43  # the mean value

            elif config.dataset == "Ticket":
                # Total =80000
                CLASSIFY_COL = 18  # 'MktFare' = 18( starting from 0)
                CLASSIFY_VAL = 204.49

            # Generating Labels
            y = []

            c = 0
            # print((idx+1) * chunk)
            # print((idx) * chunk)
            # print(len( data_x[idx * chunk : (idx+1) * chunk] ) )
            for rec in input[idx * chunk: (idx + 1) * chunk]:
                # print(c)
                c += 1
                if rec[CLASSIFY_COL] > CLASSIFY_VAL:
                    y.append(1)
                else:
                    y.append(0)

            y = np.asarray(y)

        else:
            y = lables[idx * chunk: (idx + 1) * chunk]

        y = y.reshape(-1, 1)

        y = y.astype('int16')
        y_one_hot = np.zeros((chunk, dcgan.y_dim))

        sample_input = X[idx * chunk: (idx + 1) * chunk]

        sample_input = sample_input.reshape(chunk, dim, dim, 1)

        # y indicates the index of ones in y_one_hot : in this case y_dim =2 so indexe are 0 or 1
        y_one_hot[np.arange(chunk), y] = 1

        samples = sess.run(dcgan.sampler_disc,
                           feed_dict={dcgan.inputs: sample_input, dcgan.y: y_one_hot, dcgan.y_normal: y})
        # Samples are Probability of input data (result of Sigmoid Activation in Discriminator)

        # Merging Data for each batch size
        merged_data[idx * chunk: (idx + 1) * chunk, 0] = samples[:, 0]
        merged_data[idx * chunk: (idx + 1) * chunk, 1] = y[:, 0]

    # End For

    print("hstack output  shape = " + str(merged_data.shape))

    f = open(output_file, "w+")

    f.write("Prob, Label , In/Out \n")

    for rec in merged_data:
        f.write("%.3f, %d, %s \n" % (rec[0], rec[1], title))
