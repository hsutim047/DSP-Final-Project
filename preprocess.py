import os
import random
import argparse
import scipy.io.wavfile as wavfile
import numpy as np
import python_speech_features as features

PREFIX = '/tmp2/b08201047/data/'
TRAIN_DATA = os.path.join(PREFIX, 'TRAIN')
TEST_DATA = os.path.join(PREFIX, 'TEST')

PHONEMES = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

def wav_duration(filepath):
    with open(filepath, 'r') as f:
        return int(f.readlines()[-1].split()[1])


def index_phoneme(phoneme):
    try:
        return PHONEMES.index(phoneme)
    except:
        return -1


def transform_mfcc(filepath):
    sample_rate, signal = wavfile.read(filepath)

    mfcc = features.mfcc(signal, sample_rate)

    d_mfcc = np.zeros(mfcc.shape)
    for i in range(1, mfcc.shape[0] - 1):
        d_mfcc[i, :] = mfcc[i + 1, :] - mfcc[i - 1, :]
    d_mfcc[[0, -1], :] = mfcc[[0, -1], :]

    dd_mfcc = np.zeros(d_mfcc.shape)
    for i in range(1, d_mfcc.shape[0] - 1):
        dd_mfcc[i, :] = d_mfcc[i + 1, :] - d_mfcc[i - 1, :]
    dd_mfcc[[0, -1], :] = d_mfcc[[0, -1], :]

    feat = np.concatenate([mfcc, d_mfcc, dd_mfcc], axis = 1)

    return feat, int(feat.shape[0])


def get_mu_std(X):

    mean, std = np.zeros(X[0].shape[1]), np.zeros(X[0].shape[1])

    total_duration = 0

    for x in X:
        x_len = x.shape[0]
        mean += np.mean(x, axis = 0) * x_len
        std += np.mean(x, axis = 0) * x_len
        total_duration += x_len

    mean /= total_duration
    std /= total_duration

    return mean, std, total_duration


def normalize(X, mean, std):
    return [(x - mean) / std for x in X]


def astype(X, type):
    return [x.astype(type) for x in X]


def preprocess(root):

    X, Y = [], []
    for dir, _, files in os.walk(root):

        for file in files:
            if not file.endswith('.PHN'):
                continue

            phn_file = os.path.join(dir, file)
            wav_file = os.path.join(dir, file[0: -4] + '.WAV.wav')

            duration = wav_duration(phn_file)

            _X, num_frames = transform_mfcc(wav_file)
            X.append(_X)

            y = np.zeros(num_frames) - 1
            fp = open(phn_file)
            start_index = 0
            for line in fp:
                [start, end, ph] = line.rstrip('\n').split()
                start, end = int(start), int(end)

                ph_index = index_phoneme(ph)
                end_index = int(np.round((end) / duration * num_frames))
                y[start_index : end_index] = ph_index

                start_index = end_index
            fp.close()

            assert(-1 not in y)

            Y.append(y.astype('int32'))

    return X, Y



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', type = str, default = './preprocess_data/')
    return parser


def concatenate(X, ws):
    ret = []
    for i in range(ws):
        x = list(X[i])
        ret.append(np.array(x * (2 * ws + 1)))
    for i in range(ws, X.shape[0] - ws):
        ret.append(np.concatenate(X[i - ws: (i + ws + 1)]))
    for i in range(X.shape[0] - ws, X.shape[0]):
        x = list(X[i])
        ret.append(np.array(x * (2 * ws + 1)))
    ret = np.array(ret)
    return ret


def operate(X, ws = 0):

    return [concatenate(x, ws) for x in X]


def output(X, y, filepath):
    _X = np.concatenate(X)
    _y = np.concatenate(y)

    print(_X.shape, _y.shape)

    np.save(filepath, _X)
    np.save(filepath + '_label', _y)


def main():
    parser = get_args()
    args = parser.parse_args()

    X_train, y_train = preprocess(TRAIN_DATA)
    X_test, y_test = preprocess(TEST_DATA)

    mean, std, _ = get_mu_std(X_train)

    X_train = normalize(X_train, mean, std)
    X_test = normalize(X_test, mean, std)

    X_train = astype(X_train, 'float32')
    X_test 	= astype(X_test, 'float32')

    output(X_train, y_train, args.output_prefix + 'train')
    output(X_test, y_test, args.output_prefix + 'test')


if __name__ == "__main__":
    main()
