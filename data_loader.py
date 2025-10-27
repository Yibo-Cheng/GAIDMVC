import scipy.io as scio
import torch
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
import numpy as np
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def apply_normalization(data, normalization):

    if normalization == 'minmax_scale':
        return minmax_scale(data)
    elif normalization == 'maxabs_scale':
        return maxabs_scale(data)
    elif normalization == 'normalize':
        return normalize(data)
    elif normalization == 'robust_scale':
        return robust_scale(data)
    elif normalization == 'scale':
        return scale(data)
    elif normalization == '255':
        return data / 255.0
    elif normalization == '50':
        return data / 50.0
    elif normalization == 'no':
        return data
    else:
        raise ValueError("Invalid normalization type: {}".format(normalization))

def load_multi_view_data(feature_list, normalization='scale'):

    normalized_views = []
    for feature in feature_list:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        normed = apply_normalization(feature, normalization)
        normalized_views.append(torch.tensor(normed, dtype=torch.float32))
    return normalized_views

def load_data(name):

        path = './data/{}.mat'.format(name)
        dims = []
        # Loading .mat files
        data = scio.loadmat(path)

        # Automatically identify all keys starting with 'V' as views
        view_keys = [key for key in data.keys() if key.startswith('V') and key[1:].isdigit()]

        # Sort by view number (e.g., V1, V2, V3...)
        view_keys.sort(key=lambda x: int(x[1:]))

        # Load all view data
        X = []

        for key in view_keys:
            temp=data[key].astype(np.float32)
            X.append(temp)
        X=load_multi_view_data(X, normalization='minmax_scale')
        # Load labels
        labels = data['labels']
        labels = np.reshape(labels, (labels.shape[0],))

        # Calculate the number of views and classes
        view_num = len(X)
        class_num = len(np.unique(labels))

        # get the dimension of each view
        for i in range(view_num):
            dims.append(X[i].shape[1])

        return X, labels, dims, view_num, class_num, X[0].shape[0]
