import h5py
import json
import logging
import math
import os
import numpy
import re
import sys
from collections import OrderedDict, Counter
from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from gensim.models.word2vec import Word2Vec
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from vgg import VGGClassifier


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    locals().update(json.load(f))

with open('list.txt', 'r') as f:
    files = f.read().splitlines()

## Load data and define vocab ##
logger.info('Reading json and jpeg files...')
movies = []
vocab_counts = []
clsf = VGGClassifier(model_path='vgg16.tar', synset_words='synset_words.txt')
for i, file in enumerate(files):
    with open(file) as f:
        data = json.load(f)
        data['imdb_id'] = file.split('/')[-1].split('.')[0]
        # if 'plot' in data and 'plot outline' in data:
        #    data['plot'].append(data['plot outline'])
        im_file = file.replace('json', 'jpeg')
        if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
            plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
            data['plot'] = normalizeText(data['plot'][plot_id])
            if len(data['plot']) > 0:
                vocab_counts.extend(data['plot'])
                data['cover'] = VGGClassifier.resize_and_crop_image(
                    im_file, img_size)
                data['vgg_features'] = clsf.get_features(im_file)
                movies.append(data)
    logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
        i, len(files), float(i) / len(files) * 100))

logger.info('done reading files.')

vocab_counts = OrderedDict(Counter(vocab_counts).most_common())
vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
googleword2vec = Word2Vec.load_word2vec_format(word2vec_path, binary=True)
ix_to_word = dict(zip(range(len(vocab)), vocab))
word_to_ix = dict(zip(vocab, range(len(vocab))))
lookup = numpy.array([googleword2vec[v] for v in vocab if v in googleword2vec])
numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
                            'word_to_ix': word_to_ix,
                            'vocab_size': len(vocab),
                            'lookup': lookup})


# Define train, dev and test subsets
counts = OrderedDict(
    Counter([g for m in movies for g in m['genres']]).most_common())
target_names = list(counts.keys())[:n_classes]

le = MultiLabelBinarizer()
Y = le.fit_transform([m['genres'] for m in movies])
labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]

B = numpy.copy(Y)
rng = numpy.random.RandomState(rng_seed)
train_idx, dev_idx, test_idx = [], [], []
for l in labels[::-1]:
    t = B[:, l].nonzero()[0]
    t = rng.permutation(t)
    n_test = int(math.ceil(len(t) * test_size))
    n_dev = int(math.ceil(len(t) * dev_size))
    n_train = len(t) - n_test - n_dev
    test_idx.extend(t[:n_test])
    dev_idx.extend(t[n_test:n_test + n_dev])
    train_idx.extend(t[n_test + n_dev:])
    B[t, :] = 0

indices = numpy.concatenate([train_idx, dev_idx, test_idx])
nsamples = len(indices)
nsamples_train, nsamples_dev, nsamples_test = len(
    train_idx), len(dev_idx), len(test_idx)

# Obtain feature vectors and text sequences
sequences = []
X = numpy.zeros((indices.shape[0], textual_dim), dtype='float32')
for i, idx in enumerate(indices):
    words = movies[idx]['plot']
    sequences.append([word_to_ix[w] if w in vocab else unk_idx for w in words])
    X[i] = numpy.array([googleword2vec[w]
                        for w in words if w in googleword2vec]).mean(axis=0)

del googleword2vec

# get n-grams representation
sentences = [' '.join(m['plot']) for m in movies]
ngram_vectorizer = TfidfVectorizer(
    analyzer='char', ngram_range=(3, 3), min_df=2)
ngrams_feats = ngram_vectorizer.fit_transform(sentences).astype('float32')
word_vectorizer = TfidfVectorizer(min_df=10)
wordgrams_feats = word_vectorizer.fit_transform(sentences).astype('float32')


# Store data in the hdf5 file
f = h5py.File('multimodal_imdb.hdf5', mode='w')
dtype = h5py.special_dtype(vlen=numpy.dtype('int32'))
features = f.create_dataset('features', X.shape, dtype='float32')
vgg_features = f.create_dataset(
    'vgg_features', (nsamples, 4096), dtype='float32')
three_grams = f.create_dataset(
    'three_grams', (nsamples, ngrams_feats.shape[1]), dtype='float32')
word_grams = f.create_dataset(
    'word_grams', (nsamples, wordgrams_feats.shape[1]), dtype='float32')
images = f.create_dataset(
    'images', [nsamples, num_channels] + img_size[::-1], dtype='int32')
seqs = f.create_dataset('sequences', (nsamples,), dtype=dtype)
genres = f.create_dataset('genres', (nsamples, n_classes), dtype='int32')
imdb_ids = f.create_dataset('imdb_ids', (nsamples,), dtype="S7")
imdb_ids[...] = numpy.asarray([m['imdb_id']
                               for m in movies], dtype='S7')[indices]
features[...] = X
for i, idx in enumerate(indices):
    images[i] = movies[idx]['cover']
    vgg_features[i] = movies[idx]['vgg_features']
seqs[...] = sequences
genres[...] = Y[indices][:, labels]
three_grams[...] = ngrams_feats[indices].todense()
word_grams[...] = wordgrams_feats[indices].todense()
genres.attrs['target_names'] = json.dumps(target_names)
features.dims[0].label = 'batch'
features.dims[1].label = 'features'
three_grams.dims[0].label = 'batch'
three_grams.dims[1].label = 'features'
word_grams.dims[0].label = 'batch'
word_grams.dims[1].label = 'features'
imdb_ids.dims[0].label = 'batch'
genres.dims[0].label = 'batch'
genres.dims[1].label = 'classes'
vgg_features.dims[0].label = 'batch'
vgg_features.dims[1].label = 'features'
images.dims[0].label = 'batch'
images.dims[1].label = 'channel'
images.dims[2].label = 'height'
images.dims[3].label = 'width'

split_dict = {
    'train': {
        'features': (0, nsamples_train),
        'three_grams': (0, nsamples_train),
        'sequences': (0, nsamples_train),
        'images': (0, nsamples_train),
        'vgg_features': (0, nsamples_train),
        'imdb_ids': (0, nsamples_train),
        'word_grams': (0, nsamples_train),
        'genres': (0, nsamples_train)},
    'dev': {
        'features': (nsamples_train, nsamples_train + nsamples_dev),
        'three_grams': (nsamples_train, nsamples_train + nsamples_dev),
        'sequences': (nsamples_train, nsamples_train + nsamples_dev),
        'images': (nsamples_train, nsamples_train + nsamples_dev),
        'vgg_features': (nsamples_train, nsamples_train + nsamples_dev),
        'imdb_ids': (nsamples_train, nsamples_train + nsamples_dev),
        'word_grams': (nsamples_train, nsamples_train + nsamples_dev),
        'genres': (nsamples_train, nsamples_train + nsamples_dev)},
    'test': {
        'features': (nsamples_train + nsamples_dev, nsamples),
        'three_grams': (nsamples_train + nsamples_dev, nsamples),
        'sequences': (nsamples_train + nsamples_dev, nsamples),
        'images': (nsamples_train + nsamples_dev, nsamples),
        'vgg_features': (nsamples_train + nsamples_dev, nsamples),
        'imdb_ids': (nsamples_train + nsamples_dev, nsamples),
        'word_grams': (nsamples_train + nsamples_dev, nsamples),
        'genres': (nsamples_train + nsamples_dev, nsamples)}
}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()


# Plot distribution
cm = numpy.zeros((n_classes, n_classes), dtype='int')
for i, l in enumerate(labels):
    cm[i] = Y[Y[:, l].nonzero()[0]].sum(axis=0)[labels]

cmap = pyplot.cm.Blues
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
for i in range(len(target_names)):
    cm_normalized[i, i] = 0
pyplot.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto')
for i, cas in enumerate(cm):
    for j, c in enumerate(cas):
        if c > 0:
            pyplot.text(j - .2, i + .2, c, fontsize=4)
pyplot.title('Shared labels', fontsize='smaller')
pyplot.colorbar()
tick_marks = numpy.arange(len(target_names))
pyplot.xticks(tick_marks, target_names, rotation=90)
pyplot.yticks(tick_marks, target_names)
pyplot.tight_layout()
pyplot.savefig('distribution.pdf')
pyplot.close()
