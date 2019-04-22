import numpy as np
import torch
import torch.utils.data as data
from keras.preprocessing.sequence import pad_sequences


class Dataset(data.Dataset):
    """ Digits dataset."""

    def __init__(self):
        print('Loading data...')
        self.token_ids = np.load('dataset/all_token_ids.npy')
        self.sentence_embeddings = np.concatenate(np.load('pre-trained-embedding/cls_emb.npy'))

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        tokens = self.token_ids[idx]
        sentence_embedding = self.sentence_embeddings[idx]
        return tokens, sentence_embedding


class BatchData:
    def __init__(self, batch):
        batch_data = list(zip(*batch))  # batched tokens, batched sentence_embeddings
        tokens = pad_sequences(batch_data[0], maxlen=39, dtype='int32', padding='post', truncating='post')
        embedings = np.array(batch_data[1])
        self.tokens = torch.IntTensor(tokens)
        self.embedings = torch.FloatTensor(embedings)


def collect_fn(batch):
    return BatchData(batch)


def negative_batch_generator(maxlen, batch_size, neg_size):
    token_ids = np.load('dataset/all_token_ids.npy')
    data = pad_sequences(token_ids, maxlen=maxlen, dtype='int32', padding='post', truncating='post')
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield torch.IntTensor(samples)
