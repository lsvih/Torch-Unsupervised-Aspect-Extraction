import joblib
import torch
import torch.nn as nn


class UAEModel(nn.Module):
    def __init__(self, device):
        super(UAEModel, self).__init__()
        self.token_embedding = nn.Embedding.from_pretrained(self.load_token_embeddings(), freeze=True).to(device)
        self.attention = Attention(dim=self.token_embedding.embedding_dim).to(device)
        self.weight_sum = WeightedSum().to(device)

    def forward(self, tokens, sentence_embs, neg_bags):
        token_emb = self.token_embedding(tokens.long())
        neg_embs = self.token_embedding(neg_bags.long())
        neg_emb_mean = torch.mean(neg_embs, dim=1)
        token_emb_mean = torch.mean(token_emb, dim=1)
        att_weights = self.attention(token_emb, token_emb_mean)
        out = self.weight_sum(token_emb, att_weights)
        neg_emb_mean = torch.mean(neg_emb_mean, dim=1)

        return out, neg_emb_mean, sentence_embs

    @staticmethod
    def load_token_embeddings():
        print('loading embeddings...')
        token_embedding = joblib.load('pre-trained-embedding/model.dct')['word_embedding']
        return torch.FloatTensor(token_embedding)


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.att = nn.Linear(dim, dim, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):  # batch * seq_len * emb_dim  , batch * emb_dim  => batch * seq_len
        batch_size, seq_len, emb_dim = x.size()
        y = self.att(y).repeat(1, seq_len).view(batch_size, seq_len, emb_dim)
        eij = torch.sum(x * y, dim=-1)
        att_weights = self.softmax(self.tanh(eij))
        return att_weights


class WeightedSum(nn.Module):
    def forward(self, x, a):
        batch_size, seq_len, emb_dim = x.size()
        a = a.view(batch_size, seq_len, 1).expand(batch_size, seq_len, emb_dim)
        return torch.sum(x * a, dim=1)
