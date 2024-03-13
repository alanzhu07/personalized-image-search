import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import tqdm


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dims, output_dim=1):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        if output_dim is not None:
            layers.append(torch.nn.Linear(input_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class FM(torch.nn.Module):
    def __init__(self, feature_dims, n_factors=128):
        super().__init__()
        # feature_dims = [n_users, n_items, n_keywords, ...]
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        self.n_features = int(sum(feature_dims))
        self.embedding = nn.Embedding(self.n_features, n_factors, sparse=True)
        self.fc = nn.Embedding(self.n_features, 1, sparse=True)
        self.linear_layer = nn.Linear(1, 1, bias=True)
        # self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        embed = self.embedding(x)
        square_of_sum = embed.sum(dim=1).pow(2)
        sum_of_square = (embed.pow(2)).sum(dim=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
            # + self.bias \
        # return torch.sigmoid(x).view(-1)
        return x.view(-1)
    
class MF(torch.nn.Module):
    def __init__(self, feature_dims, n_factors=128):
        super().__init__()
        # feature_dims = [n_users, n_items, n_keywords, ...]
        # self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        # self.n_features = int(sum(feature_dims))
        self.user_embedding = nn.Embedding(feature_dims[0], n_factors, sparse=True)
        self.item_embedding = nn.Embedding(feature_dims[1], n_factors, sparse=True)
        # self.keyword_embedding = nn.Embedding(feature_dims[2], n_factors, sparse=True)

        # self.fc = nn.Embedding(self.n_features, 1, sparse=True)
        # self.linear_layer = nn.Linear(1, 1, bias=True)
        # self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        user, item = x[:, 0], x[:, 1]
        return (self.user_embedding(user) * self.item_embedding(item)).sum(1)

class DeepFM(torch.nn.Module):
    def __init__(self, feature_dims, mlp_dims=[64], n_factors=128):
        super().__init__()
        # feature_dims = [n_users, n_items, n_keywords, ...]
        self.offsets = np.array((0, *np.cumsum(feature_dims)[:-1]))
        self.n_features = int(sum(feature_dims))
        self.embed_output_dim = len(feature_dims) * n_factors
        self.embedding = nn.Embedding(self.n_features, n_factors, sparse=True)
        self.fc = nn.Embedding(self.n_features, 1, sparse=True)
        self.linear_layer = nn.Linear(1, 1, bias=True)
        self.mlp = MLP(self.embed_output_dim, mlp_dims)
        # self.bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        embed = self.embedding(x)
        square_of_sum = embed.sum(dim=1).pow(2)
        sum_of_square = (embed.pow(2)).sum(dim=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(embed.view(-1, self.embed_output_dim))
            # + self.bias \
        return x.view(-1)
    

import random
import pickle
import scipy as sp
import numpy as np
from implicit.nearest_neighbours import bm25_weight
# from implicit.als import AlternatingLeastSquares

class ImageDataset(Dataset):
    def __init__(self, df, user_lookup, image_lookup, keyword_lookup, user_downloads, additional_features=[]):
        super().__init__()
        self.df = df
        # self.user_lookup = user_lookup
        # self.image_lookup = image_lookup
        # self.keyword_lookup = keyword_lookup
        self.user_downloads = user_downloads
        self.additional_features = additional_features

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        user = x['anonymous_user_id']
        image = x['photo_id']
        keyword = x['keyword']
        # user = self.user_lookup[x['anonymous_user_id']]
        # image = self.image_lookup[x['photo_id']]
        # keyword = self.keyword_lookup[x['keyword']]
        y = self.user_downloads[user, image]
        x = [user, image, keyword] + x[self.additional_features].tolist()
        return np.array(x), y
    
class ImageBPRDataset(Dataset):
    def __init__(self, df, user_lookup, image_lookup, keyword_lookup, user_downloads, additional_features=[]):
        super().__init__()
        self.df = df
        self.user_lookup = user_lookup
        self.image_lookup = image_lookup
        self.keyword_lookup = keyword_lookup
        self.user_downloads = user_downloads
        self.additional_features = additional_features

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        user = x['anonymous_user_id']
        image = x['photo_id']
        keyword = x['keyword']
        x_out = np.array([user, image, keyword] + x[self.additional_features].tolist())

        # y = negative sample, only change image
        y = self.df.iloc[np.random.randint(len(self.df))]
        while (y['anonymous_user_id'] == x['anonymous_user_id'] and
               y['keyword'] == x['keyword']):
            y = self.df.iloc[np.random.randint(len(self.df))]
        image = y['photo_id']
        y_out = np.array([user, image, keyword] + y[self.additional_features].tolist())
        
        return x_out, y_out
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, positives, negatives):
        loss = -F.logsigmoid(positives - negatives).sum()
        return loss
    
class FMModel():
    def __init__(self,
                 n_factors = 128,
                 mf = False,
                 deep_layers = [64],
                 keyword_dict_file = 'keywords_dict.pkl',
                 keyword_id_dict_file = 'keywords_id_dict.pkl',
                 user_lookup_file = 'user_lookup.pkl',
                 image_lookup_file = 'image_lookup.pkl',
                 sparse_mat_file = 'sparse_matrix.npz',
                 additional_features = ['conversion_country', 'photographer_username', 'stats_views', 'stats_downloads'],
                 features_sizes = [235, 8411, 12, 12],
                 df_photos = None,
    ):
        
        with open(keyword_dict_file, 'rb') as file:
            self.keyword_dict = pickle.load(file)
            kw_size = len(self.keyword_dict)
            self.keyword_lookup = defaultdict(lambda: kw_size, {keyword: i for i, keyword in enumerate(self.keyword_dict)})
        with open(keyword_id_dict_file, 'rb') as file:
            self.keyword_id_dict = pickle.load(file)
        with open(user_lookup_file, 'rb') as file:
            self.all_users = pickle.load(file)
            self.user_lookup = {user: i for i, user in enumerate(self.all_users)}
        with open(image_lookup_file, 'rb') as file:
            self.all_images = pickle.load(file)
            self.image_lookup = {photo: i for i, photo in enumerate(self.all_images)}
        self.df_photos = df_photos
        
        self.deep = deep_layers
        self.mf = mf
        if self.mf:
            self.model = MF([len(self.all_users), len(self.all_images), kw_size + 1, *features_sizes], n_factors=n_factors).cuda()
        elif not self.deep:
            self.model = FM([len(self.all_users), len(self.all_images), kw_size + 1, *features_sizes], n_factors=n_factors).cuda()
        else:
            self.model = DeepFM([len(self.all_users), len(self.all_images), kw_size + 1, *features_sizes], mlp_dims=deep_layers, n_factors=n_factors).cuda()
        self.additional_features = additional_features        

        sparse_mat = sp.sparse.load_npz(sparse_mat_file)
        # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
        # and to reduce the weight given to popular items
        interaction_mat = bm25_weight(sparse_mat.tocsr(), K1=100, B=0.8)

        # get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
        self.user_downloads = interaction_mat.T.tocsr()

    def train(self, df_train, batch_size=2048, epochs=2, lr=1e-3):
        self.model.train()
        ds_train = ImageDataset(df_train, self.user_lookup, self.image_lookup, self.keyword_lookup, self.user_downloads, self.additional_features)
        dl_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=2)

        # opt_sparse = optim.SparseAdam(self.model.parameters(), lr=1e-3)
        if self.mf:
            opt_sparse = optim.SparseAdam([self.model.user_embedding.weight, self.model.item_embedding.weight], lr=lr)
            optims = [opt_sparse]
        else:
            opt_sparse = optim.SparseAdam([self.model.embedding.weight, self.model.fc.weight], lr=lr)
            if not self.deep:
                opt_dense = optim.Adam([self.model.linear_layer.weight, self.model.linear_layer.bias], lr=lr)
            else:
                opt_dense = optim.Adam([p for p in self.model.mlp.parameters()] +
                                        [self.model.linear_layer.weight,
                                        self.model.linear_layer.bias,], lr=lr)
            optims = [opt_dense, opt_sparse]
        loss_fn = nn.MSELoss()

        # self.als_model = AlternatingLeastSquares(factors=128, regularization=0.05, alpha=2.0)
        # self.als_model.fit(self.user_downloads)
        for i in tqdm.tqdm(range(epochs)):
            losses = []
            for x, y in (pbar := tqdm.tqdm(dl_train)):
                x = x.cuda()
                y = y.float().cuda()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)

                for opt in optims:
                    opt.zero_grad()

                loss.backward()
                
                for opt in optims:
                    opt.step()
                losses.append(loss.item())
                
                pbar.set_postfix_str(f'Loss: {np.mean(losses)}')

            print(f'Epoch {i+1}/{epochs}, Loss: {np.mean(losses)}')
            # self.save(f'fm_model_deep_{i}.pth')

    def bpr_train(self, df_train, batch_size=2048, epochs=2, lr=1e-3):
        self.model.train()
        ds_train = ImageBPRDataset(df_train, self.user_lookup, self.image_lookup, self.keyword_lookup, self.user_downloads, self.additional_features)
        dl_train = DataLoader(ds_train, batch_size, shuffle=True)


        opt_sparse = optim.SparseAdam([self.model.embedding.weight, self.model.fc.weight], lr=lr)
        if not self.deep:
            opt_dense = optim.Adam([self.model.linear_layer.weight, self.model.linear_layer.bias], lr=lr)
        else:
            opt_dense = optim.Adam([p for p in self.model.mlp.parameters()] +
                                    [self.model.linear_layer.weight,
                                     self.model.linear_layer.bias,], lr=lr)
        loss_fn = BPRLoss()

        for i in tqdm.tqdm(range(epochs)):
            losses = []
            for x, y in (pbar := tqdm.tqdm(dl_train)):
                # x = positive, y = negative
                x = x.cuda()
                y = y.cuda()
                x_pred, y_pred = self.model(x), self.model(y)
                loss = loss_fn(x_pred, y_pred)
                opt_sparse.zero_grad()
                opt_dense.zero_grad()
                loss.backward()
                opt_sparse.step()
                opt_dense.step()
                losses.append(loss.item())
                
                pbar.set_postfix_str(f'Loss: {np.mean(losses)}')

            print(f'Epoch {i+1}/{epochs}, Loss: {np.mean(losses)}')
            self.save(f'fm_model_bpr_deep_{i}.pth')

    def load(self, model_file = 'fm_model.pth'):
        self.model.load_state_dict(torch.load(model_file))

    def save(self, model_file = 'fm_model.pth'):
        state_dict = self.model.state_dict()
        # state_dict['bias'] = self.model.bias
        torch.save(state_dict, model_file)

    # def save(self, als_model_file = 'als_model.npz'):
    #     self.als_model.save(als_model_file)

    # def search(self, user, keyword, k=10, retrieve_k=1024):
    #     self.model.eval()
    #     candidates = [self.image_lookup[image] for image, _ in self.keyword_dict[keyword][:retrieve_k]]
    #     if len(candidates) < retrieve_k:
    #         to_sample = np.ma.array(np.arange(len(self.all_images)), mask=False)
    #         to_sample.mask[candidates] = True
    #         candidates += np.random.choice(to_sample.compressed(), retrieve_k-len(candidates), replace=False).tolist()
        
    #     user_id = [self.user_lookup[user]] * len(candidates)
    #     keyword_id = [self.keyword_lookup[keyword]] * len(candidates)
    #     x = torch.tensor([user_id, candidates, keyword_id]).T.cuda()
    #     # breakpoint()
    #     scores = self.model(x).detach().flatten()
    #     _, image_ids = torch.topk(scores, k)
    #     # breakpoint()
    #     return [self.all_images[image_id] for image_id in image_ids.cpu().detach().numpy()]

        # image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k, items=candidates)
        # return [self.all_images[image_id] for image_id in image_ids]
        
    def search(self, user, keyword, k=10, country=None):
        self.model.eval()
        user_id = self.user_lookup[user]
        keyword_id = self.keyword_lookup[keyword]

        batch_size = 2048
        # image_ids = torch.arange(len(self.all_images)).cuda()
        scores = torch.tensor([]).cuda()
        user = torch.tensor([user_id] * batch_size).unsqueeze(-1)
        country = torch.tensor([country] * batch_size).unsqueeze(-1)
        keyword = torch.tensor([keyword_id] * batch_size).unsqueeze(-1)

        for i in range(0, len(self.all_images), batch_size):
            image_id = np.arange(i, min(i+batch_size, len(self.all_images)))
            image_ids = torch.tensor(image_id).unsqueeze(-1)
            add_feats = torch.tensor(self.df_photos.iloc[image_id][self.additional_features[1:]].to_numpy())
            x = torch.cat((user[:len(image_ids)], image_ids, keyword[:len(image_ids)], country[:len(image_ids)], add_feats), dim=1).cuda()
            scores = torch.cat((scores, self.model(x).detach().flatten()))
        # for image in image_ids.split(batch_size):
        #     x = torch.tensor([[user_id] * len(image), image, [keyword_id] * len(image)]).T.cuda()
        #     scores = torch.cat((scores, self.model(x).detach().flatten()))
        scores, image_ids = torch.topk(scores, k)
        return [self.all_images[image_id] for image_id in image_ids.cpu().detach().numpy()]

    # def predict(self, user, keyword, images):
    #     image_ids = [self.image_lookup[image] for image in images]
    #     user_id = self.user_lookup[user]
    #     image_factors = self.als_model.item_factors[image_ids]
    #     user_factors = self.als_model.user_factors[user_id]
    #     scores = np.dot(image_factors, user_factors)
    #     return scores
    
    def predict(self, user, keyword, images, country=None):
        self.model.eval()
        user_id = self.user_lookup[user]
        user = torch.tensor([user_id] * len(images)).unsqueeze(-1)
        image_id = [self.image_lookup[image] for image in images]
        image_ids = torch.tensor(image_id).unsqueeze(-1)
        keyword = torch.tensor([self.keyword_lookup[keyword]] * len(images)).unsqueeze(-1)
        if country is not None:
            country = torch.tensor([country] * len(images)).unsqueeze(-1)
            add_feats = torch.tensor(self.df_photos.iloc[image_id][self.additional_features[1:]].to_numpy())
            x = torch.cat((user, image_ids, keyword, country, add_feats), dim=1).cuda()
        else:
            x = torch.cat((user, image_ids, keyword), dim=1).cuda()
        return self.model(x).cpu().detach().numpy().flatten()