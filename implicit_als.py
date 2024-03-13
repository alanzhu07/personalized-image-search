
import random
import pickle
import scipy as sp
import numpy as np
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares

class ALSModel():
    def __init__(self,
                 keyword_dict_file = 'keywords_dict.pkl',
                 keyword_id_dict_file = 'keywords_id_dict.pkl',
                 user_lookup_file = 'user_lookup.pkl',
                 image_lookup_file = 'image_lookup.pkl',
                 sparse_mat_file = 'sparse_matrix.npz',
                 als_model_file = None):
        
        with open(keyword_dict_file, 'rb') as file:
            self.keyword_dict = pickle.load(file)
        with open(keyword_id_dict_file, 'rb') as file:
            self.keyword_id_dict = pickle.load(file)
        with open(user_lookup_file, 'rb') as file:
            self.all_users = pickle.load(file)
            self.user_lookup = {user: i for i, user in enumerate(self.all_users)}
        with open(image_lookup_file, 'rb') as file:
            self.all_images = pickle.load(file)
            self.image_lookup = {photo: i for i, photo in enumerate(self.all_images)}

        if als_model_file is not None:
            self.als_model = AlternatingLeastSquares(factors=128, regularization=0.05, alpha=2.0)
            self.als_model.load(als_model_file)
        else:
            self.als_model = None

        sparse_mat = sp.sparse.load_npz(sparse_mat_file)
        # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
        # and to reduce the weight given to popular items
        interaction_mat = bm25_weight(sparse_mat.tocsr(), K1=100, B=0.8)

        # get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
        self.user_downloads = interaction_mat.T.tocsr()

    def train(self):
        self.als_model = AlternatingLeastSquares(factors=128, regularization=0.05, alpha=2.0)
        self.als_model.fit(self.user_downloads)

    def save(self, als_model_file = 'als_model.npz'):
        self.als_model.save(als_model_file)

    def search(self, user, keyword, k=10):
        user_id = self.user_lookup[user]
        image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k)
        return [self.all_images[image_id] for image_id in image_ids]
    
    # def search(self, user, keyword, k=10, retrieve_k=1024):
    #     candidates = [self.image_lookup[image] for image, _ in self.keyword_dict[keyword][:retrieve_k]]
    #     if len(candidates) < k:
    #         to_sample = np.ma.array(np.arange(len(self.all_images)), mask=False)
    #         to_sample.mask[candidates] = True
    #         candidates += np.random.choice(to_sample.compressed(), retrieve_k-len(candidates), replace=False).tolist()
    #     user_id = self.user_lookup[user]
    #     image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k, items=candidates)
    #     return [self.all_images[image_id] for image_id in image_ids]

    def predict(self, user, keyword, images):
        image_ids = [self.image_lookup[image] for image in images]
        user_id = self.user_lookup[user]
        image_factors = self.als_model.item_factors[image_ids]
        user_factors = self.als_model.user_factors[user_id]
        scores = np.dot(image_factors, user_factors)
        return scores