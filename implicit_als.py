
import random
import pickle
import torch
import scipy as sp
import numpy as np
import clip
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares

torch.backends.quantized.engine = 'qnnpack'

class ALSModel():
    def __init__(self,
                 keyword_dict_file = 'data/keywords_dict.pkl',
                 keyword_id_dict_file = 'data/keywords_id_dict.pkl',
                 user_lookup_file = 'data/user_lookup.pkl',
                 image_lookup_file = 'data/image_lookup.pkl',
                 sparse_mat_file = 'data/sparse_matrix.npz',
                 image_features_file = 'data/img_features_CLIP.npz',
                 two_step = False,
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
            self.als_model = self.als_model.load(als_model_file)
            print(self.als_model)
        else:
            self.als_model = None

        if image_features_file is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.img_features = torch.tensor(np.load(image_features_file)['arr_0']).to(self.device)
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

            # quantize clip model
            self.clip_model = torch.ao.quantization.quantize_dynamic(
                self.clip_model,
                {torch.nn.Linear},
                dtype=torch.qint8)

        self.two_step = two_step

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

    # def search(self, user, keyword, k=10):
    #     user_id = self.user_lookup[user]
    #     image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k)
    #     return [self.all_images[image_id] for image_id in image_ids]
    
    def cos_sim(self, a, b):
        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        
        return torch.mm(a_norm, b_norm.transpose(0, 1)).reshape(-1)

        # return torch.nn.functional.cosine_similarity(a, b, dim=1)


    
    # def search(self, user, keyword, k=10):
    #     text = clip.tokenize([keyword]).to(self.device)
    #     with torch.no_grad():
    #         text_features = self.clip_model.encode_text(text).float()
    #         similarity = self.cos_sim(self.img_features, text_features)
    #     values, indices = torch.topk(similarity, k)
    #     indices = indices.cpu().numpy()
    #     images = [self.all_images[i] for i in indices]
    #     return images
    
    # def search(self, user, keyword, k=10, retrieve_k=100):
    #     candidates = []
    #     assert hasattr(self, 'clip_model'), 'CLIP model not loaded'

    #     import time
    #     t1 = time.time()
    #     text = clip.tokenize([keyword]).to(self.device)
    #     with torch.no_grad():
    #         text_features = self.clip_model.encode_text(text).float()
    #         t2 = time.time()
    #         similarity = self.cos_sim(self.img_features, text_features)
    #         t3 = time.time()
    #     _, indices = torch.topk(similarity, retrieve_k)
    #     t4 = time.time()
    #     candidates = indices.cpu().numpy().tolist()

    #     user_id = self.user_lookup[user]
    #     image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k, items=candidates)
    #     t5 = time.time()

    #     print(f'text encoding: {t2-t1}, similarity: {t3-t2}, topk: {t4-t3}, recommend: {t5-t4}')
    #     breakpoint()
    #     return [self.all_images[image_id] for image_id in image_ids]

    def search(self, user, keyword, k=10, retrieve_k=100, recommend_k=1000):
        candidates = []
        use_clip = hasattr(self, 'clip_model')

        # step 1: search -> recommend
        if use_clip:
            # assert hasattr(self, 'clip_model'), 'CLIP model not loaded'
            text = clip.tokenize([keyword]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text).float()
                similarity = self.cos_sim(self.img_features, text_features)
            _, indices = torch.topk(similarity, retrieve_k)
            candidates = indices.cpu().numpy().tolist()
        else:
            candidates = [self.image_lookup[image] for image, _ in self.keyword_dict[keyword][:retrieve_k]]
            if len(candidates) < retrieve_k:
                to_sample = np.ma.array(np.arange(len(self.all_images)), mask=False)
                to_sample.mask[candidates] = True
                candidates += np.random.choice(to_sample.compressed(), retrieve_k-len(candidates), replace=False).tolist()
        user_id = self.user_lookup[user]

        if not self.two_step:
            image_ids, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=k, items=candidates)
            return [self.all_images[image_id] for image_id in image_ids]
        
        image_ids_1, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=retrieve_k, items=candidates)

        # step 2: recommend -> search
        image_ids_2, _ = self.als_model.recommend(user_id, self.user_downloads[user_id], N=recommend_k)
        similiarity_2 = self.cos_sim(self.img_features[image_ids_2], text_features)
        _, indices = torch.topk(similiarity_2, retrieve_k)
        image_ids_2 = indices.cpu().numpy().tolist()

        # step 3: ranked voting
        rank_scores = torch.zeros(len(self.all_images), dtype=torch.int8)
        rank_scores[image_ids_1] += torch.arange(retrieve_k, 0, -1)
        rank_scores[image_ids_2] += torch.arange(retrieve_k, 0, -1)

        _, indices = torch.topk(rank_scores, k)
        image_ids = indices.cpu().numpy().tolist()
        return [self.all_images[image_id] for image_id in image_ids]

    def predict(self, user, keyword, images):
        image_ids = [self.image_lookup[image] for image in images]
        user_id = self.user_lookup[user]
        image_factors = self.als_model.item_factors[image_ids]
        user_factors = self.als_model.user_factors[user_id]
        scores = np.dot(image_factors, user_factors)
        return scores