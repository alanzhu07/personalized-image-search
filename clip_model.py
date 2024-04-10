import random
import pickle
import torch
import numpy as np
import clip
from collections import defaultdict
import faiss

torch.backends.quantized.engine = 'qnnpack'

class CLIPModel():
    def __init__(self,
                 keyword_dict_file = 'data/keywords_dict.pkl',
                 keyword_id_dict_file = 'data/keywords_id_dict.pkl',
                 image_lookup_file = 'data/image_lookup.pkl',
                 image_features_file = 'data/img_features_CLIP.npz',
                 fast_search = False
                 ):

        # with open(keyword_dict_file, 'rb') as file:
        #     self.keyword_dict = pickle.load(file)
        # with open(keyword_id_dict_file, 'rb') as file:
        #     self.keyword_id_dict = pickle.load(file)
        # with open(image_lookup_file, 'rb') as file:
        #     self.all_images = pickle.load(file)

        with open(keyword_dict_file, 'rb') as file:
            self.keyword_dict = pickle.load(file)
            kw_size = len(self.keyword_dict)
            self.keyword_lookup = defaultdict(lambda: kw_size, {keyword: i for i, keyword in enumerate(self.keyword_dict)})
        with open(keyword_id_dict_file, 'rb') as file:
            self.keyword_id_dict = pickle.load(file)
        with open(image_lookup_file, 'rb') as file:
            self.all_images = pickle.load(file)
            self.image_lookup = {photo: i for i, photo in enumerate(self.all_images)}
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.img_features = torch.tensor(np.load(image_features_file)['arr_0']).to(self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # quantize clip model
        # self.clip_model = torch.ao.quantization.quantize_dynamic(
        #     self.clip_model,
        #     {torch.nn.Linear},
        #     dtype=torch.qint8)
        
        self.fast_search = fast_search
        if fast_search:
            idx_str = "IVF512,Flat"
            # idx_str = "Flat"
            self.index = faiss.index_factory(512, idx_str, faiss.METRIC_INNER_PRODUCT)
            xb = self.img_features.numpy()
            faiss.normalize_L2(xb)
            if not self.index.is_trained:
                self.index.train(xb)
            self.index.add(xb)
            print('faiss index trained')

        

    def cos_sim(self, a, b):
        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1)).reshape(-1)

    def retrieve(self, a, b, k):
        if not self.fast_search:
            similarity = self.cos_sim(a, b)
            values, indices = torch.topk(similarity, k)
            return indices.cpu().numpy()
        else:
            qr = b.cpu().numpy()
            faiss.normalize_L2(qr)
            values, indices = self.index.search(qr, k)
            return indices[0]

    
    def search(self, user, keyword, k=10):
        text = clip.tokenize([keyword]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text).float()
            indices = self.retrieve(self.img_features, text_features, k)
        #     similarity = self.cos_sim(self.img_features, text_features)
        # values, indices = torch.topk(similarity, k)
        # indices = indices.cpu().numpy()
        return indices
        # images = [self.all_images[i] for i in indices]
        return images

        if keyword in self.keyword_dict:
            output = [image for image, _ in self.keyword_dict[keyword][:k]]
            if len(output) < k:
                output += random.sample(self.all_images, k-len(output))
            return output
        else:
            return random.sample(self.all_images, k)

    def predict(self, user, keywords, images):
        images = [self.image_lookup[image] for image in images]
        img_features = self.img_features[images]
        text = clip.tokenize([keywords]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text).float()
            similarity = self.cos_sim(img_features, text_features)
        similarity = similarity.cpu().numpy().tolist()
        return similarity

        if isinstance(keywords, list):
            return [self.keyword_id_dict[keyword][image] for keyword, image in zip(keywords, images)]
        return [self.keyword_id_dict[keywords][image] for image in images]



