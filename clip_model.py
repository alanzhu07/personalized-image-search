import random
import pickle
import torch
import numpy as np
import clip

class SearchModel():
    def __init__(self,
                 keyword_dict_file = 'keywords_dict.pkl',
                 keyword_id_dict_file = 'keywords_id_dict.pkl',
                 image_lookup_file = 'image_lookup.pkl',
                 image_features_file = 'data/img_features_CLIP.npz'):

        with open(keyword_dict_file, 'rb') as file:
            self.keyword_dict = pickle.load(file)
        with open(keyword_id_dict_file, 'rb') as file:
            self.keyword_id_dict = pickle.load(file)
        with open(image_lookup_file, 'rb') as file:
            self.all_images = pickle.load(file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_features = torch.tensor(np.load('img_features_CLIP.npz')['arr_0']).to(device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    def search(self, user, keyword, k=10):
        if keyword in self.keyword_dict:
            output = [image for image, _ in self.keyword_dict[keyword][:k]]
            if len(output) < k:
                output += random.sample(self.all_images, k-len(output))
            return output
        else:
            return random.sample(self.all_images, k)

    def predict(self, user, keywords, images):
        if isinstance(keywords, list):
            return [self.keyword_id_dict[keyword][image] for keyword, image in zip(keywords, images)]
        return [self.keyword_id_dict[keywords][image] for image in images]



