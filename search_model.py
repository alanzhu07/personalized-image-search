import random
import pickle

class SearchModel():
    def __init__(self,
                 keyword_dict_file = 'keywords_dict.pkl',
                 keyword_id_dict_file = 'keywords_id_dict.pkl',
                 image_lookup_file = 'image_lookup.pkl'):

        with open(keyword_dict_file, 'rb') as file:
            self.keyword_dict = pickle.load(file)
        with open(keyword_id_dict_file, 'rb') as file:
            self.keyword_id_dict = pickle.load(file)
        with open(image_lookup_file, 'rb') as file:
            self.all_images = pickle.load(file)

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



