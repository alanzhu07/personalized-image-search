# import faulthandler; faulthandler.enable()

import pandas as pd
import numpy as np
import tqdm
import time
# import faiss

from sklearn.metrics import ndcg_score

def precision_at_k(predicted, ground_truth):
    relevant = list(filter(lambda id: id in ground_truth, predicted))
    return len(relevant) / len(predicted)

# num_keywords = 50
# num_img_per_key = 50

# top_keys = df_conv['keyword'].value_counts().head(num_keywords).index.tolist()
# ndcg = []
# p_at_k = []
# for key in tqdm.tqdm(top_keys):
#     df_conv_key = df_conv[df_conv['keyword'] == key]
#     top_imgs = df_conv_key['image'].value_counts().head(num_img_per_key).index.tolist()
#     scores = search_model.predict(key, top_imgs)
#     search_results = search_model.search(key, num_img_per_key)

#     true_relevance = np.asarray(range(num_img_per_key, 0, -1))
#     ndcg.append(ndcg_score([true_relevance], [scores]))
#     p_at_k.append(precision_at_k(search_results, top_imgs))
    
# print('NDCG:', np.mean(ndcg))
# print('Precision at k:', np.mean(p_at_k))

def eval(df_test, model):
    df_test['user_keyword'] = df_test[['anonymous_user_id', 'keyword']].agg(tuple, axis=1)
    usercounts = df_test.user_keyword.value_counts()

    thresh = 10
    user_keywords = usercounts[usercounts >= thresh].index.tolist()

    ndcg = []
    p_at_k = []
    inf_latencies = []
    for user, kw in (pbar:= tqdm.tqdm(user_keywords)):
        # print(user, kw)
        user_kw_photo = df_test[(df_test['anonymous_user_id'] == user) & (df_test['keyword'] == kw)].photo_id.value_counts()
        top_photos, true_relevance = user_kw_photo.index.tolist(), user_kw_photo.values
        if len(top_photos) < 3:
            continue

        
        scores = model.predict(user, kw, top_photos)

        start = time.time()
        search_results = model.search(user, kw, k=10)
        end = time.time()
        inf_latencies.append(end-start)
        # print(kw, scores[:3])

        ndcg.append(ndcg_score([true_relevance], [scores]))
        p_at_k.append(precision_at_k(search_results, top_photos))
        
        curr_ndcg, curr_p_at_k, curr_lat = np.mean(ndcg), np.mean(p_at_k), np.mean(inf_latencies)
        pbar.set_postfix_str(f'ndcg: {curr_ndcg}, p@k: {curr_p_at_k}, lat: {curr_lat}')

    print('NDCG:', np.mean(ndcg))
    print('Precision at k:', np.mean(p_at_k))
    print('Inference latency:', np.mean(inf_latencies))
    return np.mean(ndcg), np.mean(p_at_k), np.mean(inf_latencies)

def eval2(df_test, model):
    df_test['user_keyword'] = df_test[['anonymous_user_id', 'keyword']].agg(tuple, axis=1)
    usercounts = df_test.user_keyword.value_counts()

    thresh = 10
    user_keywords = usercounts[usercounts >= thresh].index.tolist()

    ndcg = []
    p_at_k = []
    for user, kw in (pbar:= tqdm.tqdm(user_keywords)):
        # print(user, kw)
        user_df = df_test[(df_test['anonymous_user_id'] == user) & (df_test['keyword'] == kw)]
        user_kw_photo = user_df.photo_id.value_counts()
        country = int(user_df.conversion_country.values[0])
        
        top_photos, true_relevance = user_kw_photo.index.tolist(), user_kw_photo.values
        if len(top_photos) < 3:
            continue
        
        
        scores = model.predict(user, kw, top_photos, country=country)
        search_results = model.search(user, kw, k=10, country=country)

        ndcg.append(ndcg_score([true_relevance], [scores]))
        p_at_k.append(precision_at_k(search_results, top_photos))
        
        curr_ndcg, curr_p_at_k = np.mean(ndcg), np.mean(p_at_k)
        pbar.set_postfix_str(f'ndcg: {curr_ndcg}, p@k: {curr_p_at_k}')

    print('NDCG:', np.mean(ndcg))
    print('Precision at k:', np.mean(p_at_k))
    return np.mean(ndcg), np.mean(p_at_k)

if __name__ == "__main__":
    # import glob
    # path = './unsplash/'
    # doc = 'conversions'

    # files = glob.glob(path + doc + ".tsv*")
    # subsets = []
    # for filename in files:
    #     df = pd.read_csv(filename, sep='\t', header=0)
    #     subsets.append(df)
    # df_conv = pd.concat(subsets, axis=0, ignore_index=True).dropna(subset=['anonymous_user_id', 'keyword', 'photo_id'])

    # df_train, df_test = df_conv[:int(len(df_conv)*0.8)], df_conv[int(len(df_conv)*0.8):]
    # print('train size:', len(df_train))
    # df_photos = pd.read_csv('data/df_photos.csv')
    # df_train = pd.read_csv('data/df_train_formatted.csv')
    df_test = pd.read_csv('data/df_test.csv')

    # additional_features = ['conversion_country',  'photographer_username', 'stats_views', 'stats_downloads']
    # features_sizes = df_conv[additional_features].max().to_numpy()+1
    # features_sizes

    # testing search model
    # from search_model import SearchModel
    # search_model = SearchModel()
    # print('non-ML search model')
    # eval(df_test, search_model)

    from clip_model import CLIPModel
    clip_model = CLIPModel(fast_search=False)
    print('CLIP model')
    eval(df_test, clip_model)

    # testing implicit_als model
    # from implicit_als import ALSModel
    # als_model = ALSModel(image_features_file=None)
    # als_model.train()
    # als_model.save('data/als_model.npz')
    # print('implicit_als model')
    # eval(df_test, als_model)

    # del als_model
    # from implicit_als import ALSModel
    # als_model = ALSModel(als_model_file='data/als_model.npz')
    # print('implicit_als model + CLIP')
    # eval(df_test, als_model)
    # breakpoint()

    # from fm import FMModel
    # import faiss
    # fm_model = FMModel(additional_features=[], features_sizes=[], deep_layers=[], two_step=True)
    # fm_model = FMModel(deep_layers=[])
    # breakpoint()
    # fm_model = FMModel(additional_features=[], features_sizes=[], deep_layers=[], image_features_file=None)
    # fm_model.load('data/fm_bprm_6.pth')
    # eval(df_test, fm_model)
    # fm_model.train(df_train, batch_size=4096, epochs=2)
    # # fm_model.save('fm_model__.pth')
    # eval(df_test, fm_model)

    # fm_model.load('fm_model_1.pth')
    # # print('non-ML search model')
    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # (deep models)
    # print('training fm step 1')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=2)
    # fm_model.save('data/fm_features_bpr_2.pth')

    # # fm_model.load('data/fm_model_1.pth')
    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # print('training fm step 2')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=2)
    # fm_model.save('data/fm_features_bpr_4.pth')

    # # # fm_model.load('data/fm_model_2.pth')
    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # print('training fm step 3')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=1)
    # fm_model.save('data/fm_bprm_3.pth')

    # fm_model.load('data/fm_bprm_3.pth')
    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # print('training fm step 4')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=1)
    # fm_model.save('data/fm_bprm_4.pth')

    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # print('training fm step 5')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=1)
    # fm_model.save('data/fm_bprm_5.pth')

    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # print('training fm step 6')
    # fm_model.bpr_train(df_train, batch_size=4096, epochs=1)
    # fm_model.save('data/fm_bprm_6.pth')

    # fm_model.model.eval()
    # eval(df_test, fm_model)

    # except Exception as e:
    #     print(e)
    #     breakpoint()

    # # from fm import FMModel
    # # fm_model = FMModel(df_photos=df_photos)
    # # try:
    # #     fm_model.train(df_train)
    # #     fm_model.save('fm_model_features.pth')
    # #     # fm_model.load()
    # #     print('FM model')
    # #     # eval(df_test, fm_model)
    # # except Exception as e:
    # #     print(e)
    # #     breakpoint()

    # breakpoint()