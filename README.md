# Personalized Image Search 

This is a course project for [Machine Learning in Practice](https://www.cs.cmu.edu/~smithv/10718/) at CMU. The models are trained on the [Unsplash Lite Dataset v1.2.2](https://github.com/unsplash/datasets/tree/1.2.2).

The personalized image search engine has two stages, retrieval and personalization.
- Retrieval: retrieve a set of candidate images from the image database that are relevant to the search query
- Personalization (ranking): from the candidate images, rank and recommend the top images, using query and user information

## Retrieval

Two methods are implemented for retrieval.
- Keyword matching: a non-ML approach to match query against existing keywords in the database. Each image in the Unsplash dataset are associated with several keywords with confidence scores.
- CLIP retrieval: pre-compute the [CLIP](https://github.com/openai/CLIP) embeddings of all images in the database. During inference time, calculate the CLIP embedding of query, and return images with highest corresponding cosine similarity.

## Personalization

Several options are implemented for personalization.
- No personalization
- Implicit ALS: Alternating Least Squares with implicit feedback trained using [implicit](https://github.com/benfred/implicit). Takes in only user and image ids.
- Factorization Machine: PyTorch implementation of Factorization Machines (FM), trained using Bayesian Personalized Ranking (BPR) loss. Also includes the Deep Factorization Machines variant. Takes in user and image ids, query, and optionally other features such as user country, photo downloads, photo views.

## Inference Latency Optimization

To improve inference latency, the following implmentations are considered.
- Model Quantization: quantize the linear layers only for inference-time CLIP model and deep personalization models from FP32 to INT8.
- Approximate Nearest Neighbor: in CLIP retrieval, instead of calculating the cosine similarity of the embedding tensors, use approximate nearest neighbor using [FAISS](https://github.com/facebookresearch/faiss) with cosine similarity metric.

## Performance

The metrics used to measure the performance are NDCG and Precision @ K.

to be updated

## Usage

to be updated