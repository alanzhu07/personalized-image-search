# Personalized Image Search 

This is a course project for [Machine Learning in Practice](https://www.cs.cmu.edu/~smithv/10718/) at CMU. The models are trained on the [Unsplash Lite Dataset](https://github.com/unsplash/datasets).

The personalized image search engine has two stages, retrieval and personalization.
- Retrieval: retrieve a set of candidate images from the image database that are relevant to the search query
- Personalization (ranking): from the candidate images, rank and recommend the top images, using query and additional user information

## Retrieval

Currently implemented two retrieval methods.
- Keyword matching: match query against existing keywords in the database. 
- CLIP retrieval:

## Personalization

- No personalization
- Implicit ALS
- Factorization Machine

## Inference Latency Optimization

To improve inference latency, the following implmentations are considered.
- Model Quantization: quantize the linear layers only for inference-time CLIP model and deep personalization models from FP32 to INT8.
- Approximate Nearest Neighbor: in CLIP retrieval, instead of calculating the cosine similarity of the embedding tensors, use approximate nearest neighbor using [FAISS](https://github.com/facebookresearch/faiss).
