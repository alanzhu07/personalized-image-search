import torch
import numpy as np
import pandas as pd

df_train = pd.read_csv('df_train_cleaned.csv')
df_photos = pd.read_csv('df_photos.csv')
from fm import FMModel
# fm_model = FMModel(deep_layers=[64,16], additional_features=[], features_sizes=[])
fm_model = FMModel(deep_layers=[64,16], n_factors=64, df_photos=df_photos)
print(fm_model.model)

from eval import eval, eval2
df_test = pd.read_csv('df_test.csv')

# fm_model.load('fm_model_deep_1.pth')
# eval(df_test, fm_model)
# for i in range(5):
#     fm_model.train(df_train, batch_size=8192, epochs=3, lr=5e-3)
#     fm_model.save(f'features_deep_{i}.pth')
#     if i == 2:
#         eval2(df_test, fm_model)
#         break
#     else:
#         eval2(df_test[:int(len(df_test)*0.2)], fm_model)

for i in range(5):
    fm_model.bpr_train(df_train, batch_size=8192, epochs=3, lr=5e-3)
    fm_model.save(f'features_bpr_deep_{i}.pth')
    if i == 2:
        eval2(df_test, fm_model)
        break
    else:
        eval2(df_test[:int(len(df_test)*0.2)], fm_model)