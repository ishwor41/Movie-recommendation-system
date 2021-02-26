#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
title_name1= input("The movie you want recommendation with similar format in datasets: ")
datas= pd.read_csv("movies.csv")
datas['genres'] = datas['genres'].fillna('')
tdidf = TfidfVectorizer(stop_words = 'english')
tdidf_matrix = tdidf.fit_transform(datas['genres'])
tdidf_matrix.shape
cos_sim = linear_kernel(tdidf_matrix,tdidf_matrix)
indexes = pd.Series(datas.index, index=datas['title']).drop_duplicates()
def ask_recommendations(title_name, cos_sim=cos_sim):
    ind= indexes[title_name]
    similarity_points = list(enumerate(cos_sim[ind]))
    similarity_points = sorted(similarity_points, key=lambda x: x[1], reverse=True)
    similarity_points = similarity_points[1:20]
    movie_indexes = [i[0] for i in similarity_points]
    return datas['title'].iloc[movie_indexes]
ask_recommendations(title_name1)


# In[ ]:




