from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import os
from pandas import Series
import re
from sentence_transformers import SentenceTransformer, util


def normalize_text(text: str) -> str:
    """Normaliza o texto de uma matéria aplicando os seguintes passos:
    1- Converte tudo para minúsculo
    2- Remove tags, caracteres especiais e dígios
    3- Aplica as técnicas Stemming e Lemmatisation
    """
    stop_words = stopwords.words('portuguese')
    new_text = text.lower()

    # removendo tags
    new_text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",new_text)

    # removendo caracteres especiais e dígitos
    new_text=re.sub("(\\d|\\W)+"," ",new_text)

    new_text = new_text.split()
    
    # Lemmatisation
    lem = WordNetLemmatizer()
    
    new_text = [lem.lemmatize(word) for word in new_text if not word in stop_words] 
    new_text = " ".join(new_text)

    return new_text

def get_embeddings(sentences: Series, embeddings_name: str) -> np.array:
    filepath = f"../artifacts/{embeddings_name}.npy"
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"

    if os.path.isfile(filepath):
        return np.load(filepath)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    np.save(filepath, embeddings)
    
    return embeddings

def sort_keywords(coo_matrix):
    """Ordena as keywords calculadas em order decrescente"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_top_n(feature_names, tfidf_vector, topn = 5):
    """Captura as top N features e seu score"""
    
    sorted_items = sort_keywords(tfidf_vector.tocoo())[:topn]
    score_vals = []
    feature_vals = []
    
    for i, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[i])
 
    results= {}
    for i in range(len(feature_vals)):
        results[feature_vals[i]]=score_vals[i]
    
    return results

def most_similar(row, embeddings_base, embeddings_to_compare, df_to_compare, n_top=2):
    """Calculating similarity for only comparable items"""
    comp_list = row["comparable"]
    
    sim_dict = {}
    for i in comp_list:
        sim = util.cos_sim(embeddings_base[row.name], embeddings_to_compare[i]).flatten()[0]
        if str(type(sim)) == "<class 'torch.Tensor'>":
            sim = sim.item()
        sim_dict[df_to_compare.iloc[i]['ITE_ITEM_TITLE']] = sim

    # returning n_top most similar items
    items = []
    sim_values = []
    sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_sim:
        items.append(k)
        sim_values.append(round(v, 1))
        if len(items) == n_top:
            break
    
    return (items[:n_top], sim_values[:n_top])
    # return ([k for k, _ in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)][:n_top],
    #         [v for _, v in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)][:n_top])
