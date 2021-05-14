# %%
import numpy as np
from util import create_co_matrix, preprocess, cos_similarity

text = "You say goodbye and I say hellow."
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(word_to_id))
c0 = C[word_to_id['you']]
c2 = C[word_to_id['i']]
print(cos_similarity(c0, c2))

# %%
def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5):
    #クエリを出す
    if query not in word_to_id:
        print("%s is not found" % query)
        return
    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の算出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # コサイン類似度の結果から、その値を高い順に出力
    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return

# %%
most_similar('hellow', word_to_id, id_to_word, C)
# %%
