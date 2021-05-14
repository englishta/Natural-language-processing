#%%
# p64
text = "You say goodbye and I say hello."
text = text.replace('.', ' .')
text
# %%
words = text.split(' ')
words
# %%
word_to_id = {}
id_to_word = {}
for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
print(word_to_id)
print(id_to_word)
# %%
id_to_word[1]
word_to_id["hello"]
# %%
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)
corpus
# %%
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words  = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[e] for e in words])

    return corpus, word_to_id, id_to_word

# %%
print(preprocess("yorosiku onegai simasu"))
# %%
text = "You say goodbye and I say hellow."
corpus, word_to_id, id_to_word = preprocess(text)

# %%
# commonディレクトリ下のutil.pyのpreprocess関数を使う
from util import preprocess
text = "You say goodbye and I say hellow."
print(preprocess(text))
# %%
# 共起行列をつくる関数 p72
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx-i
            right_idx = idx+i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id]+=1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id]+=1
    return co_matrix

# %%
import numpy as np
from util import preprocess
text = "You say goodbye and I say hellow."
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
matrix = create_co_matrix(corpus, len(word_to_id), len(corpus))
print(matrix)

# %%
