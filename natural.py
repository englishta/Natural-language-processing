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
from util import most_similar
most_similar('hellow', word_to_id, id_to_word, C)
# %%
