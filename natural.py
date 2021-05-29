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
# 共起行列をppmi行列に変換する
import numpy as np
from util import create_co_matrix, preprocess, cos_similarity, ppmi
text = "You say goodbye and I say hellow."
corpus, word_to_id, id_to_word = preprocess(text)
W = ppmi(C, True)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)
# %%
# SVDによる次元削減p84
import numpy as np
from util import create_co_matrix, preprocess, cos_similarity, ppmi
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
W = ppmi(C, True)
U, S, V = np.linalg.svd(W)
#%%
print(C[0])
print(W[0])
print(U[0])
print(U[0, :2])
# %%
# p85
import matplotlib.pyplot as plt
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()

# %%
