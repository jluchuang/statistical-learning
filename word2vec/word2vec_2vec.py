import word2vec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# word2vec training 
# word2vec.word2vec('corpus_seg_done.txt', 'corpus_word2vec.bin', size = 300, verbose = True)

# load model
model = word2vec.load('corpus_word2vec.bin')
# print(model.vectors)

# for i in range(995, 1000): 
#     print(model.vocab[i])

index_array = []
metrics_array = []

test_word_array = (u'毕业', u'宝宝', u'打车', u'杨幂', u'腾讯')
test_word_color = ('C3', 'C1', 'C7', 'C0', 'C4')

for word in test_word_array:
    # 在高维空间中(k=300)中找出所选字典距离最近的前10名
    index,metrics = model.cosine(word)
    
    # 所使用的test词汇
    word_index = np.where(model.vocab == word)
    index = np.append(index, word_index)

    index_array.append(index)
    metrics_array.append(metrics)

    print(index)
    print(metrics)


# 引入文章断刺后转为300维向量的资料
raw_word_vec = model.vectors

# 将原本300维的向量空间降为2维
x_reduce = PCA(n_components = 3).fit_transform(raw_word_vec)
print(x_reduce)

zhfont = matplotlib.font_manager.FontProperties(fname='/Users/chuang/Downloads/wqy-MicroHei.ttf')
fig = plt.figure()
ax = Axes3D(fig)

color_index = 0
for index in index_array:
    for i in index:
        ax.text(x_reduce[i][0], x_reduce[i][1], x_reduce[i][2], model.vocab[i], fontproperties = zhfont, color = test_word_color[color_index])
    color_index = color_index + 1

ax.set_xlim(0, 0.5)
ax.set_ylim(-0.2, 0.6)
ax.set_zlim(-0.4, 0.4)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
