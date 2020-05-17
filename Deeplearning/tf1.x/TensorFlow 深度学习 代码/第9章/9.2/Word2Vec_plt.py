from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import vocabulary
import Word2Vec_skip

# 初始化TSNE类
# 构造函数原型__init__(self,n_components,perplexity,early_exaggeration,learning_rate,n_iter,
#                      n_iter_without_progress,min_grad_norm,metric,init,verbose,random_state,
#                       method,angle)
tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)
plot_only = 100

# 执行降维操作
# 函数原型TSNE.fit_transform(Self,X,y)
low_dim_embs = tsne.fit_transform(Word2Vec_skip.final_embeddings[:plot_only, :])

labels = list()
for i in range(plot_only):
    labels.append(vocabulary.reverse_dictionary[i])

# pyplot的figure()函数用于定义画布的大小，这里设为20x20，你也可以尝试其他大小
plt.figure(figsize=(20, 20))

for j, label in enumerate(labels):
    x, y = low_dim_embs[j, :]

    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points",
                 ha="right", va="bottom")

# 以png格式保存图片
plt.savefig(filename="after_tsne.png")

