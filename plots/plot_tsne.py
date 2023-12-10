from matplotlib import colors
import numpy as np
import ipdb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_tsne(x_array, label_array, **kwargs):
    
    if x_array.shape[-1] >10:
        pca = PCA(n_components=4)
        trans_x = pca.fit_transform(x_array)
        scale = 100/(trans_x.max()+0.000000000000001)
        x_array = trans_x*scale

    fig = plt.figure(dpi=150)
    fig.clf()

    X_embedded = TSNE(n_components=2,random_state=21).fit_transform(x_array[:,:])

    fig.clf()

    cmap = kwargs.get('cmap') or 'viridis'
    sizes = kwargs.get('node_size') or 4
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=label_array,  s=sizes, cmap=cmap) #scatter graph

    return fig

def plot_pca(x_array, label_array, **kwargs):

    if True:
        pca = PCA(n_components=2)
        trans_all = pca.fit_transform(x_array)
        scale = 100/(trans_all.max()+0.0000001)
        trans_all = trans_all*scale

    fig = plt.figure(dpi=150)
    fig.clf()
    all_emb = trans_all
    cmap = kwargs.get('cmap') or 'viridis'
    sizes = kwargs.get('node_size') or 4

    plt.scatter(all_emb[:,0], all_emb[:,1], c=label_array, s=sizes, cmap=cmap)

    return fig

def plot_tsne_anchored(x_array, label_array, anchor_array,anchor_label=None, **kwargs):
    anchor_num = anchor_array.shape[0]

    if x_array.shape[-1] > 10:
        pca = PCA(n_components=4)
        trans_target = pca.fit_transform(x_array)
        trans_anchor = pca.transform(anchor_array)

    #
    scaler = StandardScaler().fit(trans_target)
    trans_target = scaler.transform(trans_target)
    trans_anchor = scaler.transform(trans_anchor)

    fig = plt.figure(dpi=150)
    #fig = plt.figure(figsize=(36,48))

    fig.clf()
    all_emb = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(
        np.concatenate((trans_target,trans_anchor),axis=0))
    target_emb = all_emb[:-anchor_num,:]
    anchor_emb = all_emb[-anchor_num:,:]

    #cmap = kwargs.get('cmap') or 'viridis'
    cmap = plt.cm.rainbow
    class_num = len(set(label_array))
    norm = colors.BoundaryNorm(np.arange(-0.5,class_num,1),cmap.N)
    ticks = np.linspace(0,class_num-1,class_num)

    sizes = kwargs.get('node_size') or 4

    plt.scatter(target_emb[:,0], target_emb[:,1], c=label_array, s=sizes, cmap=cmap, norm=norm)
    plt.colorbar(ticks=ticks)

    if anchor_label is None:
        plt.scatter(anchor_emb[:,0], anchor_emb[:,1],  color='black', s=sizes*6, marker='*')
    else:
        plt.scatter(anchor_emb[:,0], anchor_emb[:,1],  c=anchor_label, s=sizes*6, marker='*')

    return fig

def plot_pca_anchored(x_array, label_array, anchor_array, anchor_label=None, **kwargs):
    anchor_num = anchor_array.shape[0]

    if True:
        pca = PCA(n_components=2)
        trans_target = pca.fit_transform(x_array)
        target_emb = trans_target
        trans_anchor = pca.transform(anchor_array)
        anchor_emb = trans_anchor

    fig = plt.figure(dpi=150)
    fig.clf()

    #cmap = kwargs.get('cmap') or 'viridis'
    cmap = plt.cm.rainbow

    class_num = len(set(label_array))
    norm = colors.BoundaryNorm(np.arange(-0.5,class_num,1),cmap.N)
    ticks = np.linspace(0,class_num-1,class_num)

    sizes = kwargs.get('node_size') or 4

    plt.scatter(target_emb[:,0], target_emb[:,1], c=label_array, s=sizes, cmap=cmap, norm=norm)
    plt.colorbar(ticks=ticks)

    if anchor_label is None:
        plt.scatter(anchor_emb[:,0], anchor_emb[:,1],  color='black', s=sizes*6, marker='*')
    else:
        plt.scatter(anchor_emb[:,0], anchor_emb[:,1],  c=anchor_label, s=sizes*6, marker='*')

    return fig