import os

import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Periods of years
PERIODS = [["1992/", "1993/"],
           ["1994/", "1995/"],
           ["1996/", "1997/"],
           ["1998/", "1999/"],
           ["2000/", "2001/"],
           ["2002/", "2003/"]]

# Path to resources
PATH = "resources/abstract/"

# Path to files should be as follows
# ./resources/abstract/1992/9201001.abs as an example


# Given tuple of years from PERIODS and path, generate model
# of word2vec for the period
def get_model(years, path):
    # Tokenizer to tokenize only words
    tokenizer = RegexpTokenizer(r'\w+')
    # Lemmatizer to lemmatize
    wnl = WordNetLemmatizer()

    abstracts = []

    # For each year
    for year in years:
        # For each file in year
        for f in os.listdir(path + year):
            # Get the file and open it for reading
            path_file = os.path.abspath(path + year + f)
            file = open(path_file, 'r')

            lines = file.readlines()

            i = 0
            text = ""
            # Get the abstract part
            for line in lines:

                if line.strip() == "\\\\":
                    if i == 2:
                        break
                    i += 1

                if i == 2:
                    text += line

            # Tokenize and lemmatize text
            sentences = []
            tokens = []
            text_split = text.split('.')

            for k in range(len(text_split)):
                sentence = text_split[k]
                sentence = sentence.strip()
                tokens.extend([wnl.lemmatize(token) for token in tokenizer.tokenize(sentence)])
                sentences.append(tokens)
                tokens = []

            abstracts.extend([x for x in sentences if len(x) > 0])

    # Train and return model
    return Word2Vec(abstracts)


# Get simple list, since model.wv has ndarray type
def get_vectors(model):
    word_vectors = model.wv
    vectors = []

    for word in word_vectors.vocab.items():
        vectors.append(word_vectors[word[0]])

    return vectors


# Choose best guess for clusters
def choose_num_clust(vectors, low, high):
    num_clust = low
    sil_score = -1
    for i in range(low, high):
        kmeans = KMeans(n_clusters=i).fit(vectors)
        ss = silhouette_score(vectors, kmeans.labels_, sample_size=len(vectors))
        if sil_score < ss:
            num_clust = i
            sil_score = ss
    return num_clust


# Plot the cluster
def plot_cls(vectors, num_clust):
    reduced_data = PCA(n_components=2).fit_transform(vectors)
    kmeans = KMeans(n_clusters=num_clust)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the abstract papers dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

model = get_model(PERIODS[0], PATH)
vectors = get_vectors(model)
num_clust = choose_num_clust(vectors, 4, 20)
plot_cls(vectors, num_clust)