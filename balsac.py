from tqdm import tqdm
from scipy.sparse import lil_matrix, save_npz, load_npz
from scipy.io import savemat
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE

def get_data():
    with open('../data/parents.asc', 'r') as infile:
        lines = infile.readlines()[1:]
        data = {}

        for line in lines:
            ind, father, mother, _ = line.strip().split('\t')
            father = father if father != '0' else None
            mother = mother if mother != '0' else None
            data[ind] = (father, mother)
    return data

def get_probands(data):
    probands = set(data.keys()) - {parent for ind in tqdm(data.keys(), desc="Loading the probands")
                                    for parent in data[ind] if parent and parent in data.keys()}
    probands = list(probands)
    probands.sort()
    return probands

def get_all_ancestors(data):
    probands = set(get_probands(data))
    inds = set(data.keys())
    ancestors = inds - probands
    ancestors = list(ancestors)
    ancestors.sort()
    return ancestors

def get_parents(data, ind):
    return data.get(ind, (None, None))

def get_genetic_contributions(data):
    if exists('gc.npz'):
        matrix = load_npz('gc.npz')
        return matrix

    probands = get_probands(data)
    ancestors = get_all_ancestors(data)
    ancestor_idx_map = {ancestor: idx for idx, ancestor in enumerate(ancestors)}

    matrix = lil_matrix((len(probands), len(ancestors)), dtype=np.float32)

    def add_contribution(proband_index, ancestor, depth):
        if ancestor is None:
            return
        
        ancestor_idx = ancestor_idx_map[ancestor]
        matrix[proband_index, ancestor_idx] += 1 / 2 ** depth

        father, mother = get_parents(data, ancestor)
        add_contribution(proband_index, father, depth+1)
        add_contribution(proband_index, mother, depth+1)

    for index, proband in enumerate(tqdm(probands, desc="Computing the genetic contributions")):
        father, mother = get_parents(data, proband)
        add_contribution(index, father, 1)
        add_contribution(index, mother, 1)

    matrix = matrix.tocsr()
    save_npz('gc.npz', matrix)

    return matrix

def get_ancestors_of(data, individual, memo=None):
    if memo is None:
        memo = {}
    if individual in memo:
        return memo[individual]

    ancestors = set()

    def add_ancestor(individual, ancestor, ancestors):
        if ancestor is None:
            return ancestors
        ancestors |= {ancestor}
        father, mother = get_parents(data, ancestor)
        add_ancestor(individual, father, ancestors)
        add_ancestor(individual, mother, ancestors)
        return ancestors

    father, mother = get_parents(data, individual)
    ancestors |= add_ancestor(individual, father, ancestors)
    ancestors |= add_ancestor(individual, mother, ancestors)
    memo[individual] = ancestors
    return ancestors

def get_all_MRCAs(data, probands):
    # Slow and eats a lof of memory
    ancestors_memo = {}
    mrca_memo = {}

    def find_MRCAs(candidate, possible_MRCAs, memo=None):
        if memo is None:
            memo = {}
        if (candidate, frozenset(possible_MRCAs)) in memo:
            return memo[(candidate, frozenset(possible_MRCAs))]
        if not candidate:
            return set()
        if candidate in possible_MRCAs:
            return {candidate}
        father, mother = get_parents(data, candidate)
        found_MRCAs = set()
        if father:
            found_MRCAs |= find_MRCAs(father, possible_MRCAs, memo)
        if mother:
            found_MRCAs |= find_MRCAs(mother, possible_MRCAs, memo)
        memo[(candidate, frozenset(possible_MRCAs))] = found_MRCAs
        return found_MRCAs

    MRCAs = set()
    proband_count = len(probands)
    for i in tqdm(range(proband_count), desc="Finding all the MRCAs"):
        ind1 = probands[i]
        possible_MRCAs = get_ancestors_of(data, ind1, ancestors_memo)
        for j in range(i + 1, proband_count):
            ind2 = probands[j]
            MRCAs |= find_MRCAs(ind2, possible_MRCAs, mrca_memo)
    return MRCAs

def get_distribution_df(values):
    series = pd.Series(values)
    occurrence = series.value_counts()
    df = pd.DataFrame({'gc': occurrence.index, 'occurrence': occurrence.values})
    df.reset_index(drop=True, inplace=True)
    return df

def plot_gc_each_scatter(df):
    plt.scatter(df['gc'], df['occurrence'], alpha=0.5)
    plt.xlabel('Genetic Contribution')
    plt.ylabel('Occurrence')
    plt.title('Genetic Contribution vs Occurrence')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_gc_sums_scatter(df):
    plt.scatter(df['gc'], df['occurrence'], alpha=0.5)
    plt.xlabel('Total Genetic Contribution')
    plt.ylabel('Occurrence')
    plt.title('Total Genetic Contribution vs Occurrence')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_gc_sums_density(df):
    sns.kdeplot(df['gc'], bw_method='scott', label='gc')
    plt.xlabel('Genetic Contribution')
    plt.ylabel('Density')
    plt.title('Density Plot of Genetic Contribution')
    plt.show()

def plot_bar_graph(df):
    plt.bar(df['gc'], df['occurrence'])
    plt.xlabel('City')
    plt.ylabel('Occurrence')
    plt.title('City Occurrence')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.show()

def get_highest_contributors(data, matrix):
    col_sums = matrix.sum(axis=0)
    cols_threshold = col_sums >= 10
    cols_above_threshold = cols_threshold.nonzero()[1]
    new_matrix = matrix[:, cols_above_threshold]
    highest_contributors = np.array(get_all_ancestors(data))[cols_above_threshold]
    return highest_contributors, new_matrix

def get_labels(probands, label):
    output = []
    with open('regions.txt', 'r') as infile:
        labels = {}
        lines = infile.readlines()[1:]
        for line in tqdm(lines, desc="Reading the history file"):
            individual, birth_date, death_date, wed_date, wed_city, wed_region, wed_country, origin = line.strip().split('\t')
            labels[individual] = int(locals()[label])
    for proband in tqdm(probands, desc="Fetching the probands' origins"):
        try:
            output.append(labels[proband])
        except KeyError:
            output.append(0)
    return output

def generate_matlab(matrix, labels=None):
    if labels:
        labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)
    else:
        labels = np.zeros(matrix.shape[0], dtype=np.uint8).reshape(-1, 1)
    data = {'X': matrix, 'Y': labels}
    savemat('balsac.mat', data)

def generate_matlab_complete(matrix, labels):
    labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)
    features = np.ones((1, len(labels)))
    data = {'X': features,'G': matrix, 'labels': labels}
    savemat('balsac.mat', data)

def generate_tf_idf(matrix):
    if exists('tf_idf.npz'):
        matrix = load_npz('tf_idf.npz')
        return matrix
    new_matrix = TfidfTransformer().fit_transform(matrix)
    save_npz('tf_idf.npz', new_matrix)
    return new_matrix

def plot_tf_idf(matrix):
    x = matrix.max(axis=0)
    plt.hist(x.data, bins=25)
    plt.yscale('log')
    plt.ylabel("Occurrence")
    plt.xlabel("Max TF-IDF score for a given ancestor")
    plt.title("Ocurrences of max TF-IDF scores")
    plt.show()

def get_highest_tf_idf(data, matrix):
    ancestors_sum = matrix.sum(axis=0)
    top_ancestors_idx = np.argsort(ancestors_sum).tolist()[0][-10000:]
    top_ancestors = np.array(get_all_ancestors(data))[top_ancestors_idx]
    new_matrix = matrix[:, top_ancestors_idx]
    return top_ancestors, new_matrix

def generate_count_matrix(data):
    if exists('count.npz'):
        matrix = load_npz('count.npz')
        return matrix

    probands = get_probands(data)
    ancestors = get_all_ancestors(data)
    ancestor_idx_map = {ancestor: idx for idx, ancestor in enumerate(ancestors)}

    matrix = lil_matrix((len(probands), len(ancestors)), dtype=np.uint8)

    def add_occurrence(proband_index, ancestor, depth):
        if ancestor is None:
            return
        
        ancestor_idx = ancestor_idx_map[ancestor]
        matrix[proband_index, ancestor_idx] += 1

        father, mother = get_parents(data, ancestor)
        add_occurrence(proband_index, father, depth+1)
        add_occurrence(proband_index, mother, depth+1)

    for index, proband in enumerate(tqdm(probands, desc="Counting the ancestor occurrences")):
        father, mother = get_parents(data, proband)
        add_occurrence(index, father, 1)
        add_occurrence(index, mother, 1)

    matrix = matrix.tocsr()
    save_npz('count.npz', matrix)

    return matrix

def load_embedding():
    matrix = np.load('emb.npy')
    return matrix

def reduce_dimensionality(matrix):
    return TSNE().fit_transform(matrix)

def use_kmeans(embedding, n_clusters):
    clusters = KMeans(n_clusters=n_clusters).fit_predict(embedding)
    return clusters

def use_spectral(embedding, n_clusters):
    clusters = SpectralClustering(n_clusters=n_clusters).fit_predict(embedding)
    return clusters

def plot_clusters(matrix, labels):
    cmap = plt.cm.get_cmap('plasma', len(np.unique(labels)))
    plt.scatter(matrix[:, 0], matrix[:, 1], c=labels, cmap=cmap)
    plt.show()

if __name__ == '__main__':
    data = get_data()
    count_matrix = generate_count_matrix(data)
    tf_idf_matrix = generate_tf_idf(count_matrix)
    labels = get_labels(probands = data.keys(), label = 'wed_city')
    generate_matlab_complete(tf_idf_matrix, labels)
