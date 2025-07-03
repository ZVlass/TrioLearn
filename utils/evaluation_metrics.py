

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel

def intrinsic_sts_correlation(embeddings1: np.ndarray,
                              embeddings2: np.ndarray,
                              gold_scores: np.ndarray) -> float:
    """
    Compute the Spearman correlation between the cosine similarities of two
    sets of embeddings and gold-standard STS scores.

    :param embeddings1: (N×D) array of sentence embeddings
    :param embeddings2: (N×D) array of paired sentence embeddings
    :param gold_scores: (N,) array of human similarity scores
    :return: Spearman's ρ
    """
    sims = np.array([cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0,0]
                     for a, b in zip(embeddings1, embeddings2)])
    rho, _ = spearmanr(sims, gold_scores)
    return rho


def extrinsic_clustering_score(embeddings: np.ndarray,
                               labels: np.ndarray) -> float:
    """
    Compute silhouette score of your embeddings with respect to ground-truth labels.

    :param embeddings: (M×D) array of document embeddings
    :param labels: (M,) array of integer cluster labels
    :return: silhouette score in [–1,1]
    """
    return silhouette_score(embeddings, labels)


def precision_at_k(recommended: list[list[str]],
                   ground_truth: list[set[str]],
                   k: int = 5) -> float:
    """
    Compute mean Precision@k for keyword recommendations.

    :param recommended: list of length M, each a list of top-k keywords
    :param ground_truth: list of length M, each a set of true keywords
    :param k: cut-off
    :return: average Precision@k
    """
    precisions = []
    for recs, true in zip(recommended, ground_truth):
        rec_k = recs[:k]
        hit_count = sum(1 for kw in rec_k if kw in true)
        precisions.append(hit_count / k)
    return float(np.mean(precisions))


def lda_coherence(lda_model, texts: list[list[str]], dictionary, coherence: str = 'c_v') -> float:
    """
    Compute coherence score of an LDA model.

    :param lda_model: trained gensim.models.LdaModel
    :param texts: tokenized documents (list of list of str)
    :param dictionary: gensim.corpora.Dictionary
    :param coherence: one of ['u_mass','c_v','c_uci','c_npmi']
    :return: coherence score
    """
    cm = CoherenceModel(model=lda_model,
                        texts=texts,
                        dictionary=dictionary,
                        coherence=coherence)
    return cm.get_coherence()


def lda_perplexity(lda_model, corpus) -> float:
    """
    Compute perplexity of an LDA model on a held-out corpus.

    :param lda_model: trained gensim.models.LdaModel
    :param corpus: bag-of-words corpus (list of lists of (int, int) tuples)
    :return: model perplexity (lower is better)
    """
    return lda_model.log_perplexity(corpus)


if __name__ == "__main__":
    # Example usage (you'll replace these stub arrays with your real data):
    import numpy as np
    from gensim.corpora import Dictionary
    from gensim.models.ldamodel import LdaModel

    # 1) STS
    emb1 = np.random.rand(100, 384)
    emb2 = np.random.rand(100, 384)
    gold = np.random.rand(100)
    print("STS Spearman ρ:", intrinsic_sts_correlation(emb1, emb2, gold))

    # 2) Clustering
    labels = np.random.randint(0, 5, size=200)
    emb_docs = np.random.rand(200, 384)
    print("Silhouette:", extrinsic_clustering_score(emb_docs, labels))

    # 3) Keyword Precision@5
    recommended = [["nlp","embedding","lda","topic","model"]]*50
    true_kws = [set(["nlp","text","model"])]*50
    print("Precision@5:", precision_at_k(recommended, true_kws, k=5))

    # 4/5) LDA metrics
    texts = [["sample","document"], ["another","doc"]]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
    print("Coherence (c_v):", lda_coherence(lda, texts, dictionary, 'c_v'))
    print("Perplexity:", lda_perplexity(lda, corpus))
