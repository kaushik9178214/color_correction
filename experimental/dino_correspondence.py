import numpy as np

from scipy.spatial.distance import cdist


def dense_match(
        ref_features,
        cand_features):

    similarity = (
        1 -
        cdist(
            ref_features,
            cand_features,
            metric="cosine"
        )
    )

    matches = np.argmax(
        similarity,
        axis=1
    )

    scores = np.max(
        similarity,
        axis=1
    )

    return (
        matches,
        scores
    )


def overall_match_score(
        scores):

    return float(
        np.mean(scores)
    ) * 100