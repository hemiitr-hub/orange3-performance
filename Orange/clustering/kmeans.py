import warnings

import numpy as np
import sklearn.cluster
from sklearn.cluster import MiniBatchKMeans

from Orange.clustering.clustering import Clustering, ClusteringModel
from Orange.data import Table


__all__ = ["KMeans"]

# Row threshold above which MiniBatchKMeans is used automatically.
# MiniBatch is O(n) vs O(n·k·iter) for full KMeans, and gives near-identical
# quality for large datasets.
MINIBATCH_THRESHOLD = 10_000


class _KProtoWrapper:
    """Makes a fitted KPrototypes look like a sklearn KMeans for KMeansModel.

    KPrototypes stores mixed-type centroids as a 2-element list:
      [numeric_centroids_array, categorical_centroids_array]
    This wrapper exposes the numeric part as cluster_centers_ so KMeansModel
    stays compatible, and forwards predict() with the stored categorical
    column indices.
    """

    def __init__(self, kproto, categorical_indices):
        self._kproto = kproto
        self._cat_indices = categorical_indices
        self.labels_ = kproto.labels_
        # Expose numeric centroids for KMeansModel.centroids property
        self.cluster_centers_ = kproto.cluster_centroids_[0]

    def predict(self, X):
        # Fill NaN before predict (same logic as fit)
        X = _fill_nans_for_kproto(X, self._cat_indices)
        return self._kproto.predict(X, categorical=self._cat_indices)

    def get_params(self):
        return {"n_clusters": self._kproto.n_clusters}


def _fill_nans_for_kproto(X, cat_indices):
    """Fill NaN values in X in-place for KPrototypes.

    Numeric NaNs → column mean (or 0 if all-NaN).
    Categorical NaNs → column mode (most frequent integer code, or 0).
    Returns a copy so the original is not mutated.
    """
    X = np.array(X, dtype=object)  # keep object dtype for mixed handling
    n_cols = X.shape[1]
    num_indices = [i for i in range(n_cols) if i not in cat_indices]

    for i in num_indices:
        col = X[:, i].astype(float)
        mask = np.isnan(col)
        if np.any(mask):
            fill = float(np.nanmean(col)) if not np.all(mask) else 0.0
            col[mask] = fill
            X[:, i] = col

    for i in cat_indices:
        col = X[:, i]
        # Orange encodes discrete vars as float ints; NaN means missing
        try:
            fcol = col.astype(float)
        except (ValueError, TypeError):
            continue
        mask = np.isnan(fcol)
        if np.any(mask):
            vals = fcol[~mask].astype(int)
            fill = int(np.bincount(vals).argmax()) if len(vals) else 0
            fcol[mask] = fill
            X[:, i] = fcol

    return X


class KMeansModel(ClusteringModel):

    InheritEq = True

    def __init__(self, projector):
        super().__init__(projector)

    @property
    def centroids(self):
        # converted into a property for __eq__ and __hash__ implementation
        return self.projector.cluster_centers_

    @property
    def k(self):
        # converted into a property for __eq__ and __hash__ implementation
        return self.projector.get_params()["n_clusters"]

    def predict(self, X):
        return self.projector.predict(X)


class KMeans(Clustering):

    __wraps__ = sklearn.cluster.KMeans
    __returns__ = KMeansModel

    def __init__(self, n_clusters=8, init='k-means++', n_init='auto', max_iter=300,
                 tol=0.0001, random_state=None, preprocessors=None,
                 compute_silhouette_score=None):
        if compute_silhouette_score is not None:
            warnings.warn(
                "compute_silhouette_score is deprecated. Please use "
                "sklearn.metrics.silhouette_score to compute silhouettes.",
                DeprecationWarning)
        super().__init__(
            preprocessors, {k: v for k, v in vars().items()
                            if k != "compute_silhouette_score"})

    # ── Mixed-type interception (before preprocessing strips domain info) ──

    def get_model(self, data):
        """Override to route mixed-type data to KPrototypes before preprocessing."""
        cat_indices = [i for i, v in enumerate(data.domain.attributes)
                       if v.is_discrete]
        num_indices = [i for i, v in enumerate(data.domain.attributes)
                       if v.is_continuous]

        if cat_indices and num_indices:
            # Mixed data: bypass Continuize/SklImpute, use KPrototypes natively
            return self._fit_kprototypes(data, cat_indices)

        # All-numeric (or all-categorical) → normal preprocessing + KMeans path
        return super().get_model(data)

    def _fit_kprototypes(self, data, cat_indices):
        """Fit KPrototypes on mixed numeric+categorical data without preprocessing.

        KPrototypes runs Euclidean distance on numeric columns and Hamming
        distance on categorical columns simultaneously, so no Continuize or
        SklImpute is needed.  This avoids both the domain-rebuild overhead and
        the information loss from one-hot encoding discrete variables.

        Falls back to the standard KMeans path (with preprocessing) if the
        kmodes package is not installed.
        """
        try:
            from kmodes.kprototypes import KPrototypes as _KProto
        except ImportError:
            warnings.warn(
                "kmodes not installed; falling back to KMeans with Continuize. "
                "Install with: pip install kmodes",
                RuntimeWarning, stacklevel=3)
            return super().get_model(data)

        # Orange stores all attributes (discrete and continuous) as float in X.
        # KPrototypes needs the raw array with categorical column indices.
        X = _fill_nans_for_kproto(data.X.copy(), cat_indices)

        n_init = self.params.get('n_init', 3)
        if n_init == 'auto':
            # KPrototypes doesn't recognise 'auto'; use its conventional default
            n_init = 3

        kp = _KProto(
            n_clusters=self.params.get('n_clusters', 8),
            init='Cao',        # Cao initialiser works well for mixed data
            n_init=n_init,
            max_iter=self.params.get('max_iter', 300),
            random_state=self.params.get('random_state'),
            verbose=0,
        )
        kp.fit(X, categorical=cat_indices)

        wrapper = _KProtoWrapper(kp, cat_indices)
        model = self.__returns__(wrapper)
        # Keep original domain — no preprocessing was applied
        model.domain = data.domain
        model.original_domain = data.domain
        return model

    # ── Large-data numeric path ───────────────────────────────────────────────

    def fit(self, X, y=None):
        """Fit K-Means, automatically using MiniBatchKMeans for large datasets.

        When the number of rows exceeds MINIBATCH_THRESHOLD (10 000), this
        method switches to sklearn's MiniBatchKMeans which is O(n) per
        iteration instead of O(n·k·iter), giving 10–100× speed improvement
        on large datasets with negligible quality loss.
        """
        if X.shape[0] > MINIBATCH_THRESHOLD:
            # MiniBatchKMeans accepts a subset of full KMeans params.
            # 'tol' is not a MiniBatchKMeans param — skip it.
            _mb_param_names = {'n_clusters', 'init', 'n_init', 'max_iter',
                               'random_state'}
            mb_params = {k: v for k, v in self.params.items()
                         if k in _mb_param_names}
            if mb_params.get('n_init') == 'auto':
                mb_params['n_init'] = 3
            fitted = MiniBatchKMeans(**mb_params).fit(X)
            return self.__returns__(fitted)
        return super().fit(X, y)


if __name__ == "__main__":
    d = Table("iris")
    km = KMeans(preprocessors=None, n_clusters=3)
    clusters = km(d)
    model = km.fit_storage(d)
