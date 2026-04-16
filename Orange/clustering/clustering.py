import numpy as np
import scipy.sparse

from Orange.data import Table, Instance
from Orange.data.table import DomainTransformationError
from Orange.misc.wrapper_meta import WrapperMeta
from Orange.preprocess import Continuize, SklImpute


class ClusteringModel:

    def __init__(self, projector):
        self.projector = projector
        self.domain = None
        self.original_domain = None

    @property
    def labels(self):
        # converted into a property for __eq__ and __hash__ implementation
        return self.projector.labels_

    def __call__(self, data):
        def fix_dim(x):
            return x[0] if one_d else x

        one_d = False
        if isinstance(data, np.ndarray):
            one_d = data.ndim == 1
            prediction = self.predict(np.atleast_2d(data))
        elif isinstance(data, scipy.sparse.csr_matrix) or \
                isinstance(data, scipy.sparse.csc_matrix):
            prediction = self.predict(data)
        elif isinstance(data, (Table, Instance)):
            if isinstance(data, Instance):
                data = Table.from_list(data.domain, [data])
                one_d = True
            if data.domain != self.domain:
                if self.original_domain.attributes != data.domain.attributes \
                        and data.X.size \
                        and not np.isnan(data.X).all():
                    data = data.transform(self.original_domain)
                    if np.isnan(data.X).all():
                        raise DomainTransformationError(
                            "domain transformation produced no defined values")
                data = data.transform(self.domain)
            prediction = self.predict(data.X)
        elif isinstance(data, (list, tuple)):
            if not isinstance(data[0], (list, tuple)):
                data = [data]
                one_d = True
            data = Table.from_list(self.original_domain, data)
            data = data.transform(self.domain)
            prediction = self.predict(data.X)
        else:
            raise TypeError("Unrecognized argument (instance of '{}')"
                            .format(type(data).__name__))

        return fix_dim(prediction)

    def predict(self, X):
        raise NotImplementedError(
            "This clustering algorithm does not support predicting.")

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) is type(other) \
            and self.projector == other.projector \
            and self.domain == other.domain \
            and self.original_domain == other.original_domain

    def __hash__(self):
        return hash((type(self), self.projector, self.domain, self.original_domain))


class Clustering(metaclass=WrapperMeta):
    """
    ${skldoc}
    Additional Orange parameters

    preprocessors : list, optional (default = [Continuize(), SklImpute()])
        An ordered list of preprocessors applied to data before
        training or testing.
    """
    __wraps__ = None
    __returns__ = ClusteringModel
    preprocessors = [Continuize(), SklImpute()]

    def __init__(self, preprocessors, parameters):
        self.preprocessors = preprocessors if preprocessors is not None else self.preprocessors
        self.params = {k: v for k, v in parameters.items()
                       if k not in ["self", "preprocessors", "__class__"]}

    def __call__(self, data):
        return self.get_model(data).labels

    def get_model(self, data):
        orig_domain = data.domain
        data = self.preprocess(data)
        model = self.fit_storage(data)
        model.domain = data.domain
        model.original_domain = orig_domain
        return model

    def fit_storage(self, data):
        # only data Table
        return self.fit(data.X)

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self.__returns__(self.__wraps__(**self.params).fit(X))

    def preprocess(self, data):
        """Apply preprocessors, skipping steps that are already satisfied.

        Optimisations applied:
        - If all domain attributes are already ContinuousVariable, skip
          Continuize (no-op domain transform avoided).
        - If the data matrix contains no NaN values, skip SklImpute (saves a
          full column-by-column pass through the data).
        Both checks are O(1) or O(cols) and cheap compared with the transform
        they avoid.
        """
        # Evaluate lazily — only compute when the first relevant preprocessor
        # is encountered so that pure-integer or mixed domains don't pay twice.
        _all_continuous = None
        _no_nans = None

        def all_continuous():
            nonlocal _all_continuous
            if _all_continuous is None:
                _all_continuous = all(
                    v.is_continuous for v in data.domain.attributes)
            return _all_continuous

        def no_nans():
            nonlocal _no_nans
            if _no_nans is None:
                try:
                    _no_nans = (
                        all_continuous()
                        and data.X is not None
                        and not np.any(np.isnan(data.X))
                    )
                except Exception:
                    _no_nans = False
            return _no_nans

        for pp in self.preprocessors:
            if isinstance(pp, Continuize) and all_continuous():
                # All features are already numeric — Continuize would be a
                # no-op domain rebuild; skip it entirely.
                continue
            if isinstance(pp, SklImpute) and no_nans():
                # No missing values present — SklImpute would be a no-op;
                # skip it entirely.
                continue
            data = pp(data)
        return data
