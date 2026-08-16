import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MyCSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns for binary EEG classification.

    Learns spatial filters that maximise the variance ratio between two
    classes, then extracts log band-power features from the projected
    signals. Expects band-pass filtered, approximately zero-mean epochs.

    Parameters
    ----------
    n_components : int, default=4
        Number of filters to keep. Must be even: ``n_components // 2`` are
        taken from each end of the eigenvalue spectrum.
    log : bool, default=True
        Apply ``log`` to the band-power features. Recommended for linear
        classifiers.

    Attributes
    ----------
    evals_ : ndarray, shape (n_channels,)
        Generalised eigenvalues from ``fit``, descending.
    filters_ : ndarray, shape (n_channels, n_channels)
        Spatial filters as rows, ordered to match ``evals_``.
    classes_ : ndarray, shape (2,)
        The two class labels found in ``y``, sorted ascending. Leading
        filters maximise variance for ``classes_[0]``.
    """

    def __init__(self, n_components=4, log=True):
        self.n_components = n_components  # number of components to keep
        self.log = log                    # log-variance features

    def _covariance(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Pooled spatial covariance across all epochs in X.

        Concatenates epochs along time and computes the (n_channels, n_channels)
        second-moment matrix. Assumes zero-mean (band-passed) signals; no
        centering is performed.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Epochs from a single class.

        Returns
        -------
        cov : ndarray, shape (n_channels, n_channels)
            Covariance matrix.
        """
        n_channels = X.shape[1]
        Z = X.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = np.matmul(Z, Z.T) / (Z.shape[1] - 1)
        return cov

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> "MyCSP":
        """
        Fit CSP spatial filters by solving a generalised eigenproblem.

        Computes the pooled covariance of each class and solves
        ``S_0 w = lambda (S_0 + S_1) w``, where ``S_i`` is the covariance
        of ``classes_[i]``, so eigenvalues lie in [0, 1] and give the
        fraction of total variance a filter captures for ``classes_[0]``.
        Filters are stored in descending eigenvalue order: leading rows
        maximise variance for ``classes_[0]``, trailing rows for
        ``classes_[1]``.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Epochs from both classes.
        y : ndarray, shape (n_epochs,)
            Labels. Exactly two distinct values required; any labels are
            accepted, sorted ascending to determine filter ordering.

        Returns
        -------
        self : MyCSP
            Fitted estimator.
        """
        if self.n_components % 2 != 0:
            raise ValueError(f"n_components must be even, got {self.n_components}")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"MyCSP requires exactly 2 classes, got {len(classes)}")

        self.classes_ = classes
        S_0 = self._covariance(X[y == classes[0]])
        S_1 = self._covariance(X[y == classes[1]])

        evals, evecs = eigh(S_0, S_0 + S_1)

        order = np.argsort(evals)[::-1]
        self.evals_ = evals[order]
        self.filters_ = evecs[:, order].T
        return self

    def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Project epochs onto the selected CSP filters and extract band-power features.

        Takes the ``m`` filters most discriminative for each class (the extremes
        of the eigenvalue spectrum), projects each epoch, and reduces each
        projected component to its mean square over time — its power, since the
        signals are assumed zero-mean. Optionally log-transformed to make the
        distribution closer to Gaussian for downstream linear classifiers.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Epochs to project. Labels are not required.

        Returns
        -------
        features : ndarray, shape (n_epochs, n_components)
            Band power per selected component; log-transformed if ``self.log``.
        """
        check_is_fitted(self)

        m = self.n_components // 2
        W_selected = np.concatenate(
            [self.filters_[:m, :], self.filters_[-m:, :]], axis=0
        )
        X_prj = np.matmul(W_selected, X)
        features = (X_prj ** 2).mean(axis=2)
        if self.log:
            features = np.log(features)
        return features
