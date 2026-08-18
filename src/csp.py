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
        Number of filters to keep. Must be even — half are taken from each
        end of the eigenvalue spectrum.
    log : bool, default=True
        Log-transform the band-power features. Recommended for linear
        classifiers.
    reg : float or None, default=None
        Shrinkage coefficient in [0, 1] applied to each class covariance:
        ``(1 - reg) * cov + reg * mean(diag(cov)) * I``. Use a small value
        (e.g. 0.01) when the data is rank-deficient.

    Attributes
    ----------
    evals_ : ndarray, shape (n_channels,)
        Generalised eigenvalues from ``fit``, descending.
    filters_ : ndarray, shape (n_channels, n_channels)
        Spatial filters as rows, ordered to match ``evals_``.
    classes_ : ndarray, shape (2,)
        The two class labels, sorted ascending. Leading filters maximise
        variance for ``classes_[0]``.
    patterns_ : ndarray, shape (n_channels, n_channels)
        Spatial patterns as rows. Plot these, not the filters, as topomaps.
    """

    def __init__(self,
                 n_components=4,
                 log=True,
                 reg=None):
        self.n_components = n_components
        self.log = log
        self.reg = reg

    def _covariance(self,
                    X: npt.NDArray[np.float64]
                    ) -> npt.NDArray[np.float64]:
        """
        Pooled spatial covariance across all epochs in X.

        Concatenates epochs along time. Assumes zero-mean signals; no
        centering is performed.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            Epochs from a single class.

        Returns
        -------
        cov : ndarray, shape (n_channels, n_channels)
            Covariance matrix, shrunk towards the identity if ``reg`` is set.
        """
        n_channels = X.shape[1]
        Z = X.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = np.matmul(Z, Z.T) / (Z.shape[1] - 1)
        if self.reg is not None:
            cov = (1 - self.reg) * cov + self.reg * np.trace(cov) / n_channels * np.eye(n_channels)
        return cov

    def fit(self,
            X: npt.NDArray[np.float64],
            y: npt.NDArray[np.int_]
            ) -> "MyCSP":
        """
        Fit spatial filters by solving a generalised eigenproblem.

        Solves ``S_0 w = lambda (S_0 + S_1) w``, where ``S_i`` is the pooled
        covariance of ``classes_[i]``, so eigenvalues lie in [0, 1]. Filters
        are stored in descending eigenvalue order: leading rows maximise
        variance for ``classes_[0]``, trailing rows for ``classes_[1]``.

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

        Raises
        ------
        ValueError
            If ``n_components`` is odd, or ``y`` does not hold exactly two
            distinct labels.
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
        self.patterns_ = np.linalg.pinv(evecs[:, order])
        return self

    def transform(self,
                  X: npt.NDArray[np.float64]
                  ) -> npt.NDArray[np.float64]:
        """
        Project epochs onto the selected filters and extract band power.

        Uses ``n_components // 2`` filters from each end of the eigenvalue
        spectrum and reduces each projected component to its mean square over
        time — its power, since the signals are assumed zero-mean.

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
