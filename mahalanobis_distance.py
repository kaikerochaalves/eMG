"""
Distance computations (:mod:`scipy.spatial.distance`)
=====================================================

.. sectionauthor:: Damian Eads

Function reference
------------------

Distance matrix computation from a collection of raw observation vectors
stored in a rectangular array.

.. autosummary::
   :toctree: generated/

   pdist   -- pairwise distances between observation vectors.
   cdist   -- distances between two collections of observation vectors
   squareform -- convert distance matrix to a condensed one and vice versa
   directed_hausdorff -- directed Hausdorff distance between arrays

Predicates for checking the validity of distance matrices, both
condensed and redundant. Also contained in this module are functions
for computing the number of observations in a distance matrix.

.. autosummary::
   :toctree: generated/

   is_valid_dm -- checks for a valid distance matrix
   is_valid_y  -- checks for a valid condensed distance matrix
   num_obs_dm  -- # of observations in a distance matrix
   num_obs_y   -- # of observations in a condensed distance matrix

Distance functions between two numeric vectors ``u`` and ``v``. Computing
distances over a large collection of vectors is inefficient for these
functions. Use ``pdist`` for this purpose.

.. autosummary::
   :toctree: generated/

   mahalanobis      -- the Mahalanobis distance.

Distance functions between two boolean vectors (representing sets) ``u`` and
``v``.  As in the case of numerical vectors, ``pdist`` is more efficient for
computing the distances between all pairs.

.. autosummary::
   :toctree: generated/

:func:`hamming` also operates over discrete numerical vectors.
"""

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

__all__ = [
    'mahalanobis'
]


import warnings
import numpy as np



def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u

    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u.squeeze())
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    warnings.warn(
        "scipy.spatial.distance metrics ignoring length-1 dimensions is "
        "deprecated in SciPy 1.7 and will raise an error in SciPy 1.9.",
        DeprecationWarning)
    return u


def mahalanobis(u, v, VI):
    """
    Compute the Mahalanobis distance between two 1-D arrays.

    The Mahalanobis distance between 1-D arrays `u` and `v`, is defined as

    .. math::

       \\sqrt{ (u-v) V^{-1} (u-v)^T }

    where ``V`` is the covariance matrix.  Note that the argument `VI`
    is the inverse of ``V``.

    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    VI : array_like
        The inverse of the covariance matrix.

    Returns
    -------
    mahalanobis : double
        The Mahalanobis distance between vectors `u` and `v`.

    Examples
    --------
    >>> from scipy.spatial import distance
    >>> iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    >>> distance.mahalanobis([1, 0, 0], [0, 1, 0], iv)
    1.0
    >>> distance.mahalanobis([0, 2, 0], [0, 1, 0], iv)
    1.0
    >>> distance.mahalanobis([2, 0, 0], [0, 1, 0], iv)
    1.7320508075688772

    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return m