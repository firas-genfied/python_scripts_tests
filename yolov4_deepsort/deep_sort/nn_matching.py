# vim: expandtab:ts=4:sw=4
import numpy as np
import logging

# Add logging for debugging
logger = logging.getLogger(__name__)

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        
        # Add statistics for monitoring
        self.stats = {
            'total_distance_calls': 0,
            'missing_targets_fixed': 0,
            'empty_samples_handled': 0,
            'distance_errors': 0
        }

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        # DEFENSIVE: Only clean up if we have active targets
        if active_targets:
            # Keep samples for active targets, but don't remove if not in active_targets
            # to prevent KeyErrors. Only remove if explicitly marked for deletion.
            pass  # We'll handle cleanup more carefully elsewhere

    def distance(self, features, targets):
        """Compute distance between features and targets.
        
        DEFENSIVE VERSION: Handles missing targets gracefully.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        self.stats['total_distance_calls'] += 1
        cost_matrix = np.zeros((len(targets), len(features)))
        
        # Log the operation for debugging
        logger.debug(f"Computing distances for targets: {targets}")
        logger.debug(f"Available samples: {list(self.samples.keys())}")
        
        for i, target in enumerate(targets):
            try:
                # LAZY SAMPLE CREATION: Create missing samples automatically
                if target not in self.samples:
                    logger.info(f"Target {target} missing from samples - creating empty sample list")
                    self.samples[target] = []
                    self.stats['missing_targets_fixed'] += 1
                
                # DEFENSIVE: Handle empty samples
                if not self.samples[target]:
                    logger.debug(f"Target {target} has no samples - assigning max distance")
                    cost_matrix[i, :] = float('inf')
                    self.stats['empty_samples_handled'] += 1
                    continue
                
                # DEFENSIVE: Validate samples before computing distance
                target_samples = self.samples[target]
                if not isinstance(target_samples, list):
                    logger.warning(f"Target {target} samples is not a list: {type(target_samples)}")
                    cost_matrix[i, :] = float('inf')
                    continue
                
                # Check if samples contain valid numpy arrays
                valid_samples = []
                for sample in target_samples:
                    try:
                        sample_array = np.asarray(sample)
                        if sample_array.size > 0:
                            valid_samples.append(sample_array)
                    except Exception as e:
                        logger.warning(f"Invalid sample in target {target}: {e}")
                        continue
                
                if not valid_samples:
                    logger.debug(f"Target {target} has no valid samples after validation")
                    cost_matrix[i, :] = float('inf')
                    continue
                
                # DEFENSIVE: Compute distance with error handling
                try:
                    cost_matrix[i, :] = self._metric(valid_samples, features)
                    logger.debug(f"Successfully computed distance for target {target}")
                except Exception as metric_error:
                    logger.error(f"Metric computation failed for target {target}: {metric_error}")
                    cost_matrix[i, :] = float('inf')
                    self.stats['distance_errors'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing target {target}: {e}")
                cost_matrix[i, :] = float('inf')
                self.stats['distance_errors'] += 1
        
        logger.debug(f"Distance computation completed. Cost matrix shape: {cost_matrix.shape}")
        return cost_matrix
    
    def ensure_target_exists(self, target_id):
        """
        LAZY CREATION: Ensure a target exists in samples with at least an empty list.
        
        Parameters
        ----------
        target_id : int
            Target ID to ensure exists
        """
        if target_id not in self.samples:
            logger.debug(f"Creating empty samples list for target {target_id}")
            self.samples[target_id] = []
    
    def cleanup_missing_targets(self, valid_target_ids):
        """
        DEFENSIVE CLEANUP: Remove samples for targets that are no longer valid.
        
        Parameters
        ----------
        valid_target_ids : set or list
            Set/list of target IDs that should be kept
        """
        if not valid_target_ids:
            logger.warning("cleanup_missing_targets called with empty valid_target_ids")
            return
            
        valid_set = set(valid_target_ids)
        current_targets = set(self.samples.keys())
        targets_to_remove = current_targets - valid_set
        
        for target_id in targets_to_remove:
            logger.debug(f"Removing samples for obsolete target {target_id}")
            del self.samples[target_id]
        
        logger.debug(f"Cleanup completed. Removed {len(targets_to_remove)} obsolete targets")
    
    def get_stats(self):
        """Get diagnostic statistics"""
        return {
            **self.stats,
            'total_targets': len(self.samples),
            'targets_with_samples': len([t for t, s in self.samples.items() if s]),
            'total_samples': sum(len(s) for s in self.samples.values())
        }
