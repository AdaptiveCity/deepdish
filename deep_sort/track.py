# vim: expandtab:ts=4:sw=4
from collections import Counter
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, detection):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = [detection.feature]
        self.labels = [detection.label]
        self.dist = {}
        self.dist[detection.label] = [detection.confidence]
        self.detections = [detection]
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        # add a vote for this particular label
        self.labels.append(detection.label)
        if detection.label not in self.dist:
            self.dist[detection.label] = []
        self.dist[detection.label].append(detection.confidence)
        self.detections.append(detection)

    def get_label(self, return_confidence=False):
        # the label of a track is designated as the most commonly
        # identified object associated with the track (with one
        # exception)
        if not self.labels:
            if return_confidence: return None, 0
            else: return None
        else:
            # Workaround for poor object recognition of motorbikes vs bicycles.
            #

            dists=[(lbl, len(scrs), np.average(scrs)) for (lbl, scrs) in self.dist.items()]
            alphas=np.array([avg for (_, _, avg) in dists])
            c=np.array([cnt for (_, cnt, _) in dists])
            lbls=[lbl for (lbl, _, _) in dists]

            # Expected value of a Multinomial with Dirichlet priors.
            # towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815
            expected=list(reversed(sorted(zip((alphas+c)/(c.sum() + alphas.sum()), lbls))))

            # how much to lean in favour of assuming a 'motorbike' is actually a 'bicycle'
            factor=4 # FIXME (experimental): needs testing

            if len(expected) > 1:
                if expected[0][1] == 'motorbike' and expected[1][1] == 'bicycle':
                    # if probability of motorbike greatly exceeds that of bicycle
                    if expected[0][0] > expected[1][0] * factor:
                        if return_confidence: return 'motorbike', np.average(self.dist['motorbike'])
                        else: return 'motorbike'
                    else:
                        if return_confidence: return 'bicycle', np.average(self.dist['bicycle'])
                        else: return 'bicycle'
            # otherwise, use most likely
            if return_confidence: return expected[0][1], np.average(self.dist[expected[0][1]])
            else: return expected[0][1]

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
