from _mycollections import mydefaultdict
from mydouble import mydouble, counts
from collections import defaultdict


class FeatureLookup:
    def __init__(self):
        None

    def __call__(self, feature_vector):
        """
        Converts a dictionary of features (string keys or anything else) and
        values (floats) into a standardized feature vector.  Removes infrequent
        features and standardizes values.
        """
        val = mydefaultdict(mydouble, feature_vector)
        return val


class FilterFeatureLookup(FeatureLookup):

    def __init__(self):
        self._fixed = False
        self._sum = defaultdict(float)
        self._sq_sum = defaultdict(float)

        self._count = defaultdict(int)
        self._all_obs = 0
        self._lookup = None
        self._mean = None
        self._std = None

    def add_observation(self, features):
        for ii in features:
            val = features[ii]
            # val >= 0, "We can't handle negative feature %s" % str(ii)
            if val > 0:
                self._sum[ii] += val
                self._sq_sum[ii] += val * val
                self._count[ii] += 1
        self._all_obs += 1

    def create_lookup(self, limit, min_count):
        total_features = 0
        self._mean = {}
        self._std = {}
        self._lookup = {}

        try:
            for sum, ii in sorted((-1.0 * self._count[x], x) for x in self._sum):
                if self._count[ii] < min_count:
                    del self._count[ii]

                feature_id = total_features
                total_features += 1
                assert not ii in self._lookup, \
                    "Feature %s already added" % str(ii)
                self._lookup[ii] = feature_id

                total_obs = float(self._all_obs)
                mean = self._sum[ii] / total_obs

                self._mean[feature_id] = mean
                self._std[feature_id] = self._sq_sum[ii] / total_obs - mean * mean

                if total_features > limit:
                    break
        except UnicodeDecodeError:
            print 'unicode problem'



        del self._sq_sum
        del self._sum

        self._fixed = True
        return self

    def normalize(self, feat, val):

        assert feat in self._mean, "Feature %i not in lookup" % feat

        if self._count[feat] > 1:
            x = val
            m = self._mean[feat]
            s = self._std[feat]

            return (x - m) / s
        else:
            return val

    def __call__(self, feature_vector):
        """
        Converts a dictionary of features (string keys or anything else) and
        values (floats) into a standardized feature vector.  Removes infrequent
        features and standardizes values.
        """
        val = mydefaultdict(mydouble)
        for ii in feature_vector:
            if ii in self._lookup:
                feat = self._lookup[ii]
                val[feat] = self.normalize(feat, feature_vector[ii])
        return val
