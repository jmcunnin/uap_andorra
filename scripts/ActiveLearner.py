from collections import defaultdict, Counter
from functools import partial
from operator import itemgetter
import editdistance


def score_lsts(to_compare, path):
    s1, s2 = set(to_compare), set(path)
    return len(s1.intersection(s2))/len(s1.union(s2))


def _get_nearest(path, paths, k, return_scores=False):
    if return_scores:
        scored = sorted([(p, score_lsts(p, path)) for p in paths], key=itemgetter(1))
        return scored[:k]
    # score_partial = partial(score_lsts, path=path)
    score_partial = partial(editdistance.eval, path)
    return sorted(paths, key=score_partial)[:k]


class ActiveLearner:

    def __init__(self):
        self.paths_bynat = defaultdict(list)

    def add_path(self, nationality, path):
        self.paths_bynat[nationality].append(path)

    def get_knn(self, nationality, path, k):
        nat_paths = self.paths_bynat.get(nationality, list())
        top_paths = _get_nearest(path, nat_paths, k)
        new_stops = [x for l in top_paths for x in l if x not in set(path)]
        return [v[0] for v in Counter(new_stops).most_common(10)]

