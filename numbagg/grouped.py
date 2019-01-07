from functools import partial

import numpy as np
from numba import float32, float64, int32, int64

from .decorators import groupndreduce


def group_wrapper(func):
    @groupndreduce(
        [
            (float64, int64, float64),
            (float64, int32, float64),
            (float32, int64, float32),
            (float32, int32, float32),
        ]
    )
    def group_func(values, labels, out):
        labels = labels.ravel()
        values = values.ravel()
        grouped = []
        for i in range(out.size):
            # nan required in order to satisfy numba type inference
            grouped.append([np.nan])

        for l, v in zip(labels, values):
            if l >= 0:
                grouped[l].append(v)

        for i in range(out.size):
            group = grouped[i]
            out[i] = func(np.array(group))

    return group_func


group_nanmean = group_wrapper(np.nanmean)
group_nansum = group_wrapper(np.nansum)
group_nanstd = group_wrapper(np.nanstd)
