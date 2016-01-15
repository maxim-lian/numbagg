import numpy as np
from numba import jit

from .decorators import ndreduce


@ndreduce(['bool_(float32)', 'bool_(float64)'])
def allnan(a):
    f = True
    for ai in a.flat:
        if not np.isnan(ai):
            f = False
            break
    return f


@ndreduce(['bool_(float32)', 'bool_(float64)'])
def anynan(a):
    f = False
    for ai in a.flat:
        if np.isnan(ai):
            f = True
            break
    return f


@ndreduce(['int64(float32)', 'int64(float64)'])
def count(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing


@ndreduce(['int64(int32)', 'int64(int64)', 'float32(float32)', 'float64(float64)'])
def nansum(a):
    asum = 0.0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
    return asum


@ndreduce
def nanmean(a):
    asum = 0.0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > 0:
        return asum / count
    else:
        return np.nan


@ndreduce
def nanstd(a):
    # for now, fix ddof=0
    ddof = 0
    asum = 0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > ddof:
        amean = asum / count
        asum = 0
        for ai in a.flat:
            if not np.isnan(ai):
                ai -= amean
                asum += (ai * ai)
        return np.sqrt(asum / (count - ddof))
    else:
        return np.nan


@ndreduce
def nanvar(a):
    # for now, fix ddof=0
    ddof = 0
    asum = 0
    count = 0
    for ai in a.flat:
        if not np.isnan(ai):
            asum += ai
            count += 1
    if count > ddof:
        amean = asum / count
        asum = 0
        for ai in a.flat:
            if not np.isnan(ai):
                ai -= amean
                asum += (ai * ai)
        return asum / (count - ddof)
    else:
        return np.nan


@ndreduce(['int64(float32)', 'int64(float64)'])
def nanargmax(a):
    if not a.size:
        raise ValueError('numpy.nanargmax raises on a.shape[axis]==0; '
                         'numbagg too')
    amax = -np.infty
    # use -1 as a sentinel value for "not found"
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai > amax or (idx == -1 and not np.isnan(ai)):
            amax = ai
            idx = i
    if idx == -1:
        raise ValueError('All-NaN slice encountered')
    return idx


@ndreduce(['int64(float32)', 'int64(float64)'])
def nanargmin(a):
    if not a.size:
        raise ValueError('numpy.nanargmin raises on a.shape[axis]==0; '
                         'numbagg too')
    amin = np.infty
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai < amin or (idx == -1 and not np.isnan(ai)):
            amin = ai
            idx = i
    if idx == -1:
        raise ValueError('All-NaN slice encountered')
    return idx


@ndreduce
def nanmax(a):
    if not a.size:
        raise ValueError('numpy.nanmax raises on a.shape[axis]==0; '
                         'numbagg too')
    amax = -np.infty
    all_missing = 1
    for ai in a.flat:
        if ai >= amax:
            amax = ai
            all_missing = 0
    if all_missing:
        amax = np.nan
    return amax


@ndreduce
def nanmin(a):
    if not a.size:
        raise ValueError('numpy.nanmin raises on a.shape[axis]==0; '
                         'numbagg too')
    amin = np.infty
    all_missing = 1
    for ai in a.flat:
        if ai <= amin:
            amin = ai
            all_missing = 0
    if all_missing:
        amin = np.nan
    return amin


@jit(nopython=True)
def flatten(a):
    b = np.empty(a.size, a.dtype)
    for i, ai in enumerate(a.flat):
        b[i] = ai
    return b


@ndreduce(['f8(i4)', 'f8(i8)', 'f4(f4)', 'f8(f8)'])
def median(a):
    n0 = a.size
    if n0 == 0:
        return np.nan
    b = flatten(a)
    k = n0 >> 1
    l = 0
    r = n0 - 1
    while l < r:
        x = b[k]
        i = l
        j = r
        while 1:
            while b[i] < x: i += 1
            while x < b[j]: j -= 1
            if i <= j:
                tmp = b[i]
                b[i] = b[j]
                b[j] = tmp
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j
    if n0 % 2 == 0:
        amax = -np.infty
        for i in range(k):
            ai = b[i]
            if ai >= amax:
                amax = ai
        return 0.5 * (b[k] + amax)
    else:
        return b[k]


@ndreduce
def nanmedian(a):
    n0 = a.size
    if n0 == 0:
        return np.nan
    b = flatten(a)
    j = n0 - 1
    flag = 1
    for i in range(n0):
        if b[i] != b[i]:
            while b[j] != b[j]:
                if j <= 0:
                    break
                j -= 1
            if i >= j:
                flag = 0
                break
            tmp = b[i]
            b[i] = b[j]
            b[j] = tmp
    n = i + flag
    k = n >> 1
    l = 0
    r = n - 1
    while l < r:
        x = b[k]
        i = l
        j = r
        while 1:
            while b[i] < x: i += 1
            while x < b[j]: j -= 1
            if i <= j:
                tmp = b[i]
                b[i] = b[j]
                b[j] = tmp
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j
    if n % 2 == 0:
        amax = -np.infty
        allnan = 1
        for i in range(k):
            ai = b[i]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:
            return 0.5 * (b[k] + amax)
        else:
            return b[k]
    else:
        return b[k]
