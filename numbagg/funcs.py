import numpy as np

from .decorators import ndreduce


@ndreduce(['float32,bool_', 'float64,bool_'])
def allnan(a):
    f = True
    for ai in a.flat:
        if not np.isnan(ai):
            f = False
            break
    return f


@ndreduce(['float32,bool_', 'float64,bool_'])
def anynan(a):
    f = False
    for ai in a.flat:
        if np.isnan(ai):
            f = True
            break
    return f


@ndreduce(['float32,int64', 'float64,int64'])
def count(a):
    non_missing = 0
    for ai in a.flat:
        if not np.isnan(ai):
            non_missing += 1
    return non_missing


@ndreduce
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


@ndreduce(['float32,int64', 'float64,int64'])
def nanargmax(a):
    amax = -np.infty
    # we can't raise in numba's nopython mode, so use -1 as a sentinel value
    # for "not found" (pandas uses the same convention)
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai > amax:
            amax = ai
            idx = i
    if idx == -1:
        # take another pass to look for -infty
        for i, ai in enumerate(a.flat):
            if ai == amax:
                idx = i
                break
    return idx


@ndreduce(['float32,int64', 'float64,int64'])
def nanargmin(a):
    amin = np.infty
    idx = -1
    for i, ai in enumerate(a.flat):
        if ai < amin:
            amin = ai
            idx = i
    if idx == -1:
        # take another pass to look for +infty
        for i, ai in enumerate(a.flat):
            if ai == amin:
                idx = i
                break
    return idx


@ndreduce
def nanmax(a):
    amax = -np.infty
    allnan = 1
    for ai in a.flat:
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan:
        amax = np.nan
    return amax


@ndreduce
def nanmin(a):
    amin = np.infty
    allnan = 1
    for ai in a.flat:
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan:
        amin = np.nan
    return amin
