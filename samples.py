# -*- coding: utf-8 -*-

import random
import numpy as np
from scipy import stats
import plots


def random_trials_prevalence(n, successes=10, min_prev=0.1, max_prev=10, prev_steps=100, 
                             min_sample_size=100, max_sample_size=None, sample_steps=None):
    # conver percent prevalence extremes
    min_prev, max_prev = min_prev / 100, max_prev / 100
    
    # log-spaced range of prevalences
    prevalences = np.logspace(min_prev, max_prev, prev_steps, endpoint=True)
    prevalences = np.log10(prevalences)
    
    # at most draw 5% random samples if max sampling effort not provided
    max_sample_size = int(0.05 * n) if not max_sample_size else max_sample_size
    # divide sampling space to 10 steps, unless user provided
    sample_steps = int(max_sample_size / 10) if sample_steps == None else sample_steps 
    # linear spaced x-axis (number of samples)
    sample_sizes = np.linspace(min_sample_size, max_sample_size, sample_steps, endpoint=True, dtype='int32')
    
    output = list()
    for p in prevalences:
        r = int(p*n)
        p = round(p, 3)
        x = np.hstack([np.ones(r), np.zeros(n-r)])
        np.random.shuffle(x)
        for size in sample_sizes:
            draws = np.random.choice(x, size, replace=False)
            num_ones = np.sum(draws, dtype='int32')
            if num_ones >= successes:
                output.append((p, r, num_ones, size))
                break
    
    if output:
        plots.plot_rs_prevalence(output, successes)
    
    return output


def random_uniform_sample(y, n):
    id2y = {i:yi for i, yi in enumerate(y)}
    drawn = {0: list(), 1: list()}
    rcount = 0
    while True:
        i = random.choice(list(id2y.keys()))
        if id2y[i] == 1:
            rcount += 1
            drawn[1].append(i)
        else:
            drawn[0].append(i)
        if rcount == n:
            break
        id2y = {j:yj for j, yj in id2y.items() if j!=i}
    return drawn


def clopper_pearson_ci(x, n, alpha=0.05):
    """
    x: number of successes
    n: number of trials
    alpha: 1 - confidence_level
    """
    ci_low = stats.beta.ppf(alpha / 2, x, n - x + 1)
    ci_upp = stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    q_ = x * 1. / n
    
    if np.ndim(ci_low) > 0:
        ci_low[q_ == 0] = 0
        ci_upp[q_ == 1] = 1
    else:
        ci_low = ci_low if (q_ != 0) else 0
        ci_upp = ci_upp if (q_ != 1) else 1
    
    return ci_low, ci_upp
