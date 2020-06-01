#!/usr/bin/env python3
"""
Contains required modules to process simulations

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""

import os
import sys
import csv
import allel
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def read_msms(filename, NCHROMS, N):
    """
    Reads msms file to an haplotype matrix

    Parameters:
        filename: full path and name of the .txt MSMS file
        NCHROMS: number of samples(haploid individuals, or chromosoms
        N: length of the simulated sequence(bp)
    Output:
        Returns an haplotype array, and an array containing positions
    """
    file = open(filename).readlines()
    if len(file) == 0:
        raise Exception('The file {} is empty'.format(filename.split('/')[-1]))
    # look for the // character in the file
    pointer = file.index('//\n') + 3

    # Get positions
    pos = file[pointer - 1].split()
    del pos[0]
    pos = np.array(pos, dtype='float')
    pos = pos * N

    # Get the number of genomic positions(determined be the number or pointers)
    n_columns = len(list(file[pointer])) - 1
    # Intialize the empty croms matrix: of type: object
    croms = np.empty((NCHROMS, n_columns), dtype=np.object)
    # Fill the matrix with the simulated data
    for j in range(NCHROMS):
        f = list(file[pointer + j])
        del f[-1]
        F = np.array(f)
        croms[j, :] = F
    croms = croms.astype(int)
    return croms, pos


def rearrange_neutral(croms, pos, length):
    """
    rearranges neutral simulations such that each simulation results in
    <length>bp in length with selected mutation at the center

    Parameters:
        croms: input haplotype matrix
        pos: array containing position information for croms
        length: desired length of the output(bp)
    Returns:
        a haplotype matrix that is <length_out> bp in length and target snp at center,
        and an array containing new positions
    """
    freqs = np.true_divide(np.sum(croms, axis=0), croms.shape[0])
    # positions of mutations within [0.4,0.6]
    poss = pos[np.logical_and(freqs > 0.4, freqs < 0.6)]
    # position of target mutation
    pos_mut = poss[len(poss) // 2]
    # upper and lower boundaries of the region that will be selected
    up_bound = pos_mut + length / 2
    low_bound = pos_mut - length / 2
    target_range = np.logical_and(pos > low_bound, pos < up_bound)
    pos_new = pos[target_range] + length / 2 - pos_mut
    croms_new = croms[:, target_range]
    return croms_new, pos_new


def sort_freq(im_matrix):
    """
    This function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted
    by genetic similarity.

    Parameters:
        im_matrix: Array containing sequence data
    Returns:
        Sorted array containing sequence data
    """
    # u: Sorted Unique arrays
    # index: Index of 'im_matrix' that corresponds to each unique array
    # count: The number of instances each unique array appears in the 'im_matrix'
    u, index, count = np.unique(im_matrix, return_index=True, return_counts=True, axis=0)
    # b: Intitialised matrix the size of the original im_matrix[where new sorted data will be stored]
    b = np.zeros((np.size(im_matrix, 0), np.size(im_matrix, 1)), dtype=int)
    # c: Frequency table of unique arrays and the number of times they appear in the original 'im_matrix'
    c = np.stack((index, count), axis=-1)
    # The next line sorts the frequency table based mergesort algorithm
    c = c[c[:, 1].argsort(kind='mergesort')]
    pointer = np.size(im_matrix, 0) - 1
    for j in range(np.size(c, 0)):
        for conta in range(c[j, 1]):
            b[pointer, :] = im_matrix[c[j, 0]]
            pointer -= 1
    return b


def sort_min_diff(im_matrix):
    """
    This function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted
    by genetic similarity. this problem is NP, so here we use a nearest neighbors approx.  it's not perfect,
    but it's fast and generally performs ok.
    Implemented from https://github.com/flag0010/pop_gen_cnn/blob/master/sort.min.diff.py#L1

    Parameters:
        im_matrix: haplotype matrix (np array)

    Returns:
        Sorted numpy array
    """
    mb = NearestNeighbors(len(im_matrix), metric='manhattan').fit(im_matrix)
    v = mb.kneighbors(im_matrix)
    smallest = np.argmin(v[0].sum(axis=1))
    return im_matrix[v[1][smallest]]


def order_data(im_matrix, pos, sort, method):
    """
    Sorts haplotype matrix

    Parameters:
        im_matrix: input haplotype matrix
        pos: position of target SNP. not required if method = t
        sort: sorting method. either
            gen_sim: based on genetic similarity, or
            freq: based on frequency
        method: either
            t: together. sort the whole array together
            s: seperate. sort two haplotype groups seperately

    Returns:
        sorted haplotype matrix
    """
    if method == "t":
        if sort == "gen_sim":
            croms = sort_min_diff(im_matrix)
        elif sort == "freq":
            croms = sort_freq(im_matrix)
        else:
            raise ValueError("sort must be either 'freq' or 'gen_sim'")

    elif method == "s":
        if not isinstance(pos, int):
            raise ValueError("Position of the target SNP must be an integer")
        index_1 = (im_matrix[:, pos] == 1).reshape(im_matrix.shape[0], )
        index_0 = (im_matrix[:, pos] == 0).reshape(im_matrix.shape[0], )
        croms_1 = im_matrix[index_1, :]
        croms_0 = im_matrix[index_0, :]
        if sort == "gen_sim":
            croms1 = sort_min_diff(croms_1)
            croms0 = sort_min_diff(croms_0)
        elif sort == "freq":
            croms1 = sort_freq(croms_1)
            croms0 = sort_freq(croms_0)
        else:
            raise ValueError("sort must be either 'freq' or 'gen_sim'")
        croms = np.concatenate((croms0, croms1), axis=0)
    return croms


def sim_to_matrix(filename, NCHROMS, N, N_NE, sort, method):
    """
    Generates ordered haplotype_matrix from simulation results(must be ms format in .txt)

    Parameters:
        filename: full path and name of the .txt MSMS file
        NCHROMS: number of samples(haploid individuals, or chromosoms)
        N: length of the simulated sequence(bp)
        sort: sorting method. either:
            gen_sim: based on genetic similarity
            freq: based on frequency
        method: sorting method. either:
            t: together. sort the whole array together
            s: seperate. sort two haplotype groups seperately

    Returns:
        ordered haplotype matrix, where columns are positions and rows are individuals
    """
    if filename.split("/")[-1].startswith("NE"):
        crom, pos = read_msms(filename, NCHROMS, N_NE)
        croms, positions = rearrange_neutral(crom, pos, N)
    else:
        croms, positions = read_msms(filename, NCHROMS, N)

    pos = np.where(np.abs(positions - N / 2) < 1)[0]
    if len(pos) == 0:
        print(filename)
        print(positions)
        raise IndexError("Target SNP not found")
    if len(pos) > 1:
        print("Target SNP found at multiple positions:")
        print(positions[pos])
        pos = np.array([pos[0]], dtype='int64')
        print(filename)
    pos = int(pos[0])

    sorted_croms = order_data(croms, pos, sort, method)
    return sorted_croms


def matrix_to_image(croms, n_row, n_col):
    """
    Generates image from sorted haplotype matrix and resizes image into (n_row, ncol)

    Parameters:
        croms: Haplotype array
        n_row: number of rows
        n_col: number of cols

    Returns:
        resized image
    """
    # Generate image
    all1 = np.ones(croms.shape)
    cromx = all1 - croms
    bw_croms_uint8 = np.uint8(cromx)
    bw_croms_im = Image.fromarray(bw_croms_uint8 * 255, mode='L')
    # Resize
    im_resized = bw_croms_im.resize((n_col, n_row))
    return im_resized


def sim_to_image(path_to_sim, path_to_image, SIM_FROM, SIM_TO, NCHROMS, N, N_NE, img_dim=(128, 128),
                 clss=("NE", "IS", "OD", "FD"), sort="freq", method="s"):
    """
    Converts MSMS simulation output files into images

    Parameters:
        path_to_sim: path to the folder containing simulation files
        path_to_image: path to a folder in which output images will be saved
        SIM_FROM: number of simulations (replicate), starting from
        SIM_TO: number of simulations, until
        NCHROMS: number of samples(haploid individuals, or chromosomes)
        N: length of the simulated sequence(bp) for selection scenarios
        N_NE: length of simulated sequence(bp) for neutral scenario
        img_dim: image dimensions (nrow, ncol)
        clss: a tuple of target classes-
            "NE": neutral
            "IS": incomplete sweep
            "OD": overdominance
            "FD": negative freq-dependent selection
        sort: sorting algorithm. either
            gen_sim: based on genetic similarity
            freq: based on frequency
        method: either
            t: together. sort the whole array together
            s: seperate. sort two haplotype groups seperately

    """
    if sys.version_info >= (3, 6):
        with os.scandir(path_to_sim) as fdir:
            for file in fdir:
                if file.name.startswith(tuple(clss)) and file.is_file() and int(file.name.replace(".txt", "").split("_")[-1]) in range(SIM_FROM, SIM_TO+1):
                    croms = sim_to_matrix(file.path, NCHROMS, N, N_NE, sort=sort, method=method)
                    im_resized = matrix_to_image(croms, n_row=img_dim[0], n_col=img_dim[1])
                    im_resized.save("{}{}.bmp".format(path_to_image, file.name.replace(".txt", "")))
    else:
        files = [file for file in os.scandir(path_to_sim)
                 if file.is_file()
                 if file.name.startswith(tuple(clss))
                 if int(file.name.replace(".txt", "").split("_")[-1]) in range(SIM_FROM, SIM_TO + 1)]

        for file in files:
            croms = sim_to_matrix(file.path, NCHROMS, N, N_NE, sort=sort, method=method)
            im_resized = matrix_to_image(croms, n_row=img_dim[0], n_col=img_dim[1])
            im_resized.save("{}{}.bmp".format(path_to_image, file.name.replace(".txt", "")))
    return 0


def calc_median_r2(g):
    """Calculates median LD r^2"""
    gn = g.to_n_alt(fill=-1)
    LDr = allel.rogers_huff_r(gn)
    LDr2 = LDr ** 2
    median_r2 = np.nanmedian(LDr2)
    return median_r2


def calc_kelly_zns(g, n_pos):
    """Calculates Kelly's Zns statistic"""
    gn = g.to_n_alt(fill=-1)
    LDr = allel.rogers_huff_r(gn)
    LDr2 = LDr ** 2
    kellyzn = (np.nansum(LDr2) * 2.0) / (n_pos * (n_pos - 1.0))
    return kellyzn


def calc_pi(croms):
    """Calculates pi"""
    dis1 = []
    for i in range(croms.shape[0]):
        d1 = []
        for j in range(i + 1, croms.shape[0]):
            d1.append(sum(croms[i, :] != croms[j, :]))
        dis1.append(sum(d1))
    pi_est1 = (sum(dis1) / ((croms.shape[0] * (croms.shape[0] - 1.0)) / 2.0))
    return pi_est1


def calc_faywu_h(croms):
    """Calculates Fay and Wu's H statistic"""
    n_sam1 = croms.shape[0]
    counts1 = croms.sum(axis=0)
    S_i1 = []
    for i in range(1, n_sam1):
        S_i1.append(sum(counts1 == i))
    i = range(1, n_sam1)
    n_i = np.subtract(n_sam1, i)
    thetaP1 = sum((n_i * i * S_i1 * 2) / (n_sam1 * (n_sam1 - 1.0)))
    thetaH1 = sum((2 * np.multiply(S_i1, np.power(i, 2))) / (n_sam1 * (n_sam1 - 1.0)))
    Hstat1 = thetaP1 - thetaH1
    return Hstat1


def calc_fuli_f_star(croms):
    """Calculates Fu and Li's D* statistic"""
    n_sam1 = croms.shape[0]
    n_pos1 = np.size(croms, 1)
    an = np.sum(np.divide(1.0, range(1, n_sam1)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n_sam1), 2)))
    an1 = an + np.true_divide(1, n_sam1)

    vfs = (((2 * (n_sam1 ** 3.0) + 110.0 * (n_sam1 ** 2.0) - 255.0 * n_sam1 + 153) / (
            9 * (n_sam1 ** 2.0) * (n_sam1 - 1.0))) + ((2 * (n_sam1 - 1.0) * an) / (n_sam1 ** 2.0)) - (
                   (8.0 * bn) / n_sam1)) / ((an ** 2.0) + bn)
    ufs = ((n_sam1 / (n_sam1 + 1.0) + (n_sam1 + 1.0) / (3 * (n_sam1 - 1.0)) - 4.0 / (
            n_sam1 * (n_sam1 - 1.0)) + ((2 * (n_sam1 + 1.0)) / ((n_sam1 - 1.0) ** 2)) * (
                    an1 - ((2.0 * n_sam1) / (n_sam1 + 1.0)))) / an) - vfs

    pi_est = calc_pi(croms)
    ss = sum(np.sum(croms, axis=0) == 1)
    Fstar1 = (pi_est - (((n_sam1 - 1.0) / n_sam1) * ss)) / ((ufs * n_pos1 + vfs * (n_pos1 ** 2.0)) ** 0.5)
    return Fstar1


def calc_fuli_d_star(croms):
    """Calculates Fu and Li's D* statistic"""
    n_sam1 = croms.shape[0]
    n_pos1 = np.size(croms, 1)
    an = np.sum(np.divide(1.0, range(1, n_sam1)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n_sam1), 2)))
    an1 = an + np.true_divide(1, n_sam1)

    cn = (2 * (((n_sam1 * an) - 2 * (n_sam1 - 1))) / ((n_sam1 - 1) * (n_sam1 - 2)))
    dn = (cn + np.true_divide((n_sam1 - 2), ((n_sam1 - 1) ** 2)) + np.true_divide(2, (n_sam1 - 1)) * (
            3.0 / 2 - (2 * an1 - 3) / (n_sam1 - 2) - 1.0 / n_sam1))

    vds = (((n_sam1 / (n_sam1 - 1.0)) ** 2) * bn + (an ** 2) * dn - 2 * (n_sam1 * an * (an + 1)) / (
            (n_sam1 - 1.0) ** 2)) / (an ** 2 + bn)
    uds = ((n_sam1 / (n_sam1 - 1.0)) * (an - n_sam1 / (n_sam1 - 1.0))) - vds

    ss = sum(np.sum(croms, axis=0) == 1)
    Dstar1 = ((n_sam1 / (n_sam1 - 1.0)) * n_pos1 - (an * ss)) / (uds * n_pos1 + vds * (n_pos1 ^ 2)) ** 0.5
    return Dstar1


def calc_zeng_e(croms):
    """Calculates Zeng et al's E statistic"""
    n_sam1 = croms.shape[0]
    n_pos1 = np.size(croms, 1)
    an = np.sum(np.divide(1.0, range(1, n_sam1)))
    bn = np.sum(np.divide(1.0, np.power(range(1, n_sam1), 2)))
    counts1 = croms.sum(axis=0)
    S_i1 = []
    for i in range(1, n_sam1):
        S_i1.append(sum(counts1 == i))
    thetaW = n_pos1 / an
    thetaL = np.sum(np.multiply(S_i1, range(1, n_sam1))) / (n_sam1 - 1.0)
    theta2 = (n_pos1 * (n_pos1 - 1.0)) / (an ** 2 + bn)

    var1 = (n_sam1 / (2.0 * (n_sam1 - 1.0)) - 1.0 / an) * thetaW
    var2 = theta2 * (bn / (an ** 2.0)) + 2 * bn * (n_sam1 / (n_sam1 - 1.0)) ** 2.0 - (
            2.0 * (n_sam1 * bn - n_sam1 + 1.0)) / ((n_sam1 - 1.0) * an) - (3.0 * n_sam1 + 1.0) / (
               (n_sam1 - 1.0))
    varlw = var1 + var2

    ZengE1 = (thetaL - thetaW) / (varlw) ** 0.5
    return ZengE1


def calc_rageddness(croms):
    """Calculates rageddness index"""
    mist = []
    for i in range(croms.shape[0] - 1):
        for j in range(i + 1, croms.shape[0]):
            mist.append(sum(croms[i, :] != croms[j, :]))
    mist = np.array(mist)
    lnt = mist.shape[0]
    fclass = []
    for i in range(1, np.max(mist) + 2):
        fclass.append((np.true_divide(sum(mist == i), lnt) - np.true_divide(sum(mist == (i - 1)), lnt)) ** 2)
    rgd1 = np.sum(fclass)
    return rgd1


def sum_stats(croms, pos, NCHROMS, sname):
    """
    Calculates summary statistics

    Parameters:
        croms: haplotype matrix
        pos: positions
        sname: simulation name
        NCHROMS: number of chromosomes

    Returns:
        A list of labels (names of statistics)
        A list of values
    """
    cls = sname.split('_')[0]
    time = sname.split('_')[1]
    r = sname.split('_')[-1]

    # SUMMARY STATISTICS
    # REGION 1: central 10kb([20kb:30kb])
    pos1 = pos[np.logical_and(pos > 20000, pos < 30000)]
    croms1 = croms[:, np.logical_and(pos > 20000, pos < 30000).tolist()]
    n_pos1 = np.size(croms1, 1)
    freq1 = np.true_divide(croms1.sum(axis=0), NCHROMS)
    freq1 = np.array(freq1)

    haplos = np.transpose(croms1)
    h1 = allel.HaplotypeArray(haplos)
    ac1 = h1.count_alleles()
    g1 = h1.to_genotypes(ploidy=2, copy=True)

    # mean_pairwise_distance
    mean_mean_pwise_dis1 = np.mean(allel.mean_pairwise_difference(ac1))
    median_mean_pwise_dis1 = np.median(allel.mean_pairwise_difference(ac1))
    max_mean_pwise_dis1 = np.max(allel.mean_pairwise_difference(ac1))

    # tajimasd
    TjD1 = allel.tajima_d(ac1)

    # watterson
    theta_hat_w1 = allel.watterson_theta(pos1, ac1)

    # heterogeneity
    obs_het1 = allel.heterozygosity_observed(g1)
    af1 = ac1.to_frequencies()
    exp_het1 = allel.heterozygosity_expected(af1, ploidy=2)
    mean_obs_het1 = np.mean(obs_het1)
    median_obs_het1 = np.median(obs_het1)
    max_obs_het1 = np.max(obs_het1)
    # return 0, if 0/0 encountered
    ob_exp_het1 = np.divide(obs_het1, exp_het1, out=np.zeros_like(obs_het1), where=exp_het1 != 0)
    mean_obs_exp1 = np.nanmean(ob_exp_het1)
    median_obs_exp1 = np.nanmedian(ob_exp_het1)
    max_obs_exp1 = np.nanmax(ob_exp_het1)

    # LD r
    median_r21 = calc_median_r2(g1)

    # Haplotype_stats
    hh1 = allel.garud_h(h1)
    h11 = hh1[0]
    h121 = hh1[1]
    h1231 = hh1[2]
    h2_h11 = hh1[3]
    n_hap1 = np.unique(croms1, axis=0).shape[0]
    hap_div1 = allel.haplotype_diversity(h1)

    ehh1 = allel.ehh_decay(h1)
    mean_ehh1 = np.mean(ehh1)
    median_ehh1 = np.median(ehh1)

    ihs1 = allel.ihs(h1, pos1, include_edges=True)
    median_ihs1 = np.nanmedian(ihs1)

    # nsl
    nsl1 = allel.nsl(h1)
    max_nsl1 = np.nanmax(nsl1)
    median_nsl1 = np.nanmedian(nsl1)

    # NCD
    n = n_pos1
    tf = 0.5
    ncd21 = (sum((freq1 - tf) ** 2) / n) ** 0.5
    freq11 = freq1[freq1 < 1]
    n1 = freq11.shape[0]
    ncd11 = (sum((freq11 - tf) ** 2) / n1) ** 0.5

    # kellyZns
    kellyzn1 = calc_kelly_zns(g1, n_pos1)

    # pi
    pi_est1 = calc_pi(croms1)

    # FayWusH
    Hstat1 = calc_faywu_h(croms1)

    # of singletons
    Ss1 = sum(np.sum(croms1, axis=0) == 1)

    # fu_li Dstar
    Dstar1 = calc_fuli_d_star(croms1)

    # fu_li Fstar
    Fstar1 = calc_fuli_f_star(croms1)

    # Zeng_E
    ZengE1 = calc_zeng_e(croms1)

    # rageddness index
    rgd1 = calc_rageddness(croms1)

    ###################################################################
    # REGION 2: 20kb regions far from selected site([0:20kb]&[30:50kb])
    ###################################################################
    pos2 = pos[np.logical_or(pos <= 20000, pos >= 30000)]
    croms2 = croms[:, np.logical_or(pos <= 20000, pos >= 30000).tolist()]
    n_pos2 = np.size(croms2, 1)
    freq2 = np.true_divide(croms2.sum(axis=0), NCHROMS)
    freq2 = np.array(freq2)

    haplos = np.transpose(croms2)
    h2 = allel.HaplotypeArray(haplos)
    ac2 = h2.count_alleles()
    g2 = h2.to_genotypes(ploidy=2, copy=True)

    # mean_pairwise_distance
    mean_mean_pwise_dis2 = np.mean(allel.mean_pairwise_difference(ac2))
    median_mean_pwise_dis2 = np.median(allel.mean_pairwise_difference(ac2))
    max_mean_pwise_dis2 = np.max(allel.mean_pairwise_difference(ac2))

    # tajimasd
    TjD2 = allel.tajima_d(ac2)

    # watterson
    theta_hat_w2 = allel.watterson_theta(pos2, ac2)

    # heterogeneity
    obs_het2 = allel.heterozygosity_observed(g2)
    af2 = ac2.to_frequencies()
    exp_het2 = allel.heterozygosity_expected(af2, ploidy=2)
    mean_obs_het2 = np.mean(obs_het2)
    median_obs_het2 = np.median(obs_het2)
    max_obs_het2 = np.max(obs_het2)
    # Return 0, if 0/0 encountered
    ob_exp_het2 = np.divide(obs_het2, exp_het2, out=np.zeros_like(obs_het2), where=exp_het2 != 0)
    mean_obs_exp2 = np.nanmean(ob_exp_het2)
    median_obs_exp2 = np.nanmedian(ob_exp_het2)
    max_obs_exp2 = np.nanmax(ob_exp_het2)

    # LD r
    median_r2 = calc_median_r2(g2)

    # Haplotype_stats
    hh2 = allel.garud_h(h2)
    h12 = hh2[0]
    h122 = hh2[1]
    h1232 = hh2[2]
    h2_h12 = hh2[3]
    n_hap2 = np.unique(croms2, axis=0).shape[0]
    hap_div2 = allel.haplotype_diversity(h2)

    ehh2 = allel.ehh_decay(h2)
    mean_ehh2 = np.mean(ehh2)
    median_ehh2 = np.median(ehh2)

    ihs2 = allel.ihs(h2, pos2, include_edges=True)
    median_ihs2 = np.nanmedian(ihs2)

    # NCD
    n = n_pos2
    tf = 0.5
    ncd22 = (sum((freq2 - tf) ** 2) / n) ** 0.5
    freq12 = freq2[freq2 < 1]
    n1 = freq12.shape[0]
    ncd12 = (sum((freq12 - tf) ** 2) / n1) ** 0.5

    # nsl
    nsl2 = allel.nsl(h2)
    max_nsl2 = np.nanmax(nsl2)
    median_nsl2 = np.nanmedian(nsl2)

    # kellyZns
    kellyzn2 = calc_kelly_zns(g2, n_pos2)

    # pi
    pi_est2 = calc_pi(croms2)

    # FayWusH
    Hstat2 = calc_faywu_h(croms2)

    # of singletons
    Ss2 = sum(np.sum(croms2, axis=0) == 1)

    # fu_li Dstar
    Dstar2 = calc_fuli_d_star(croms2)

    # fu_li Fstar
    Fstar2 = calc_fuli_f_star(croms2)

    # Zeng_E
    ZengE2 = calc_zeng_e(croms2)

    # rageddness index
    rgd2 = calc_rageddness(croms2)

    stats = [str(cls), str(time), str(r), mean_mean_pwise_dis1, median_mean_pwise_dis1, max_mean_pwise_dis1,
             TjD1, theta_hat_w1, mean_obs_het1, median_obs_het1, max_obs_het1, mean_obs_exp1, median_obs_exp1,
             max_obs_exp1, median_r21, h11, h121, h1231, h2_h11, hap_div1, n_hap1, mean_ehh1, median_ehh1, median_ihs1,
             max_nsl1, median_nsl1, ncd11, ncd21, kellyzn1, pi_est1, Hstat1, Ss1, Dstar1, Fstar1, ZengE1, rgd1,
             #
             mean_mean_pwise_dis2, median_mean_pwise_dis2, max_mean_pwise_dis2,
             TjD2, theta_hat_w2, mean_obs_het2, median_obs_het2, max_obs_het2, mean_obs_exp2, median_obs_exp2,
             max_obs_exp2, median_r2, h12, h122, h1232, h2_h12, hap_div2, n_hap2, mean_ehh2, median_ehh2, median_ihs2,
             max_nsl2, median_nsl2, ncd12, ncd22, kellyzn2, pi_est2, Hstat2, Ss2, Dstar2, Fstar2, ZengE2, rgd2]

    labs = ['Class', 'Time', 'Iteration', 'Mean(MeanPwiseDist)1', 'Median(MeanPwiseDist)1', 'Max(MeanPwiseDist)1',
            'Tajimas D1', 'Watterson1', 'Mean(ObservedHet)1', 'Median(ObservedHet)1', 'Max(ObservedHet)1',
            'Mean(Obs/Exp Het)1', 'Median(Obs/Exp Het)1', 'Max(Obs/Exp Het)1', 'Median(r2)1', 'H1_1', 'H12_1',
            'H123_1', 'H2/H1_1', 'Haplotype Diversity1', '# of Hap1', 'Mean(EHH)1', 'Median(EHH)1', 'Median(ihs)1',
            'Max(nsl)1', 'Median(nsl)1', 'NCD1_1', 'NCD2_1', 'KellyZns1', 'pi1', 'faywuH1', '#ofSingletons1',
            'Dstar1', 'Fstar1', 'ZengE1', 'Rageddnes1',
            #
            'Mean(MeanPwiseDist)2', 'Median(MeanPwiseDist)', 'Max(MeanPwiseDist)2', 'Tajimas D2', 'Watterson2',
            'Mean(ObservedHet)2', 'Median(ObservedHet)2', 'Max(ObservedHet)2', 'Mean(Obs/Exp Het)2',
            'Median(Obs/Exp Het)2', 'Max(Obs/Exp Het)2', 'Median(r2)', 'H1_2', 'H12_2', 'H123_2', 'H2/H1_2',
            'Haplotype Diversity_2', '# of Hap2', 'Mean(EHH)2', 'Median(EHH)2', 'Median(ihs)2', 'Max(nsl)2',
            'Median(nsl)2', 'NCD1_2', 'NCD2_2', 'KellyZns2', 'pi2', 'faywuH2', '#ofSingletons2', 'Dstar2',
            'Fstar2', 'ZengE2', 'Rageddnes2']

    return labs, stats


def sim_to_stats(path_to_sim, path_to_stat, clss, NCHROMS, SIM_FROM, SIM_TO, N, N_NE):
    """
    Calculates summary statistics for simulation outputs. Creates .csv file at path_to_stat containing summary
    statistics for specified simulations.

    Parameters:
        path_to_sim: Path to directory where the simulation files exist
        path_to_stat: Path to directory where the summary statistics will be stored
        clss: Class of the simulation(either FD, OD, IS or NE):
            -FD: negative-frequency dependent selection
            -OD: over dominance
            -IS: incomplete sweep
            -NE: neutral
        NCHROMS: number of samples(haploid individuals, or chromosoms)
        SIM_FROM: number of simulations (replicate) -starting from
        SIM_TO: number of simulations -until
        N: length of the simulated sequence(bp) for selection scenarios
        N_NE: length of simulated sequence(bp) for neutral scenario
    """
    if SIM_FROM == 1:
        once = 0
    else:
        once = 1

    files = [file for file in os.scandir(path_to_sim)
             if file.is_file()
             if file.name.startswith(tuple(clss))
             if int(file.name.replace(".txt", "").split("_")[-1]) in range(SIM_FROM, SIM_TO + 1)]

    files = [f.path for f in files]
    for file in sorted(files):
        if file.split("/")[-1].startswith("NE"):
            crom, pos = read_msms(file, NCHROMS, N_NE)
            croms, positions = rearrange_neutral(crom, pos, N)
        else:
            croms, positions = read_msms(file, NCHROMS, N)

        fname = file.split("/")[-1].replace(".txt", "")
        labs, stats = sum_stats(croms, positions, NCHROMS, fname)

        f = open("{}{}.csv".format(path_to_stat, fname.split('_')[0]), 'a+')
        with f:
            writer = csv.DictWriter(f, fieldnames=labs)
            if once == 0:
                writer.writeheader()
                writer.writerow(dict(zip(labs, stats)))
                once = 1
            else:
                writer.writerow(dict(zip(labs, stats)))
    return 0


class BaSe(object):
    """
    Distinguishing between balancing selection and incomplete sweep.
    Includes three possible tests:
        test 1: neutrality vs selection
        test 2: incomplete sweep vs balancing selection
        test 3: overdominance vs negative frequency-dependent selection
    For three possible bins of the time of onset of selection:
        recent: includes selection scenarios ranging from 20k to 26k years old.
        medium: includes selection scenarios ranging from 27k to 33k years old.
        old: includes selection scenarios ranging from 34k to 40k years old.
    """
    
    def __init__(self, test, selection_category, N):
        """
        Parameters:
            test: test number to be performed, either:
                1: neutrality vs selection
                2: incomplete sweep vs balancing selection
                3: overdominance vs negative frequency-dependent selection
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                recent: includes selection scenarios ranging from 20k to 26k years old.
                medium: includes selection scenarios ranging from 27k to 33k years old.
                old: includes selection scenarios ranging from 34k to 40k years old.
            N: sample size for each class.
        """

        if test == 1:
            self.classes = ['NE', 'S']
            self.labels = ['NE', 'IS', 'FD', 'OD']
        elif test == 2:
            self.classes = ['IS', 'BS']
            self.labels = ['IS', 'FD', 'OD']
        elif test == 3:
            self.classes = ['OD', 'FD']
            self.labels = ['OD', 'FD']
        else:
            raise ValueError("Test number must be 1, 2, or 3")
        self.test = test
        if selection_category == "recent":
            self.ss = [i for i in range(20, 27)]
        elif selection_category == "medium":
            self.ss = [i for i in range(27, 34)]
        elif selection_category == "old":
            self.ss = [i for i in range(34, 41)]
        else:
            raise ValueError("'{}' is not defined.".format(selection_category))
        self.N = N

            
class Images(BaSe):
    """
    Loads and preprocesses images that will be used to train ConvNet based classifier.
    """
    
    def __init__(self, test, selection_category, N, img_dim):
        """
        Parameters:
            test: test number to be performed, either:
                1: neutrality vs selection
                2: incomplete sweep vs balancing selection
                3: overdominance vs negative frequency-dependent selection
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                recent: includes selection scenarios ranging from 20k to 26k years old.
                medium: includes selection scenarios ranging from 27k to 33k years old.
                old: includes selection scenarios ranging from 34k to 40k years old.
            N: sample size for each class.
            image size: dimensions of input images in the format of (nrow, ncol)
                                
        """
        super().__init__(test, selection_category, N)
        self.image_row = img_dim[0]
        self.image_col = img_dim[1]

    def load_images(self, path_to_images, val_size, toshuffle=True, random_state=None, verbose=0):
        """"
        Loads and preprocesses images.
        
        Parameters:
            path_to_images: full path to directory containing images
            val_size: float between 0 and 1 specifying proportion of data that will used for validation
            toshuffle: boolean. if true, shuffles input images
            random_state: controls the shuffling applied to the data before applying the split. pass an int for
                        reproducible output. Default is None
            verbose: specifies verbosity mode

        Returns:
            x_training, x_valitation, y_training, y_valitation
        """

        if verbose > 0:
            print('Times of onset of selection for selection scenarios: {}'.format(self.ss))

        files = [f for f in os.scandir(path_to_images) if f.is_file() if f.name.endswith(".bmp")]

        im_ne, im_is, im_od, im_fd = [], [], [], []

        for file in files:
            if file.name.startswith("NE"):
                im_ne.append(file.path)
            elif file.name.startswith("IS"):
                im_is.append(file.path)
            elif file.name.startswith("OD"):
                im_od.append(file.path)
            elif file.name.startswith("FD"):
                im_fd.append(file.path)
            else:
                raise ValueError("{} file belongs to an unknown class")

        if self.test == 1:
            sseach = int(self.N / (7 * 3))

            m_ne = im_ne[0:sseach * 3 * 7]
            m_is = [f for f in im_is
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            listing = m_ne + m_is + m_od + m_fd
            if verbose > 0:
                print('Total sample sizes:\nNeutral: {}\n'
                      'Selection: {} = {}(IS) + {}(FD) + {}(OD) '.format(len(m_ne),
                                                                         (len(m_is) + len(m_od) + len(m_fd)),
                                                                         len(m_is), len(m_od), len(m_fd)))

        elif self.test == 2:
            sseach = int(self.N / (7 * 2))

            m_is = [f for f in im_is
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach * 2 + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            listing = m_is + m_od + m_fd
            if verbose > 0:
                print('Total sample sizes:\nIncomplete sweep: {}\n'
                      'Balancing selection: {} = {}(FD) + {}(OD) '.format(len(m_is),
                                                                          (len(m_od) + len(m_fd)),
                                                                          len(m_od), len(m_fd)))

        elif self.test == 3:
            sseach = int(self.N / 7)

            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, sseach + 1)
                    if int(f.replace(".bmp", "").split("_")[-2]) in self.ss]
            listing = m_od + m_fd
            if verbose > 0:
                print('Total sample sizes:\nOverdominance: {}\nNeg. freq-dependent selection: {}'.format(len(m_od),
                                                                                                         len(m_fd)))

        if verbose > 0:
            counter_list = [l.split("/")[-1].split("_")[0] + "_" + l.split("/")[-1].split("_")[1] for l in listing
                            if not l.split("/")[-1].startswith("NE")]
            counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))

            print("\nSample sizes for each selection start time:")
            for s in sorted(counter_dict):
                print(s.split("_")[0], 'for', s.split("_")[1], 'k years old selection: ', counter_dict[s])

        im_matrix_rows = len(listing)
        im_matrix_cols = self.image_row * self.image_col
        im_matrix = np.empty((im_matrix_rows, im_matrix_cols), dtype='float32')

        for i, im in enumerate(listing):
            image = np.asarray(Image.open(im)).flatten()
            image = image.astype('float32')
            im_matrix[i, :] = image

        labels = np.zeros((len(listing),), dtype=int)
        labels[(len(listing) // 2):] = 1

        X_train, X_val, y_train, y_val = train_test_split(im_matrix, labels, test_size=val_size,
                                                          shuffle=toshuffle, random_state=random_state)

        # reshaping, normalizing, convert to categorical data, save
        CHANNELS = 1
        X_train = X_train.reshape(X_train.shape[0], self.image_row, self.image_col, CHANNELS)
        X_val = X_val.reshape(X_val.shape[0], self.image_row, self.image_col, CHANNELS)

        X_train /= 255
        X_val /= 255

        return X_train, X_val, y_train, y_val


class SumStats(BaSe):
    """
    Load and preprocess summary statistics that will be used to train NeuralNet
    """
    
    def __init__(self, test, selection_category, N):
        """
        Parameters:
            test: test number to be performed, either:
                1: neutrality vs selection
                2: incomplete sweep vs balancing selection
                3: overdominance vs negative frequency-dependent selection
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                recent: includes selection scenarios ranging from 20k to 26k years old.
                medium: includes selection scenarios ranging from 27k to 33k years old.
                old: includes selection scenarios ranging from 34k to 40k years old.
            N: sample size for each class.
            image size: dimensions of input images in the format of (nrow, ncol)
        """
        super().__init__(test, selection_category, N)

    def load_sumstats(self, path_to_stats, val_size, toshuffle=True, scale=True, pca=True, random_state=None, verbose=0):
        """
        Loads and preprocess summary statistics. Preprocessing includes dealing with missing values, shuffling,
        spliting them into training and valitation set, feature scaling and pca.
        
        Note: csv files containing summary statistics must be named as: NE.csv, IS.csv, OD.csv, and FD.csv
        
        Parameters:
            path_to_stats: full path to directory containing summary statistics
            val_size: float between 0 and 1 specifying proportion of data that will used for validation
            toshuffle: boolean. if true, shuffles input images
            scale: True (default) to perform feature scaling
            pca: True (default) to perform pca
            random_state: controls the shuffling applied to the data before applying the split. pass an int for
                            reproducible output. Default is None
            verbose: specifies verbosity mode
            
        Returns:
            x_training, x_valitation, y_training, y_valitation
        """
        f1 = pd.read_csv(path_to_stats + "FD.csv")
        f2 = pd.read_csv(path_to_stats + "OD.csv")
        f3 = pd.read_csv(path_to_stats + "IS.csv")
        f4 = pd.read_csv(path_to_stats + "NE.csv")

        if verbose > 0:
            print('Times of onset of selection for selection scenarios: {}'.format(self.ss))

        f1 = f1.iloc[[i for i, j in enumerate(f1.iloc[:, 1]) if j in self.ss], :]
        f2 = f2.iloc[[i for i, j in enumerate(f2.iloc[:, 1]) if j in self.ss], :]
        f3 = f3.iloc[[i for i, j in enumerate(f3.iloc[:, 1]) if j in self.ss], :]

        if self.test == 1:
            sseach = int(self.N / (3 * 7))
            f4 = f4.iloc[0:sseach * 3 * 7, :]

            f3 = f3.iloc[[i for i, j in enumerate(f3.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            f2 = f2.iloc[[i for i, j in enumerate(f2.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            f1 = f1.iloc[[i for i, j in enumerate(f1.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            data = f3.append(f1.append(f2))

            if verbose > 0:
                counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])
                                if not data.iloc[i, 0].startswith("NE")]
                counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
                print('Total sample sizes:\nNeutral: {}\n'
                      'Selection: {} = {}(IS) + {}(FD) + {}(OD) '.format(f4.shape[0],
                                                                         (f3.shape[0] + f2.shape[0] + f1.shape[0]),
                                                                         f3.shape[0], f2.shape[0], f1.shape[0]))
            data.iloc[:, 0:1] = 'S'
            data = f4.append(data)

        elif self.test == 2:
            sseach = int(self.N / (2 * 7))

            f3 = f3.iloc[[i for i, j in enumerate(f3.iloc[:, 2]) if j in range(1, sseach * 2 + 1)], :]
            f2 = f2.iloc[[i for i, j in enumerate(f2.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            f1 = f1.iloc[[i for i, j in enumerate(f1.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            data = f1.append(f2)

            if verbose > 0:
                counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])]
                counter_list2 = [f3.iloc[i, 0] + "_" + str(f3.iloc[i, 1]) for i in range(0, f3.shape[0])]
                counter_list = counter_list + counter_list2
                counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
                print('Total sample sizes:\nIncomplete sweep: {}\n'
                      'Balancing selection: {} = {}(FD) + {}(OD) '.format(f3.shape[0],
                                                                          (f2.shape[0] + f1.shape[0]),
                                                                          f2.shape[0], f1.shape[0]))
            data.iloc[:, 0:1] = 'BS'
            data = f3.append(data)

        elif self.test == 3:
            sseach = int(self.N / 7)
            f2 = f2.iloc[[i for i, j in enumerate(f2.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            f1 = f1.iloc[[i for i, j in enumerate(f1.iloc[:, 2]) if j in range(1, sseach + 1)], :]
            data = f1.append(f2)

            if verbose > 0:
                counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])]
                counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
                print('Total sample sizes:\nOverdominance: {}\nNeg. freq-dependent selection: {}'.format(f2.shape[0],
                                                                                                         f1.shape[0]))

        if verbose > 0:
            print("\nSample sizes for each selection start time:")
            for s in sorted(counter_dict):
                print(s.split("_")[0], 'for', s.split("_")[1], 'k years old selection: ', counter_dict[s])

        stat_matrix = data.iloc[:,3:].values
        y = data.iloc[:,0].values
        
        labels = np.zeros((len(y),), dtype=int)
        labels[(len(y) // 2):] = 1

        # Missing data
        pos=np.argwhere(np.isnan(stat_matrix))
        if len(pos) != 0:    
            pos1=pos[:,1]
            for i in range(pos1.shape[0]):
                imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
                ind = pos1[i]
                imputer.fit(stat_matrix[:,ind,None])
                stat_matrix[:,ind,None] = imputer.transform(stat_matrix[:,ind,None])

        # Split the dataset into the Training set and Test set
        X_train, X_val, y_train, y_val = train_test_split(stat_matrix, labels, test_size=val_size,
                                                          shuffle=toshuffle, random_state=random_state)

        if scale:
            sc = StandardScaler()
            sc.fit(X_train)	
            X_train = sc.transform(X_train)
            X_val = sc.transform(X_val)
        
        if pca:
            pca = PCA() 
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_val_pca = pca.transform(X_val)
            
            X_train = X_train_pca
            X_val = X_val_pca
            
        return X_train, X_val, y_train, y_val

class VCF(object):

    def __init__(self, file_name):
        """Loads VCF file

        Parameters:
            file_name: target vcf file
        """
        with open(file_name, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]

        header = lines.pop(0)
        ind_pos = header.split('\t').index('POS')
        ind_snp = header.split('\t').index('ID')
        ind_format = header.split('\t').index('FORMAT')

        nr_individuals = len(header.split('\t')) - ind_format - 1
        nr_sites = len(lines)

        haplotypes = np.zeros(((nr_individuals * 2), nr_sites, 1), dtype='uint8')
        positions = []
        snp_id = []

        for j in range(nr_sites):
            # extract genotypes
            genotypes = lines[j].split('\t')[(ind_format + 1):]
            genotypes[len(genotypes) - 1] = genotypes[len(genotypes) - 1].split('\n')[0]

            for i in range(len(genotypes)):
                if i == 0:
                    i1 = 0
                    i2 = 1
                else:
                    i2 = i * 2
                    i1 = i2 - 1
                if genotypes[i].split('|')[0] == '1':
                    haplotypes[i1, j] = '1'
                if genotypes[i].split('|')[1] == '1':
                    haplotypes[i2, j] = '1'

            positions.append(int(lines[j].split('\t')[ind_pos]))
            snp_id.append(lines[j].split('\t')[ind_snp])
        croms = haplotypes.reshape(haplotypes.shape[0], haplotypes.shape[1])

        self.croms, self.positions, self.snp_id = croms, positions, snp_id

    def scan_targets(self, N, target_range, target_freq):
        """Scans vcf file for candidate targets

        Parameters:
            N: length of the (simulated) sequence
            target_range: A tuple specifying the target range of positions. If None, scans all the positions.
            target_freq: A tuple specifying the frequency range for targets.
        Returns:
            A list of candidate targets
        """
        if not target_range:
            target_range = [int(min(self.positions) + N // 2), int(max(self.positions) - N // 2)]

        freqs = np.true_divide(np.sum(self.croms, axis=0), self.croms.shape[0])
        targets = np.logical_and(np.logical_and(freqs > target_freq[0], freqs < target_freq[1]),
                                 np.logical_and(np.array(self.positions) > target_range[0],
                                                np.array(self.positions) < target_range[1]))
        if len(targets) == 0:
            raise LookupError("No SNP found for the given target frequency range")

        target_list = np.array(self.snp_id)[targets]

        return target_list

    def _crop_for_target(self, N, target_snp):
        """
        Given the target SNP, crops the sequence so that the target SNP is at the center and length of the sequence
        is equal to N.

        Parameters:
            N: length of the (simulated) sequence
            target_snp: target snp id
        Return:
            Haplotype matrix
            Positions of segregating sites
            SNP IDs
        """
        snp_idx = self.snp_id.index(target_snp)
        pos_mut = self.positions[snp_idx]
        up_bound = pos_mut + N / 2
        low_bound = pos_mut - N / 2
        if low_bound < 0 or up_bound > max(self.positions):
            raise IndexError("Flanking regions around the target SNP must be greater than {}".format(N // 2))
        target_range = np.logical_and(np.array(self.positions) > low_bound, np.array(self.positions) < up_bound)

        pos_new = np.array(self.positions)[target_range] + N / 2 - pos_mut
        croms_new = self.croms[:, target_range]
        new_snp_id = np.array(self.snp_id)[target_range]
        # npos_mut = pos_new[np.where(new_snp_id == target_snp)[0][0]]

        return croms_new, pos_new, new_snp_id

    def filter_data(self, filter_freq=0.1):
        """Filters positions whose frequency is less than filter_freq"""

        freqs = np.true_divide(np.sum(self.croms, axis=0), self.croms.shape[0])

        self.croms = self.croms[:, freqs > filter_freq]
        self.positions = np.array(self.positions)[freqs > filter_freq].tolist()
        self.snp_id = np.array(self.snp_id)[freqs > filter_freq].tolist()

    def create_image(self, N, sort, method, target_freq=(0.4, 0.6), target_list=None,
                     target_range=None, img_dim=(128, 128)):
        """
        Scans the VCF file and create and preprocess images

        Parameters:
            N: length of the (simulated) sequence
            sort: sorting algorithm (gen_sim or freq)
            method: sorting method (t or s)
            target_freq: A tuple specifying the frequency range for targets
            target_list: A list of target SNPs. If None, scans the target region for all candidate targets
            target_range:  A tuple specifying the target range of positions. If None, scans all the positions
            img_dim: Image dimension (nrow, ncol)

        Returns:
            A matrix containing images
            A list containing corresponding SNP IDs
        """

        if not target_list:
            target_list = self.scan_targets(N=N, target_range=target_range, target_freq=target_freq)
            print("{} candidate targets have been found.".format(len(target_list)))

        if not all(t in self.snp_id for t in target_list):
            raise ValueError("Target SNP not found in VCF file")

        nrow = len(target_list)
        ncol = img_dim[0] * img_dim[1]
        im_matrix = np.empty((nrow, ncol), dtype='float32')

        target_pos = []
        for i, target_snp in enumerate(target_list):
            croms, pos, snp_id = self._crop_for_target(N, target_snp)
            npos = int(np.where(snp_id == target_snp)[0][0])
            ncrom = order_data(croms, npos, sort, method)
            im_resized = matrix_to_image(ncrom, img_dim[0], img_dim[1])

            im = np.asarray(im_resized).flatten()
            im = im.astype('float32')
            im_matrix[i, :] = im
            target_pos.append(self.positions[self.snp_id.index(target_snp)])

        CHANNELS = 1
        im_matrix = im_matrix.reshape(im_matrix.shape[0], img_dim[0], img_dim[1], CHANNELS)
        im_matrix /= 255

        return im_matrix, target_list, target_pos

    def create_stat(self, N, target_freq=(0.4, 0.6), target_list=None, target_range=None, scale=False, pca=False):
        """
        Scans the VCF file and create and preprocess summary statistics

        Parameters:
            N: length of the (simulated) sequence
            target_freq: A tuple specifying the frequency range for targets
            target_list: A list of target SNPs. If None, scans the target region for all candidate targets
            target_range:  A tuple specifying the target range of positions. If None, scans all the positions
            scale: If True, performs feature scaling
            pca: If True, performs pca

        Returns:
            A matrix containing summary statistics
            A list containing corresponding SNP IDs
        """
        if not target_list:
            target_list = self.scan_targets(N=N, target_range=target_range, target_freq=target_freq)
            print("{} candidate targets have been found.".format(len(target_list)))

        if not all(t in self.snp_id for t in target_list):
            raise ValueError("Target SNP not found in VCF file")

        statdf = pd.DataFrame()
        target_pos = []
        for target_snp in target_list:
            croms, pos, _ = self._crop_for_target(N, target_snp)
            labs, stats = sum_stats(croms, pos, croms.shape[0], sname="0_0_0")

            statdf = statdf.append(dict(zip(labs, stats)), ignore_index=True)
            statdf = statdf[labs]
            target_pos.append(self.positions[self.snp_id.index(target_snp)])

        stat_matrix = statdf.iloc[:, 3:].values

        if scale:
            sc = StandardScaler()
            sc.fit(stat_matrix)
            stat_matrix = sc.transform(stat_matrix)

        if pca:
            pca = PCA()
            pca.fit(stat_matrix)
            stat_matrix_pca = pca.transform(stat_matrix)
            stat_matrix = stat_matrix_pca

        return stat_matrix, target_list, target_pos
