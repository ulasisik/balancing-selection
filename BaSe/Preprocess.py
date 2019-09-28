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

from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def read_msms(filename, NCHROMS, N):
    '''
    Reads msms file to an haplotype matrix
    Arguments:
        filename (string) -- full path and name of the .txt MSMS file
        NCHROMS (int) -- number of samples(haploid individuals, or chromosoms
        N (int) -- length of the simulated sequence(bp)
    Output:
        Returns an haplotype array, and an array containing positions
    '''
    file = open(filename).readlines()
    if len(file) == 0:
        raise Exception('The file ' + filename.split('/')[-1] + ' is empty')
    # look for the // character in the file
    find = []
    for i, string in enumerate(file):
        if string == '//\n':
            find.append(i + 3)
    for ITER, pointer in enumerate(find):
        # Get positions
        pos = file[pointer - 1].split()
        del pos[0]
        pos = np.array(pos, dtype='float')
        pos = pos * N
        # Get the number of genomic positions(determined be the number or pointers)
        n_columns = len(list(file[pointer])) - 1
        # Intialise the empty croms matrix: of type: object
        croms = np.empty((NCHROMS, n_columns), dtype=np.object)
        # Fill the matrix with the simuated data
        for j in range(NCHROMS):
            f = list(file[pointer + j])
            del f[-1]
            F = np.array(f)
            croms[j, :] = F
    croms = croms.astype(int)
    return croms, pos


def rearrange_neutral(croms, pos, NCHROMS, length):
    """
    rearranges neutral simulations such that each simulation results in
    <length>bp in length with selected mutation at the center
    arguments:
        croms -- input haplotype matrix
        pos -- array containing position information for croms
        NCHROMS (int) -- number of samples(haploid individuals, or chromosoms
        length (int) -- desired length of the output(bp)
    output:
        return a haplotype matrix that is <length_out> bp in length and target snp at center,
        and an array containing new positions
    """
    freqs = np.true_divide(np.sum(croms, axis=0), NCHROMS)
    poss = pos[np.logical_and(freqs > 0.4, freqs < 0.6)]  # positions of mutations within [0.4,0.6]
    pos_mut = poss[len(poss) // 2]  # position of target mutation
    up_bound = pos_mut + length / 2  # upper and lower boundaries of the region that will be selected
    low_bound = pos_mut - length / 2
    target_range = np.logical_and(pos > low_bound, pos < up_bound)
    pos_new = pos[target_range] + length / 2 - pos_mut
    croms_new = croms[:, target_range]
    return croms_new, pos_new


def order_data(im_matrix):
    """
    Sorts matrix containing sequence data
    Keyword Arguments:
        im_matrix (Array [2D]) -- Array containing sequence data
    Returns:
        b (Array [2D]) -- Sorted array containing sequence data
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


def sort_min_diff(amat):
    '''
    This function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array
    Implemented from https://github.com/flag0010/pop_gen_cnn/blob/master/sort.min.diff.py#L1
    '''
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]


def sim_to_matrix(filename, NCHROMS, N, N_NE, sort, method):
    """
    Generates ordered haplotype_matrix from simulation results(must be ms format in .txt)
    Arguments:
        filename (str) -- full path and name of the .txt MSMS file
        NCHROMS (int) -- number of samples(haploid individuals, or chromosoms)
        N (int) -- length of the simulated sequence(bp)
        sort (str) -- sorting method. either
            gen_sim: based on genetic similarity
            freq: based on frequency
        method (str) -- sorting method
            t: together. sort the whole array together
            s: seperate. sort two haplotype groups seperately
    Returns ordered haplotype matrix, where columns are positions and rows are individuals
    """
    if 'NE' in filename.split("/")[-1]:
        crom, pos = read_msms(filename, NCHROMS, N_NE)
        croms, positions = rearrange_neutral(crom, pos, NCHROMS, N)
    else:
        croms, positions = read_msms(filename, NCHROMS, N)
    
    if method == "s":
        pos = np.where(np.abs(positions - N/2) < 1)[0]
        if len(pos) == 0:
            print(filename)
            print(positions)
            raise IndexError("Target SNP not found")
        elif len(pos) > 1:
            print("Target SNP found at multiple positions:")
            print(positions[pos])
            pos = np.array([pos[0]], dtype='int64')
            print(filename)
        elif len(pos) == 1:
            index_1 = (croms[:, pos] == 1).reshape(NCHROMS, )
            index_0 = (croms[:, pos] == 0).reshape(NCHROMS, )
            croms_1 = croms[index_1, :]
            croms_0 = croms[index_0, :]
            if sort == "gen_sim":
                croms1 = sort_min_diff(croms_1)
                croms0 = sort_min_diff(croms_0)
            elif sort == "freq":
                croms1 = order_data(croms_1)
                croms0 = order_data(croms_0)
            else:
                raise ValueError("sort must be either 'freq' or 'gen_sim'")
            croms = np.concatenate((croms0, croms1), axis=0)
    elif method == "t":
        if sort == "gen_sim":
            croms = sort_min_diff(croms)
        elif sort == "freq":
            croms = order_data(croms)
        else:
            raise ValueError("sort must be either 'freq' or 'gen_sim'")
    
    return croms


def matrix_to_image(croms, n_row, n_col, flip='none'):
    """
    Generates image from sorted haplotype matrix
    Resize image -> (n_row, ncol)
    Arguments:
        -croms (Array [2D]) -- Array containing sequence data
        -flip (none, lr, ud, full) -- creates a flipped copy of the orginal image:
            -none:
            -lr: left to right
            -ud: upside down
            -lrud: lr+ud
            -full: lr, ud, lr+ud
    Returns:
        -image file size of (n_row, n_col)
    """
    # Generate image
    all1 = np.ones(croms.shape)
    cromx = all1 - croms
    bw_croms_uint8 = np.uint8(cromx)
    bw_croms_im = Image.fromarray(bw_croms_uint8 * 255, mode='L')
    im_resized = bw_croms_im.resize((n_col, n_row))

    bw_lr = np.fliplr(bw_croms_uint8)
    bw_im_lr = Image.fromarray(bw_lr * 255, mode='L')
    im_lr = bw_im_lr.resize((n_col, n_row))

    bw_ud = np.flipud(bw_croms_uint8)
    bw_im_ud = Image.fromarray(bw_ud * 255, mode='L')
    im_ud = bw_im_ud.resize((n_col, n_row))

    bw_lrud = np.flipud(bw_lr)
    bw_im_lrud = Image.fromarray(bw_lrud * 255, mode='L')
    im_lrud = bw_im_lrud.resize((n_col, n_row))

    if flip == 'none':
        return im_resized
    elif flip == 'lr':
        return im_resized, im_lr
    elif flip == 'ud':
        return im_resized, im_ud
    elif flip == 'lrud':
        return im_resized, im_lrud
    elif flip == 'full':
        return im_resized, im_lr, im_ud, im_lrud


def sim_to_image(path_sim, path_image, REP_FROM, REP_TO, NCHROMS, N, N_NE, img_dim=(128,128), clss=("NE", "IS", "OD", "FD"), sort="freq", method="s"):
    """
    Converts msms simulation output files into images
    Args:
        path_sim: path to the folder containing simulation files
        path_image: path to a folder in which output images will be saved
        REP_FROM: number of simulations (replicate), starting from
        REP_TO: number of simulations, until
        NCHROMS: number of samples(haploid individuals, or chromosomes)
        N: length of the simulated sequence(bp) for selection scenarios
        N_NE: length of simulated sequence(bp) for neutral scenario
        img_dim: image dimensions as (nrow, ncol)
        clss: a tuple of target classes-
            "NE": neutral
            "IS": incomplete sweep
            "OD": overdominance
            "FD": negative freq-dependent selection
        sort: sorting algorithm. either
            gen_sim: based on genetic similarity
            freq: based on frequency
        method: sorting method. either
            t: together. sort the whole array together
            s: seperate. sort two haplotype groups seperately

    """

    print(sys.version_info)
    if sys.version_info >= (3, 6):
        with os.scandir(path_sim) as fdir:
            for file in fdir:
                if file.name.startswith(tuple(clss)) and file.is_file() and int(file.name.replace(".txt", "").split("_")[-1]) in range(REP_FROM, REP_TO+1):
                    croms = sim_to_matrix(file.path, NCHROMS, N, N_NE, sort=sort, method=method)
                    im_resized = matrix_to_image(croms, n_row=img_dim[0], n_col=img_dim[1], flip='none')
                    im_resized.save("{}{}.bmp".format(path_image,file.name.replace(".txt","")))
    else:
        files = [file for file in os.scandir(path_sim)
                 if file.is_file()
                 if file.name.startswith(tuple(clss))
                 if int(file.name.replace(".txt", "").split("_")[-1]) in range(REP_FROM, REP_TO + 1)]
        print("Sample size: {}".format(len(files)))

        for file in files:
            croms = sim_to_matrix(file.path, NCHROMS, N, N_NE, sort=sort, method=method)
            im_resized = matrix_to_image(croms, n_row=img_dim[0], n_col=img_dim[1], flip='none')
            im_resized.save("{}{}.bmp".format(path_image, file.name.replace(".txt", "")))


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


def sum_stat(path_to_sim, path_to_stat, cls, NCHROMS, REP_FROM, REP_TO, N, N_NE):
    '''
    Calculates summary statistics for simulation data:
        -input file must be in a msms(and .txt) format and
    Arguments:
        path_to_sim (string) -- Path to directory where the simulation files exist
        path_to_stat (string) -- Path to directory where the summary statistics will be stored
        cls (string) -- Class of the simulation(either FD, OD, IS or NE):
            -FD: negative-frequency dependent selection
            -OD: over dominance
            -IS: incomplete sweep
            -NE: neutral
        NCHROMS (int) -- number of samples(haploid individuals, or chromosoms)
        REP_FROM: number of simulations (replicate) -starting from
        REP_TO: number of simulations -until
        N: length of the simulated sequence(bp) for selection scenarios
        N_NE: length of simulated sequence(bp) for neutral scenario
    Output:
        csv file containing summary statistics
    '''
    if REP_FROM == 1:
        once = 0
    else:
        once = 1

    all_files = os.scandir(path_to_sim)
    files = [file for file in all_files
             if file.is_file()
             if file.name.startswith(cls)
             if int(file.name.replace(".txt", "").split("_")[-1]) in range(REP_FROM, REP_TO + 1)]

    if cls == "NE":
        counter_ne = [f for f in files if f.name.startswith("NE")]
        print("Sample size of neutral scenario: {}".format(len(counter_ne)))
    elif cls in ("IS", "OD", "FD"):
        counter_list = [f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in files
                        if not f.name.startswith("NE")]
        counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))

        print("\nNumber of samples for each selection start time:")
        for s in sorted(counter_dict):
            print(s.split("_")[0], 'for', s.split("_")[1], 'k years old selection: ', counter_dict[s])
    else:
        raise ValueError("'cls' must be either NE, IS, OD, or FD")

    files = [f.path for f in files]
    for file in sorted(files):
        fname = file.split("/")[-1].replace(".txt", "")
        time = fname.split('_')[1]
        r = fname.split('_')[-1]

        if fname.startswith("NE"):
            crom, positions = read_msms(file, NCHROMS, N_NE)
            croms, pos = rearrange_neutral(crom, positions, NCHROMS, N)
        else:
            croms, pos = read_msms(file, NCHROMS, N)

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
        mean_obs_exp1 = np.nanmean(obs_het1 / exp_het1)
        median_obs_exp1 = np.nanmedian(obs_het1 / exp_het1)
        max_obs_exp1 = np.nanmax(obs_het1 / exp_het1)

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
        mean_obs_exp2 = np.nanmean(obs_het2 / exp_het2)
        median_obs_exp2 = np.nanmedian(obs_het2 / exp_het2)
        max_obs_exp2 = np.nanmax(obs_het2 / exp_het2)

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


        # Write on csv file
        f = open("{}{}.csv".format(path_to_stat, cls), 'a+')

        with f:
            header = ['Class', 'Time', 'Iteration',
                      'Mean(MeanPwiseDist)1',
                      'Median(MeanPwiseDist)1',
                      'Max(MeanPwiseDist)1',
                      'Tajimas D1', 'Watterson1',
                      'Mean(ObservedHet)1', 'Median(ObservedHet)1',
                      'Max(ObservedHet)1', 'Mean(Obs/Exp Het)1',
                      'Median(Obs/Exp Het)1', 'Max(Obs/Exp Het)1',
                      'Median(r2)1',
                      'H1_1', 'H12_1', 'H123_1', 'H2/H1_1', 'Haplotype Diversity1',
                      '# of Hap1', 'Mean(EHH)1', 'Median(EHH)1',
                      'Median(ihs)1', 'Max(nsl)1', 'Median(nsl)1',
                      'NCD1_1', 'NCD2_1', 'KellyZns1', 'pi1', 'faywuH1',
                      '#ofSingletons1', 'Dstar1', 'Fstar1', 'ZengE1', 'Rageddnes1',
                      #
                      'Mean(MeanPwiseDist)2',
                      'Median(MeanPwiseDist)',
                      'Max(MeanPwiseDist)2',
                      'Tajimas D2', 'Watterson2',
                      'Mean(ObservedHet)2', 'Median(ObservedHet)2',
                      'Max(ObservedHet)2', 'Mean(Obs/Exp Het)2',
                      'Median(Obs/Exp Het)2', 'Max(Obs/Exp Het)2',
                      'Median(r2)',
                      'H1_2', 'H12_2', 'H123_2', 'H2/H1_2', 'Haplotype Diversity_2',
                      '# of Hap2', 'Mean(EHH)2', 'Median(EHH)2',
                      'Median(ihs)2', 'Max(nsl)2', 'Median(nsl)2',
                      'NCD1_2', 'NCD2_2', 'KellyZns2', 'pi2', 'faywuH2',
                      '#ofSingletons2', 'Dstar2', 'Fstar2', 'ZengE2', 'Rageddnes2']
            writer = csv.DictWriter(f, fieldnames=header)

            if once == 0:
                writer.writeheader()
                writer.writerow({'Class': str(cls), 'Iteration': str(r),
                                 'Time': str(time),
                                 'Mean(MeanPwiseDist)1': mean_mean_pwise_dis1,
                                 'Median(MeanPwiseDist)1': median_mean_pwise_dis1,
                                 'Max(MeanPwiseDist)1': max_mean_pwise_dis1,
                                 'Tajimas D1': TjD1, 'Watterson1': theta_hat_w1,
                                 'Mean(ObservedHet)1': mean_obs_het1, 'Median(ObservedHet)1': median_obs_het1,
                                 'Max(ObservedHet)1': max_obs_het1, 'Mean(Obs/Exp Het)1': mean_obs_exp1,
                                 'Median(Obs/Exp Het)1': median_obs_exp1, 'Max(Obs/Exp Het)1': max_obs_exp1,
                                 'Median(r2)1': median_r21,
                                 'H1_1': h11, 'H12_1': h121, 'H123_1': h1231, 'H2/H1_1': h2_h11,
                                 'Haplotype Diversity1': hap_div1,
                                 '# of Hap1': n_hap1, 'Mean(EHH)1': mean_ehh1, 'Median(EHH)1': median_ehh1,
                                 'Median(ihs)1': median_ihs1, 'Max(nsl)1': max_nsl1, 'Median(nsl)1': median_nsl1,
                                 'NCD1_1': ncd11, 'NCD2_1': ncd21, 'KellyZns1': kellyzn1, 'pi1': pi_est1,
                                 'faywuH1': Hstat1,
                                 '#ofSingletons1': Ss1, 'Dstar1': Dstar1, 'Fstar1': Fstar1, 'ZengE1': ZengE1,
                                 'Rageddnes1': rgd1,
                                 #
                                 'Mean(MeanPwiseDist)2': mean_mean_pwise_dis2,
                                 'Median(MeanPwiseDist)': median_mean_pwise_dis2,
                                 'Max(MeanPwiseDist)2': max_mean_pwise_dis2,
                                 'Tajimas D2': TjD2, 'Watterson2': theta_hat_w2,
                                 'Mean(ObservedHet)2': mean_obs_het2, 'Median(ObservedHet)2': median_obs_het2,
                                 'Max(ObservedHet)2': max_obs_het2, 'Mean(Obs/Exp Het)2': mean_obs_exp2,
                                 'Median(Obs/Exp Het)2': median_obs_exp2, 'Max(Obs/Exp Het)2': max_obs_exp2,
                                 'Median(r2)': median_r2,
                                 'H1_2': h12, 'H12_2': h122, 'H123_2': h1232, 'H2/H1_2': h2_h12,
                                 'Haplotype Diversity_2': hap_div2,
                                 '# of Hap2': n_hap2, 'Mean(EHH)2': mean_ehh2, 'Median(EHH)2': median_ehh2,
                                 'Median(ihs)2': median_ihs2, 'Max(nsl)2': max_nsl2, 'Median(nsl)2': median_nsl2,
                                 'NCD1_2': ncd12, 'NCD2_2': ncd22, 'KellyZns2': kellyzn2, 'pi2': pi_est2,
                                 'faywuH2': Hstat2,
                                 '#ofSingletons2': Ss2, 'Dstar2': Dstar2, 'Fstar2': Fstar2, 'ZengE2': ZengE2,
                                 'Rageddnes2': rgd2})
                once = 1
            else:
                writer.writerow({'Class': str(cls), 'Iteration': str(r),
                                 'Time': str(time),
                                 'Mean(MeanPwiseDist)1': mean_mean_pwise_dis1,
                                 'Median(MeanPwiseDist)1': median_mean_pwise_dis1,
                                 'Max(MeanPwiseDist)1': max_mean_pwise_dis1,
                                 'Tajimas D1': TjD1, 'Watterson1': theta_hat_w1,
                                 'Mean(ObservedHet)1': mean_obs_het1, 'Median(ObservedHet)1': median_obs_het1,
                                 'Max(ObservedHet)1': max_obs_het1, 'Mean(Obs/Exp Het)1': mean_obs_exp1,
                                 'Median(Obs/Exp Het)1': median_obs_exp1, 'Max(Obs/Exp Het)1': max_obs_exp1,
                                 'Median(r2)1': median_r21,
                                 'H1_1': h11, 'H12_1': h121, 'H123_1': h1231, 'H2/H1_1': h2_h11,
                                 'Haplotype Diversity1': hap_div1,
                                 '# of Hap1': n_hap1, 'Mean(EHH)1': mean_ehh1, 'Median(EHH)1': median_ehh1,
                                 'Median(ihs)1': median_ihs1, 'Max(nsl)1': max_nsl1, 'Median(nsl)1': median_nsl1,
                                 'NCD1_1': ncd11, 'NCD2_1': ncd21, 'KellyZns1': kellyzn1, 'pi1': pi_est1,
                                 'faywuH1': Hstat1,
                                 '#ofSingletons1': Ss1, 'Dstar1': Dstar1, 'Fstar1': Fstar1, 'ZengE1': ZengE1,
                                 'Rageddnes1': rgd1,
                                 #
                                 'Mean(MeanPwiseDist)2': mean_mean_pwise_dis2,
                                 'Median(MeanPwiseDist)': median_mean_pwise_dis2,
                                 'Max(MeanPwiseDist)2': max_mean_pwise_dis2,
                                 'Tajimas D2': TjD2, 'Watterson2': theta_hat_w2,
                                 'Mean(ObservedHet)2': mean_obs_het2, 'Median(ObservedHet)2': median_obs_het2,
                                 'Max(ObservedHet)2': max_obs_het2, 'Mean(Obs/Exp Het)2': mean_obs_exp2,
                                 'Median(Obs/Exp Het)2': median_obs_exp2, 'Max(Obs/Exp Het)2': max_obs_exp2,
                                 'Median(r2)': median_r2,
                                 'H1_2': h12, 'H12_2': h122, 'H123_2': h1232, 'H2/H1_2': h2_h12,
                                 'Haplotype Diversity_2': hap_div2,
                                 '# of Hap2': n_hap2, 'Mean(EHH)2': mean_ehh2, 'Median(EHH)2': median_ehh2,
                                 'Median(ihs)2': median_ihs2, 'Max(nsl)2': max_nsl2, 'Median(nsl)2': median_nsl2,
                                 'NCD1_2': ncd12, 'NCD2_2': ncd22, 'KellyZns2': kellyzn2, 'pi2': pi_est2,
                                 'faywuH2': Hstat2,
                                 '#ofSingletons2': Ss2, 'Dstar2': Dstar2, 'Fstar2': Fstar2, 'ZengE2': ZengE2,
                                 'Rageddnes2': rgd2})


class BaSe(object):
    '''
    Distinguishing between balancing selection and incomplete sweep.
    Includes three possible tests:
        test 1: neutrality vs selection
        test 2: incomplete sweep vs balancing selection
        test 3: overdominance vs negative frequency-dependent selection
    '''
    
    def __init__(self, test, selection_category, N):
        '''
        Args:
            test: test number to be performed, either 1, 2, or 3.
            N: sample size for each class.
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                recent: includes selection scenarios ranging from 20k to 26k years old.
                medium: includes selection scenarios ranging from 27k to 33k years old.
                old: includes selection scenarios ranging from 34k to 40k years old.
        '''
        
        if test not in [1, 2, 3]:
            raise ValueError("Test number must be 1, 2, or 3")
        else:
            self.test = test
            if test == 1:
                self.classes = ['NE', 'S']
                self.labels = ['NE', 'IS', 'FD', 'OD']
            elif test == 2:
                self.classes = ['IS', 'BS']
                self.labels = ['IS', 'FD', 'OD']
            elif test == 3:
                self.classes = ['OD', 'FD']
                self.labels = ['OD', 'FD']
                
        if selection_category not in ['recent', 'medium', 'old']:
            raise ValueError("'{}' is not defined.".format(selection_category))
        else:
            self.selection_category = selection_category
        self.N = N

            
class Images(BaSe):
    '''
    Load and preprocess images that will be used to train ConvNet based
    classifier.
    '''
    
    def __init__(self, test, selection_category, N, image_size):
        '''
        Args:
            test: test number to be performed, either 1, 2, or 3.
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                            recent: includes selection scenarios ranging from 20k to 26k years old.
                            medium: includes selection scenarios ranging from 27k to 33k years old.
                            old: includes selection scenarios ranging from 34k to 40k years old.
            N: number of simulations that will be included per class.
            image size (tuple) -- dimensions of input images in the format of (nrow, ncol)
                                
        '''
        super().__init__(test, selection_category, N)
        self.image_row = image_size[0]
        self.image_col = image_size[1]

    def load_images(self, path_to_images):
        """"
        Loads images.
        
        Args:
            path_to_images: full path to directory containing images
            
        Returns:
            x_training, x_valitation, y_training, y_valitation
 
        """
        if self.selection_category == "recent":
            ss = [i for i in range(20, 27)]
        elif self.selection_category == "medium":
            ss = [i for i in range(27, 34)]
        elif self.selection_category == "old":
            ss = [i for i in range(34, 41)]
        print('Times of onset of selection for selection scenarios: {}'.format(ss))

        files = [f for f in os.scandir(path_to_images) if f.is_file() if f.name.endswith(".bmp")]
        im_ne = [file.path for file in files
                 if file.name.startswith("NE")
                 if int(file.name.replace(".bmp", "").split("_")[-1]) in range(1, self.N + 1)]
        im_is = [file.path for file in files
                 if file.name.startswith("IS")
                 if int(file.name.replace(".bmp", "").split("_")[-1]) in range(1, self.N + 1)
                 if int(file.name.replace(".bmp", "").split("_")[1]) in ss]
        im_od = [file.path for file in files
                 if file.name.startswith("OD")
                 if int(file.name.replace(".bmp", "").split("_")[-1]) in range(1, self.N + 1)
                 if int(file.name.replace(".bmp", "").split("_")[1]) in ss]
        im_fd = [file.path for file in files
                 if file.name.startswith("FD")
                 if int(file.name.replace(".bmp", "").split("_")[-1]) in range(1, self.N + 1)
                 if int(file.name.replace(".bmp", "").split("_")[1]) in ss]

        if self.test == 1:
            m_ne = im_ne[0:self.N]
            m_is = [f for f in im_is
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int((self.N / 3) / len(ss)) + 1)]
            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int((self.N / 3) / len(ss)) + 1)]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int((self.N / 3) / len(ss)) + 1)]

            listing = m_ne + m_is + m_od + m_fd
            print('Total sample sizes:\nNeutral: {}\n'
                  'Selection: {} = {}(IS) + {}(FD) + {}(OD) '.format(len(m_ne),
                                                                     (len(m_is) + len(m_od) + len(m_fd)),
                                                                     len(m_is), len(m_od), len(m_fd)))

        elif self.test == 2:
            m_is = [f for f in im_is
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int(self.N / len(ss)) + 1)]
            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int((self.N / 2) / len(ss)) + 1)]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int((self.N / 2) / len(ss)) + 1)]

            listing = m_is + m_od + m_fd
            print('Total sample sizes:\nIncomplete sweep: {}\n'
                  'Balancing selection: {} = {}(FD) + {}(OD) '.format(len(m_is),
                                                                      (len(m_od) + len(m_fd)),
                                                                      len(m_od), len(m_fd)))

        elif self.test == 3:
            m_od = [f for f in im_od
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int(self.N / len(ss)) + 1)]
            m_fd = [f for f in im_fd
                    if int(f.replace(".bmp", "").split("_")[-1]) in range(1, int(self.N / len(ss)) + 1)]

            listing = m_od + m_fd
            print('Total sample sizes:\nOverdominance: {}\nNeg. freq-dependent selection: {}'.format(len(m_od),
                                                                                                     len(m_fd)))

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

        im_matrix, labels = shuffle(im_matrix, labels, random_state=2)
        X_train, X_val, y_train, y_val = train_test_split(im_matrix, labels, test_size=0.2, random_state=4)

        # reshaping, normalizing, convert to categorical data, save
        CHANNELS = 1
        X_train = X_train.reshape(X_train.shape[0], self.image_row, self.image_col, CHANNELS)
        X_val = X_val.reshape(X_val.shape[0], self.image_row, self.image_row, CHANNELS)

        X_train /= 255
        X_val /= 255

        return X_train, X_val, y_train, y_val


class SumStats(BaSe):
    '''
    Load and preprocess summary statistics that will be used to train NeuralNet
    '''
    
    def __init__(self, test, selection_category, N):
        '''
        Args:
            test: test number to be performed, either 1, 2, or 3.
            selection_category: specifies the time of onset of selection for selection scenarios. Possible bins:
                recent: includes selection scenarios ranging from 20k to 26k years old.
                medium: includes selection scenarios ranging from 27k to 33k years old.
                old: includes selection scenarios ranging from 34k to 40k years old.
            N: number of simulations that will be included per class.

        '''
        super().__init__(test, selection_category, N)

    def load_sumstats(self, path_to_stats, scale=True, pca=True):
        '''
        Loads and preprocess summary statistics. Preprocessing includes dealing 
        with missing values, shuffling, spliting them into training and valitation 
        set, feature scaling and pca.
        
        Note: csv files containing summary statistics must be in a form of
            NE.csv, IS.csv, OD.csv, and FD.csv
        
        Args:
            path_to_stats: full path to directory containing summary statistics
            scale: True (default) to perform feature scaling
            pca: True (default) to perform pca
            
        Returns:
            x_training, x_valitation, y_training, y_valitation
        '''
        f1 = pd.read_csv(path_to_stats + "FD.csv")
        f2 = pd.read_csv(path_to_stats + "OD.csv")
        f3 = pd.read_csv(path_to_stats + "IS.csv")
        f4 = pd.read_csv(path_to_stats + "NE.csv")

        if self.selection_category == "recent":
            ss = [i for i in range(20, 27)]
        elif self.selection_category == "medium":
            ss = [i for i in range(27, 34)]
        elif self.selection_category == "old":
            ss = [i for i in range(34, 41)]
        print('Times of onset of selection for selection scenarios: {}'.format(ss))

        f1 = f1.iloc[[i for i, j in enumerate(f1.iloc[:, 1]) if j in ss], :]
        f2 = f2.iloc[[i for i, j in enumerate(f2.iloc[:, 1]) if j in ss], :]
        f3 = f3.iloc[[i for i, j in enumerate(f3.iloc[:, 1]) if j in ss], :]

        if self.test == 1:
            f4 = f4.iloc[0:self.N, ]
            sseach = f4.shape[0] / 3
            f1 = f1.iloc[0:int(sseach), ]
            f2 = f2.iloc[0:int(sseach), ]
            f3 = f3.iloc[0:int(sseach), ]
            data = f3.append(f1.append(f2))
            counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])
                            if not data.iloc[i, 0].startswith("NE")]
            counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
            data.iloc[:, 0:1] = 'S'
            data = f4.append(data)
            print('Total sample sizes:\nNeutral: {}\n'
                  'Selection: {} = {}(IS) + {}(FD) + {}(OD) '.format(f4.shape[0],
                                                                     (f3.shape[0] + f2.shape[0] + f1.shape[0]),
                                                                     f3.shape[0], f2.shape[0], f1.shape[0]))

        elif self.test == 2:
            f3 = f3.iloc[0:self.N, ]
            sseach = f3.shape[0] / 2
            f1 = f1.iloc[0:int(sseach), ]
            f2 = f2.iloc[0:int(sseach), ]
            data = f1.append(f2)
            counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])
                            if not data.iloc[i, 0].startswith("NE")]
            counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
            data.iloc[:, 0:1] = 'BS'
            data = f3.append(data)
            print('Total sample sizes:\nIncomplete sweep: {}\n'
                  'Balancing selection: {} = {}(FD) + {}(OD) '.format(f3.shape[0],
                                                                      (f2.shape[0] + f1.shape[0]),
                                                                      f2.shape[0], f1.shape[0]))

        elif self.test == 3:
            f1 = f1.iloc[0:self.N, :]
            f2 = f2.iloc[0:self.N, :]
            data = f1.append(f2)
            counter_list = [data.iloc[i, 0] + "_" + str(data.iloc[i, 1]) for i in range(0, data.shape[0])
                            if not data.iloc[i, 0].startswith("NE")]
            counter_dict = dict((x, counter_list.count(x)) for x in set(counter_list))
            print('Total sample sizes:\nOverdominance: {}\nNeg. freq-dependent selection: {}'.format(f2.shape[0],
                                                                                                     f1.shape[0]))

        print("\nSample sizes for each selection start time:")
        for s in sorted(counter_dict):
            print(s.split("_")[0], 'for', s.split("_")[1], 'k years old selection: ', counter_dict[s])

        stat_matrix=data.iloc[:,3:].values
        y=data.iloc[:,0].values
        
        labels = np.zeros((len(y),), dtype=int)
        labels[(len(y) // 2):] = 1

        #Missing data
        pos=np.argwhere(np.isnan(stat_matrix))
        if len(pos) != 0:    
            pos1=pos[:,1]
            for i in range(pos1.shape[0]):
                imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
                ind = pos1[i]
                imputer.fit(stat_matrix[:,ind,None])
                stat_matrix[:,ind,None] = imputer.transform(stat_matrix[:,ind,None])

        #shuffle data
        stat_matrix, labels = shuffle(stat_matrix, labels, random_state=2)     
        #split the dataset into the Training set and Test set
        X_train, X_val, y_train, y_val = train_test_split(stat_matrix, labels, test_size = 0.2)
        
        if scale:
            sc = StandardScaler()
            sc.fit(X_train)	
            X_train = sc.transform(X_train)
            X_val = sc.transform(X_val)
        
        if pca:
            pca = PCA() 
            pca.fit(X_train)
            X_train_pca = pca.transform(X_train)
            X_val_pca=pca.transform(X_val)
            
            X_train = X_train_pca
            X_val = X_val_pca
            
        return X_train, X_val, y_train, y_val
