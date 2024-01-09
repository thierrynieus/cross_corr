"""Additional functions to test."""

import numpy as np
import pandas as pd
import pylab as plt
import matplotlib.cm as cm
import os

from tqdm import tqdm
from purk_func import load_purk_pos_channels, load_spike_sort
import general

debug_mode = 0
debug_nch = -1

# dt = 0.1

params = {'dt': 0.5, 'tw': 5, 'tmax': 6e5, 'corr': True, 'nboot': 100,
          'perc': [5, 50, 90, 95, 99]}


def collect_spikes_ch_su_spont(fn_bxr, fn_pos, tw_accept=None):
    """Collect spikes_ch_su of a BXR file of spontaneous activity.

    comment: works with concatenated BXR file (e.g. all trials in 1 file).

    fn_bxr          file name of the concatenated BXR file
    fn_pos          coordinates of electrodes to extract
                    (a BWCG file obtained in BrainWave)
                    .bwcg file -> pos['row'],pos['col'] range in (1,64)
    tw_accept:      acceptance time window (not yet implemented)
    """
    spk_ch_time, dummy, spk_ch_id, spike_units = load_spike_sort(fn_bxr)

    pos = load_purk_pos_channels(fn_pos)
    ch_id_all = np.array(pos['col'])-1 + 64*(np.array(pos['row'])-1)

    units_lst = []
    # keep units positions with at least on spike
    pos_unit = {'row': [], 'col': []}
    spikes_ch_su = {}
    count_no_units = 0
    for k, chid in enumerate(ch_id_all):
        spikes_ch_su[chid] = {}
        idx = np.where(spk_ch_id == chid)[0]
        spike_units_unique = np.unique(spike_units[idx])
        no_units = True
        for su in spike_units_unique:
            if su > 0:
                # negative values are bad clusters/units
                idx_su = np.where(spike_units[idx] == su)[0]
                spikes_ch_su[chid][su] = spk_ch_time[idx[idx_su]]
                units_lst.append(su)
                pos_unit['row'].append(pos['row'][k])
                pos_unit['col'].append(pos['col'][k])
                no_units = False
        if no_units:
            # count how many channels have no units
            count_no_units += 1
            # pos_unit['row'].append(pos['row'][k])
            # pos_unit['col'].append(pos['col'][k])
    print('%d channels have no units!' % count_no_units)
    spk_ch_time, spk_ch_id, spike_units = [], [], []  # clean space
    return spikes_ch_su, units_lst, pos_unit


def conv_to_spikes_su(spikes_ch_su):
    """Convert spikes_ch_su to spikes_su."""
    id_ch_lst = list(spikes_ch_su)
    spikes_su = {}
    for id_ch in id_ch_lst:
        for su in list(spikes_ch_su[id_ch]):
            spikes_su[su] = spikes_ch_su[id_ch][su]
    return spikes_su


def raster_plot_spikes_su(fn_spikes_su, plot_unit=False, bin_size=20):
    """Plot raster of PC spike trains from fn_spikes_su."""
    data = np.load(fn_spikes_su, allow_pickle=True).item()
    num_neurons = len(data)
    spk = []
    # plot
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    tmax = -1
    for k, su in enumerate(list(data)):
        tspk = data[su]
        ax1.plot(tspk, np.repeat(k, len(tspk)), 'ko')
        spk.extend(tspk)
        tmax = max(tmax, np.max(spk))
    tmax += bin_size
    bins = np.arange(0, tmax, bin_size)
    h = np.histogram(spk, bins=bins)[0]/(bin_size/1000.*num_neurons)
    ax2.plot(bins[:-1], h, 'r-', lw=2)
    ax1.set_xlabel('time (ms)', fontsize=14)
    ax1.set_ylabel('PC unit', fontsize=14, color='k')
    ax2.set_ylabel('istantaneous firing rate (Hz)', fontsize=14, color='r')
    if plot_unit:
        yticks = [str(su) for su in list(data)]
        ax1.set_yticks(np.arange(num_neurons))
        ax1.set_yticklabels(yticks, fontsize=2)
    plt.tight_layout(pad=1)


def cross_corr_fast(spk_train1, spk_train2, params):
    """Compute spike cross-correlation among two spike trains.

    The algorithm implements the somehow trivial idea spike-cross-correlation
    measures the amount of coincidences for shifted vectors.
    The code is houndreds of times faster than more naive algorithms.

    spk_train1, spk_train2:  spike train list
    params (dictionary):
        dt
        tw
        tmax
        corr
    """
    st1 = (np.array(spk_train1)/params['dt']).astype(int)
    st2 = (np.array(spk_train2)/params['dt']).astype(int)
    l1 = len(st1)
    l2 = len(st2)
    l12 = l1 * l2
    nstep = int(params['tw'] / params['dt'])
    cc_vect = np.zeros(2 * nstep + 1)
    for k in range(-nstep, nstep+1):
        cc_vect[nstep+k] = len(np.intersect1d(st1, st2+k, True))
    if params['corr']:
        cc_vect -= (l12 / params['tmax']) * params['dt']
    cc_vect /= np.sqrt(l12)
    return cc_vect[::-1]


def cross_corr_bootstrap(spk_train1, spk_train2, params):
    """Compute bootstrap spike-cross-correlation.

    Implements the simplest bootstrap by randomly shuffling the spikes.
    """
    tmax = max(np.max(spk_train1), np.max(spk_train2))
    l1 = len(spk_train1)
    l2 = len(spk_train2)
    spk1_mat_boot = np.sort(np.random.rand(params['nboot'], l1), 1) * tmax
    spk2_mat_boot = np.sort(np.random.rand(params['nboot'], l2), 1) * tmax
    cc_vect_boot = np.zeros(params['nboot'])
    for k in range(params['nboot']):
        cc_boot = cross_corr_fast(spk1_mat_boot[k], spk2_mat_boot[k], params)
        # spk1_mat_boot[k] same as [k,:]
        cc_vect_boot[k] = np.max(cc_boot)
    return np.percentile(cc_vect_boot, q=params['perc'])


def run_cross(fn_data, fn_pos, params):
    """Run cross-correlation."""
    if fn_data.find('.bxr') >= 0:
        # load the bxr file
        spikes_ch_su, units_lst, pos_unit = collect_spikes_ch_su_spont(fn_data,
                                                                       fn_pos)
        spikes_su = conv_to_spikes_su(spikes_ch_su)
    else:
        print(' not yet implemented !!!')

    spike_trains = [spikes_su[id_su] for id_su in list(spikes_su)]
    num_trains = len(spike_trains)
    cc_mat = np.zeros((num_trains, num_trains))
    cc_boot_mat = np.zeros((num_trains, num_trains))
    tmax = np.max([np.max(tk) for tk in spike_trains])
    params['tmax'] = tmax

    for idx_src in tqdm(range(num_trains)):
        spk_train1 = spike_trains[idx_src]
        for idx_dst in range(idx_src+1, num_trains):
            spk_train2 = spike_trains[idx_dst]
            cc_graph = cross_corr_fast(spk_train1, spk_train2, params)
            cc_boot = cross_corr_bootstrap(spk_train1, spk_train2, params)
            cc_mat[idx_src, idx_dst] = np.max(cc_graph)
            cc_boot_mat[idx_src, idx_dst] = cc_boot
    return cc_mat, cc_boot_mat, pos_unit


def run_cross_par(fn_data, fn_pos, params, n_jobs=8):
    """Run cross-correlation in parallel."""
    from joblib import Parallel, delayed
    import time
    if fn_data.find('.bxr') >= 0:
        # load the bxr file
        spikes_ch_su, units_lst, pos_unit = collect_spikes_ch_su_spont(fn_data,
                                                                       fn_pos)
        spikes_su = conv_to_spikes_su(spikes_ch_su)
    else:
        print(' not yet implemented !!!')

    spike_trains = [spikes_su[id_su] for id_su in list(spikes_su)]
    num_trains = len(spike_trains)
    tmax = np.max([np.max(tk) for tk in spike_trains])
    params['tmax'] = tmax

    # build indexes
    idx_sd = []
    for idx_src in range(num_trains):
        for idx_dst in range(idx_src+1, num_trains):
            idx_sd.append((idx_src, idx_dst))

    t0 = time.time()
    cc_lst = Parallel(n_jobs=n_jobs)(
             delayed(cross_corr_fast)
             (spike_trains[idx_src], spike_trains[idx_dst], params)
             for (idx_src, idx_dst) in idx_sd)
    print('cross-corr computed in %g seconds' % (time.time()-t0))

    t0 = time.time()
    cc_lst_boot = Parallel(n_jobs=n_jobs)(
                  delayed(cross_corr_bootstrap)
                  (spike_trains[idx_src], spike_trains[idx_dst], params)
                  for (idx_src, idx_dst) in idx_sd)
    print('cross-corr bootstraps computed in %g seconds' % (time.time()-t0))
    tcc = np.arange(-params['tw'], params['tw'] + params['dt'], params['dt'])
    return cc_lst, cc_lst_boot, idx_sd, tcc, pos_unit


def plot_crosscorr_spiketrain(fn_cc, fn_su, fn_pos, src_dst, dt_tol=0.1):
    """Plot cross-correlation and spike trains.

    fn_cc (keys)
        'cc_lst': spike cross-correlations (size n)
        'cc_lst_boot': max spike cross-corr of shuffled spike trains (size n)
        'idx_sd': source -> destination indexes (size n), relative to BWCG file
        'tcc': time window of spike cross-correlation
        'pos_unit': row, col of units with at least one spike (size m<=n)

    fn_su (keys)

    fn_pos (keys)
        'row':
        'col':

    dt_tol: threshold to declare two spikes coincident

    """
    # cross_correlograms
    data_cc = np.load(fn_cc, allow_pickle=1).item()

    # spike trains
    data_su = np.load(fn_su, allow_pickle=1).item()
    spikes_su = conv_to_spikes_su(data_su['spikes_ch_su'])
    #  spike_trains, size m<=n, same as data_cc['pos_unit']
    spike_trains = [spikes_su[id_su] for id_su in list(spikes_su)]

    # positions
    # fn_pos = general.get_fn_roi(exp_name, 'PC')
    # pos = np.load(fn_pos, allow_pickle=1).item()
    pos = data_cc['pos_unit']

    mat_sd = np.array([(pos['row'][idx_src], pos['col'][idx_src],
                        pos['row'][idx_dst], pos['col'][idx_dst])
                       for idx_src, idx_dst in data_cc['idx_sd']])
    idx_all = []
    for rs, cs, rd, cd in zip(src_dst['row_src'], src_dst['col_src'],
                              src_dst['row_dst'], src_dst['col_dst']):

        idx_search = np.where((mat_sd[:, 0] == rs) & (mat_sd[:, 1] == cs) &
                              (mat_sd[:, 2] == rd) & (mat_sd[:, 3] == cd))[0]
        if len(idx_search):
            for idx in idx_search:
                idx_all.append(idx)
        else:
            print('(%d,%d)->(%d,%d) not found!' % (rs, cs, rd, cd))

        tinf, tsup = data_cc['tcc'][0], data_cc['tcc'][-1]
        for idx in idx_all:
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 1, 1)
            plt.plot(data_cc['tcc'], data_cc['cc_lst'][idx], 'ko-')
            cc_boot = data_cc['cc_lst_boot'][idx]
            plt.plot([tinf, tsup], [cc_boot, cc_boot], 'r-')
            plt.xlabel('correlation lag (ms)', fontsize=18)
            plt.ylabel('spike cross-corr', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid()

            plt.subplot(2, 1, 2)
            rs, cs, rd, cd = mat_sd[idx, :]
            idx_src = np.where((data_cc['pos_unit']['row'] == rs) &
                               (data_cc['pos_unit']['col'] == cs))[0]
            idx_dst = np.where((data_cc['pos_unit']['row'] == rd) &
                               (data_cc['pos_unit']['col'] == cd))[0]
            lun_src = len(idx_src)
            lun_dst = len(idx_dst)
            if lun_src > 1:
                print('There are %d units at position (%d,%d).' % (lun_src,
                                                                   rs, cs))
            if lun_dst > 1:
                print('There are %d units at position (%d,%d).' % (lun_dst,
                                                                   rd, cd))

            spk_src = spike_trains[idx_src[0]]
            spk_dst = spike_trains[idx_dst[0]]
            lun_src = len(spk_src)
            lun_dst = len(spk_dst)
            plt.plot(spk_src, np.repeat(1, lun_src), 'ko', markersize=14)
            plt.plot(spk_dst, np.repeat(2, lun_dst), 'ko', markersize=14)
            # check for almost coincidences
            if lun_src > lun_dst:
                spk_ref = np.copy(spk_dst)
                spk_dst = np.copy(spk_src)
            else:
                spk_ref = np.copy(spk_src)
                spk_dst = np.copy(spk_dst)

            count_coincident = 0
            for spk in spk_ref:
                idx = np.where(np.abs(spk_dst-spk) < dt_tol)[0]
                lun = len(idx)
                if lun:
                    if lun > 1:
                        print('more than one!')
                    plt.plot([spk, spk], [1, 2], 'b--')
                    count_coincident += 1

            plt.xlabel('time (ms)', fontsize=18)
            plt.ylabel('spike trains', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks([1, 2], fontsize=16)
            plt.suptitle('(%d,%d)->(%d,%d)' % (rs, cs, rd, cd), fontsize=20)
            frac = 100 * count_coincident / min(lun_src, lun_dst)
            plt.title('%2.3g %% of coincident spikes (tolerance %g ms)' %
                      (frac, dt_tol), fontsize=20)
            plt.tight_layout(pad=1)


def plot_mfr_pos(fn_ch_su, r=0.4, title=''):
    """Colormap of mean firing rates."""
    data = np.load(fn_ch_su, allow_pickle=1).item()
    spikes_ch_su, pos_unit = data['spikes_ch_su'], data['pos_unit']
    row_min, row_max = np.min(pos_unit['row']), np.max(pos_unit['row'])
    col_min, col_max = np.min(pos_unit['col']), np.max(pos_unit['col'])
    ch_lst = list(spikes_ch_su)
    num_ch_mfr0 = 0
    mfr_dict = {'x': [], 'y': [], 'mfr': []}
    mfr_min, mfr_max = 1e3, -1e3
    k = 0
    for ch in ch_lst:
        su_lst = list(spikes_ch_su[ch])
        num_su_lst = len(su_lst)
        if num_su_lst == 0:
            num_ch_mfr0 += 1
            mfr_dict['x'].append(pos_unit['col'][k])
            mfr_dict['y'].append(pos_unit['row'][k])
            mfr_dict['mfr'].append(0)
            k += 1
        else:
            theta = 2 * np.pi / num_su_lst
            for j, su in enumerate(su_lst):
                nspk = len(spikes_ch_su[ch][su])
                if nspk > 1:
                    mfr = 1000 * nspk / (spikes_ch_su[ch][su][-1] -
                                         spikes_ch_su[ch][su][0])
                else:
                    mfr = 0
                    num_ch_mfr0 += 1
                x = pos_unit['col'][k]
                y = pos_unit['row'][k]
                if len(su_lst) > 1:
                    x += r * np.cos(theta * j)
                    y += r * np.sin(theta * j)
                    # print(x, y)
                mfr_dict['x'].append(x)
                mfr_dict['y'].append(y)
                mfr_dict['mfr'].append(mfr)
                mfr_min = min(mfr, mfr_min)
                mfr_max = max(mfr, mfr_max)
                k += 1
    print('number of electrodes with zero firing = %d' % num_ch_mfr0)

    fig, ax = plt.subplots(figsize=(14, 14))

    plt.ion()
    lun = len(mfr_dict['mfr'])

    mfr_scaled = (np.array(mfr_dict['mfr']) - mfr_min)/(mfr_max - mfr_min)
    colors = [cm.jet(color) for color in mfr_scaled]
    for k in range(lun):
        circle = plt.Circle((mfr_dict['x'][k], mfr_dict['y'][k]), r,
                            color=colors[k])
        ax.add_patch(circle)
    #
    sc = plt.scatter(mfr_dict['x'], mfr_dict['y'], s=0, c=mfr_dict['mfr'],
                     cmap='jet', vmin=0, vmax=mfr_max,
                     facecolors='none')
    plt.grid()
    cbar = plt.colorbar(sc)

    cbar.set_label('MFR (Hz)', rotation=90, labelpad=10, fontsize=20)
    plt.show()
    plt.xticks(np.arange(col_min-1, col_max+2), fontsize=16)
    plt.yticks(np.arange(row_min-1, row_max+2), fontsize=16)

    plt.xlim(col_min-1, col_max+1)
    plt.ylim(row_max+1, row_min-1)
    ax.set_aspect('equal')

    plt.title(title, fontsize=18)


def crosscorr_vs_distance(fn_cc, stitle='', col_boot=-1, tpeak_max=0.1):
    """Perform cross-correlation analysis."""
    from math import sqrt
    data = np.load(fn_cc, allow_pickle=1).item()
    cc_max = np.max(np.array(data['cc_lst']), axis=1)
    idx_sign = np.where(cc_max > np.array(data['cc_lst_boot'])[:, col_boot])[0]
    # print(100*len(idx_sign)/cc_max.shape[0], np.max(cc_max))
    cc = cc_max[idx_sign]
    idx_sd = np.array(data['idx_sd'])[idx_sign, :]
    row, col = data['pos_unit']['row'], data['pos_unit']['col']
    dist = np.array([sqrt((row[s]-row[d])**2+(col[s]-col[d])**2)
                    for s, d in idx_sd])
    # select based on peak cross-correlation time lag
    idx = np.argmax(np.array(data['cc_lst'])[idx_sign], axis=1)
    peak_delay = np.abs(data['tcc'][idx])
    print(peak_delay.min(), peak_delay.max())
    idx = np.where(peak_delay <= tpeak_max)[0]
    plt.plot(dist[idx], cc[idx], 'ko')
    plt.xlabel('inter-electrode-distance', fontsize=14)
    plt.ylabel('spike cross-corr', fontsize=14)
    plt.title(stitle, fontsize=18)
    return cc


def crosscorr_vs_delay(fn_cc, stitle='', col=-1):
    """Perform cross-correlation analysis."""
    data = np.load(fn_cc, allow_pickle=1).item()
    cc_max = np.max(np.array(data['cc_lst']), axis=1)
    idx_sign = np.where(cc_max > np.array(data['cc_lst_boot'])[:, col])[0]
    print(100*len(idx_sign)/cc_max.shape[0], np.max(cc_max))
    # cc
    cc = cc_max[idx_sign]
    # lag
    idx = np.argmax(np.array(data['cc_lst'])[idx_sign], axis=1)
    peak_delay = np.abs(data['tcc'][idx])

    plt.plot(peak_delay, cc, 'ko')
    plt.xlabel('peak_delay (ms)', fontsize=14)
    plt.ylabel('spike cross-corr', fontsize=14)
    plt.title(stitle, fontsize=18)
    # return cc


def plot_cum_cc(fn_cc, exp_name='coronal', col='b', bins=50):
    """Plot cumulative cross-correlation distribution.

    fn_cc
    """
    data = np.load(fn_cc, allow_pickle=1).item()
    cc_max = np.max(np.array(data['cc_lst']), axis=1)
    count, xbin = np.histogram(cc_max, bins=bins)
    freq = count / np.sum(count)
    cumsum = np.cumsum(freq)
    xbin = xbin[:-1] + .5 * (xbin[1] - xbin[0])
    if len(exp_name):
        plt.plot(xbin, cumsum, c=col, label=exp_name, lw=2)
    else:
        plt.plot(xbin, cumsum, c=col, lw=2)


def plot_cross_correlogram(fn_cc, title='', fpath_png='', num_cc=10,
                           significant=True, peak_int=(2, 10), col=-1):
    """Plot cross_correlograms.

    input:
        fn_cc:   cross-correlation data

        title:   title to add to plot and png

        fpath_png: file path of the plots

        num_cc:  > 0 cc plot with the num_cc highest cc peak
                < 0 pick randomly num_cc cc plots

        significant: plot any correlation or just the significant ones

        col:     what percentile of the bootstrap distribution to use for
                 acceptance. -1 corresponds to highest percentile, typically
                 it will correspond to the 99th percentile
    """
    data = np.load(fn_cc, allow_pickle=1).item()
    pos_unit = data['pos_unit']
    cc_max = np.max(np.array(data['cc_lst']), axis=1)
    num_val = len(cc_max)
    if significant:
        idx_sign = np.where(cc_max > np.array(data['cc_lst_boot'])[:, col])[0]
    else:
        idx_sign = np.arange(num_val)

    # lag
    if len(peak_int) == 2:
        idx = np.argmax(np.array(data['cc_lst'])[idx_sign], axis=1)
        peak_delay = np.abs(data['tcc'][idx])
        idx = np.where((peak_delay > peak_int[0]) & (peak_delay < peak_int[1]))
        idx_sign = idx_sign[idx]

    # indexes of the cc to plot
    if num_cc > 0:
        # sort the cc_max of index idx_sign and the highest num_cc values
        idx_sort = np.argsort(cc_max[idx_sign])[-num_cc:]
        idx_sel = idx_sign[idx_sort]
    else:
        idx_sel = np.random.choice(idx_sign, np.abs(num_cc),
                                   replace=False)
    # indexes of the cc to plot
    tinf, tsup = data['tcc'][0], data['tcc'][-1]
    plt.ioff()
    for k in idx_sel:
        plt.figure()
        src = data['idx_sd'][k][0]
        dst = data['idx_sd'][k][1]
        src_str = '(%d, %d)' % (pos_unit['row'][src], pos_unit['col'][src])
        dst_str = '(%d, %d)' % (pos_unit['row'][dst], pos_unit['col'][dst])
        stit = '%s [ %s -> %s ]' % (title, src_str, dst_str)
        plt.title(stit, fontsize=16)
        plt.plot(data['tcc'], data['cc_lst'][k], 'ko-')
        cc_boot = data['cc_lst_boot'][k]
        plt.plot([tinf, tsup], [cc_boot, cc_boot], 'r-')
        plt.grid()
        fn_png = os.path.join(fpath_png, stit)
        plt.savefig(fn_png)
        plt.close()
    plt.ion()
