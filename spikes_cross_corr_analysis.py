import pandas as pd
import numpy as np
import pylab as plt
import glob
import os
from scipy.stats import ttest_ind

import spike_train_analysis as sta

fpath_code = '/home/tnieus/Projects/CODE/PCIcerebellum/'
fpath_results = '/home/tnieus/Projects/RESULTS/PCIcerebellum/'

fpath_res = os.path.join(fpath_results, 'spontaneous')

fn_xls = os.path.join(fpath_code, 'support/temporal_sequence.xls')
df = pd.read_excel(fn_xls)
# df[df['filename'].str.contains('coronal')]['exp_name'].to_numpy()
# df[df['filename'].str.contains('sagittal')]['exp_name'].to_numpy()

perc = [5, 50, 90, 95, 99]


fn_cc_replace = 'activity_cross_corr_tw2_dt0.1.npy'
folder_png_tmp = 'MFR_filtered_min%dHz_tw2_dt0.1/perc_%d'
# folder_png_tmp = 'MFR_filtered_min%dHz/perc_%d'
do_fit = True


def spike_cross_correlation_boxplot(fname_lst, sign):
    """Report box-plot of spike cross-correlation values subdivided by slice cut.

    fname_lst: { = glob.glob(os.path.join(fpath_res, '**', str_search),
                          recursive=True)
                where str_search='*tw2_dt0.1_filtered_MFRmin10Hz.npy'
                    or str_search='*tw20_dt1_filtered.npy' }
    sign: { = 99, 95 ... }
    """
    col = perc.index(sign)
    spike_cross_corr = {}
    spike_cross_corr['coronal'] = []
    spike_cross_corr['sagittal'] = []
    for fname in fname_lst:
        data = np.load(fname, allow_pickle=1).item()
        cc_max = np.max(np.array(data['cc_lst']), axis=1)
        idx_sign = np.where(cc_max > np.array(data['cc_lst_boot'])[:, col])[0]
        cc_max = cc_max[idx_sign]
        exp_name = os.path.basename(fname).split('_')[0].lower()
        slice_cut = df.loc[df['exp_name'] == exp_name, 'slice'].to_numpy()[0]
        spike_cross_corr[slice_cut].extend(cc_max)
    print(len(spike_cross_corr['sagittal']), len(spike_cross_corr['coronal']))
    plt.boxplot([spike_cross_corr['sagittal'], spike_cross_corr['coronal']],
                whis=[5, 95])
    plt.xticks([1, 2], ['sagittal', 'coronal'], fontsize=12)
    plt.ylabel('spike crosscorrelation peak', fontsize=14)
    plt.grid()


def spike_cross_correlation_delay_distance(fname_lst, sign, mfr_min=30):
    """Analyze spike cross-correlation values versus delay and distance.

    fname_lst: { = glob.glob(os.path.join(fpath_res, '**', '*activity.npy'),
                         recursive=True) }

    """
    folder_png = folder_png_tmp % (mfr_min, sign)
    plt.ioff()
    col = perc.index(sign)
    dct_out = {}
    dct_out['fname'] = []
    dct_out['b_cc_dist'] = []
    dct_out['b_cc_delay'] = []
    for fn in fname_lst:
        # activity & mfr
        print(fn)
        dct_out['fname'].append(fn)
        d = np.load(fn, allow_pickle=1).item()
        spikes_su = sta.conv_to_spikes_su(d['spikes_ch_su'])
        spikes_row_col = sta.conv_to_spikes_row_col(spikes_su, d['pos_unit'])
        mfr_dict = sta.mean_firing_spikes_row_col(spikes_row_col)
        # cross-corr
        fn_cc = fn.replace('activity.npy', fn_cc_replace)
        d = np.load(fn_cc, allow_pickle=1).item()
        out = sta.filter_crosscorr_files(fn_cc, mfr_dict, mfr_lim=mfr_min)
        fn_cc_new = fn_cc.replace('.npy', '_filtered_MFRmin%gHz.npy' % mfr_min)
        np.save(fn_cc_new, out)
        # cc vs delay
        popt = sta.crosscorr_vs_delay(fn_cc_new, col=col, do_fit=do_fit)
        dct_out['b_cc_delay'].append(popt[1])
        fn_cc_new_png = fn_cc_new.replace('.npy', '_cc_vs_delay.png')
        fn_png = os.path.basename(fn_cc_new_png)
        fn_cc_new_png = os.path.join(fpath_res, folder_png, 'cc_vs_delay',
                                     fn_png)

        plt.savefig(fn_cc_new_png)
        plt.close()
        # cc vs distance
        popt = sta.crosscorr_vs_distance(fn_cc_new, dist_min=0, tpeak_max=20,
                                         col=col, do_fit=do_fit,
                                         int_fit=[0.5, 10])
        dct_out['b_cc_dist'].append(popt[1])
        fn_cc_new_png = fn_cc_new.replace('.npy', '_cc_vs_distance.png')
        fn_png = os.path.basename(fn_cc_new_png)
        fn_cc_new_png = os.path.join(fpath_res, folder_png, 'cc_vs_distance',
                                     fn_png)

        plt.savefig(fn_cc_new_png)
        plt.close()

    plt.ion()
    return dct_out


def spike_cross_correlation_delay_distance_bp(dct_out):
    """Report spike-cross-correlation values versus distance and delay."""
    fpath_code = '/home/tnieus/Projects/CODE/PCIcerebellum/'
    fn_xls = os.path.join(fpath_code, 'support/temporal_sequence.xls')
    df = pd.read_excel(fn_xls)

    b_coeff = {}
    b_coeff['b_cc_dist'] = {}
    b_coeff['b_cc_delay'] = {}
    for slice_cut in ['coronal', 'sagittal']:
        b_coeff['b_cc_dist'][slice_cut] = []
        b_coeff['b_cc_delay'][slice_cut] = []

    for fname, b_cc_dist, b_cc_delay in zip(dct_out['fname'],
                                            dct_out['b_cc_dist'],
                                            dct_out['b_cc_delay']):
        exp_name = os.path.basename(fname).split('_')[0].lower()
        slice_cut = df.loc[df['exp_name'] == exp_name, 'slice'].to_numpy()[0]
        b_coeff['b_cc_dist'][slice_cut].append(b_cc_dist)
        b_coeff['b_cc_delay'][slice_cut].append(b_cc_delay)

    pop1_dist = np.array(b_coeff['b_cc_dist']['coronal'])
    pop2_dist = np.array(b_coeff['b_cc_dist']['sagittal'])
    pop1_del = np.array(b_coeff['b_cc_delay']['coronal'])
    pop2_del = np.array(b_coeff['b_cc_delay']['sagittal'])

    plt.figure(1)
    plt.boxplot([pop1_dist, pop2_dist], whis=[5, 95])
    plt.xticks([1, 2], ['coronal', 'sagittal'], fontsize=12)
    plt.title('cc vs dist', fontsize=14)
    plt.figure(2)
    plt.boxplot([pop1_del, pop2_del], whis=[5, 95])
    plt.xticks([1, 2], ['coronal', 'sagittal'], fontsize=12)
    plt.title('cc vs delay', fontsize=14)
    print('cc_delay', ttest_ind(pop1_dist, pop2_dist, equal_var=False))
    print('cc_dist', ttest_ind(pop1_del, pop2_del, equal_var=False))
