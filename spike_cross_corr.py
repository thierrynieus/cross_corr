"""
In the current version it includes the function <<calc_cc_sp>> by F Marmoreo
see:
    https://github.com/FMarmoreo/cross_spike_train/blob/main/spike_train.ipynb
"""


import numpy as np

params = {'dt': 0.5, 'tw': 5, 'tmax': 6e5, 'corr': True}


def load_data(fname_spike_train, update_tmax=True):
    """Load data."""
    data = np.load(fname_spike_train, allow_pickle=1)
    # convert to list of spike trains
    spk_trains = []
    id_ch_unique = np.unique(data[0, :])
    for id_ch in id_ch_unique:
        idx = np.where(data[0, :] == id_ch)[0]
        spk_trains.append(data[1, idx])
    tmax = np.max(data[1, idx])
    if update_tmax:
        params['tmax'] = tmax
    # id_ch_unique: id <-> (row, column) spatial coordinates
    return spk_trains, id_ch_unique, tmax


def format_spike_train(spk_trains):
    """Format spike trains.

    spk_trains  a list of spike trains
    """
    idx_spk_tr, len_spk_tr = [], []
    for spk_train in spk_trains:
        spk_train_arr = np.array(spk_train)
        idx_spk_train = (spk_train_arr/params['dt']).astype(int)
        len_spk_tr.append(len(idx_spk_train))
        idx_spk_tr.append(idx_spk_train)
    return idx_spk_tr, len_spk_tr


def calc_cc(idx_spk_tr1, idx_spk_tr2, lun_spk_tr1, lun_spk_tr2):
    """Compute spike cross-correlation across two spike trains.

    The algorithm implements the somehow trivial idea that spike cross
    correlation measures the amount of coincidences for shifted vectors.
    The code is houndreds of times faster than more naive algorithms.
    """
    l12 = lun_spk_tr1 * lun_spk_tr2
    nstep = int(params['tw'] / params['dt'])
    cc_vect = np.zeros(2 * nstep + 1)
    for k in range(-nstep, nstep+1):
        coincidences = np.intersect1d(idx_spk_tr1, idx_spk_tr2+k, True)
        cc_vect[nstep+k] = len(coincidences)
    if params['corr']:
        cc_vect -= (l12 / params['tmax']) * params['dt']
    cc_vect /= np.sqrt(l12)
    return cc_vect[::-1]


def run_cross_par(idx_spk_tr, len_spk_tr, n_jobs=4):
    """Run cross-correlation in parallel.

    idx_spk_tr
    len_spk_tr
    params

    """
    from joblib import Parallel, delayed
    import time

    num_trains = len(idx_spk_tr)

    # build indexes
    idx_sd = []
    idx = np.triu_indices(num_trains, 1)
    for idx_src, idx_dst in zip(*idx):
        idx_sd.append((idx_src, idx_dst))

    t0 = time.time()
    cc_lst = Parallel(n_jobs=n_jobs)(
             delayed(calc_cc)
             (idx_spk_tr[idx_src], idx_spk_tr[idx_dst],
              len_spk_tr[idx_src], len_spk_tr[idx_dst])
             for (idx_src, idx_dst) in idx_sd)
    print('cross-corr computed in %g seconds' % (time.time()-t0))
    tcc = np.arange(-params['tw'], params['tw'] + params['dt'], params['dt'])
    return cc_lst, idx_sd, tcc


def calc_cc_sp(idx_spk_tr, len_spk_tr, params):
    """
    compute cross correletions between channels.

    return KxMxM matrix. K time delays, M channels
    """
    import time
    from scipy.sparse import csr_matrix, lil_matrix
    
    timo = time.time()
    max_idx = np.max([np.max(i) for i in idx_spk_tr])
    num_trains = len(idx_spk_tr)
    nstep = int(params['tw'] / params['dt'])
    padded = lil_matrix((len(idx_spk_tr),max_idx+2*nstep+1))
    ccmat= np.zeros((nstep*2+1,num_trains,num_trains))
    for i in range(num_trains):
        padded[i,(idx_spk_tr[i]-1+nstep)]=1
    padded = csr_matrix(padded)
    ref = padded[:,nstep:-(nstep+1)]
    
    sizes = np.array(len_spk_tr)
    sizes = sizes.reshape(len(sizes), 1)
    lijs = csr_matrix(sizes.dot(sizes.T))
    lijs_sc = (lijs/params['tmax'])*params['dt']
    for k in range(nstep*2+1): 
        ccmat_ = ref.dot(padded[:,k:(max_idx+k)].T)
        ccmat_ -= lijs_sc
        ccmat_ /=  lijs.sqrt()
        ccmat[k] = ccmat_
    print('cross-corr computed in {:.2f} seconds \n'.format(time.time()-timo))

    return ccmat
