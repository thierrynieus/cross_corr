# Spike cross-correlation module


The repository is designed for efficient spike cross-correlation calculations. It uses the module joblib to split the calculations on different cores.

NOTE: Have a look to @FMarmoreo 's implementation based on sparse matrix (https://github.com/FMarmoreo/cross_spike_train/blob/main/spike_train.ipynb)

## File description
1. spike_cross_corr.py, a module to load and compute cross-correlations
2. cross_corr_demo1.ipynb, a demo to illustrate how to use the module
3. cross_corr_demo2.ipynb, a demo to illustrate the performances of the extremely faster calc_cc_sp function by F Marmoreo.

Parameters of spike_cross_corr.py:
````
params = {'dt': 0.5, 'tw': 5, 'tmax': 6e5, 'corr': True}
````
1. 'dt' is the time resolution to use for spike cross-correlation
2. 'tw' is the time window ([-tw, +tw]) on which the correlation has to be computed 
3. 'tmax' is the duration of recording, by default it is estimated from the data 
4. 'corr' if True (default), 'tmax' will be adjusted (see before)


## Functions


load_data 
    The function loads a 2D array (2 rows x n columns, n is the totale amount of spikes).
    row 1: channel ID
    row 2: spike timing (ms)

The working principle of the remaining functions is illustrated in the notebook (cross_corr_demo1.ipynb).
    

If you use the code in your own work please cite the following publication:

Nieus, T., D’Andrea, V., Amin, H., Di Marco, S., Safaai, H., Maccione, A., et al. (2018). State-dependent representation of stimulus-evoked activity in high-density recordings of neural cultures. Sci Rep 8, 5578. doi: 10.1038/s41598-018-23853-x.


NOTE: The module now includes the much faster algorithm by F Marmoreo.