import numpy as np
import mne
import matplotlib.pyplot as plt
# %%
# Chemin vers les fichiers EEG et les positions des électrodes
raw_fnames = 'C:/recordings/Game_recordings_test/TOUT_4_second_apple/GAMING_APPLE_1_noc.set'
montage_fname = 'path/to/montage_file.elc'  # Remplacez par le chemin vers votre fichier de montage

# Charger les données EEG
raw = mne.io.read_raw_eeglab(raw_fnames, preload=True)

# Charger et définir le montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
raw.set_eeg_reference('average', projection=True)

new_srate = 25
tmin = -0.5  # début de chaque époque (1,0 secondes avant le déclencheur)
tmax = 1.5  # fin de chaque époque (au déclencheur)
baseline = (-0.5, 0)  # baseline du début à t = 0
reject = dict(eeg=150e-3)  # seuil de rejet pour les électrodes EEG
L_H_freq = [(8,13),(13,30)]
Notchs_freq = (50,60)
f_low = 1
raw.filter( f_low, None)
raw.notch_filter(Notchs_freq)
# raw.resample(new_srate)

events , event_id= mne.events_from_annotations(raw)
event_id_RL= {'right' : event_id['right'] , 'left' : event_id['left']}
event_id_R =  {'right' : event_id['right']}
event_id_L= {'left' : event_id['left']}
epochs = mne.Epochs(raw, events, event_id_RL, tmin, tmax, proj=True, baseline=baseline, reject=reject, preload=True)
epochsR =  mne.Epochs(raw, events, event_id_R, tmin, tmax, proj=True, baseline=baseline, reject=reject, preload=True)
epochsL =  mne.Epochs(raw, events, event_id_L, tmin, tmax, proj=True, baseline=baseline, reject=reject, preload=True)

info = epochs.info

# Calculer la matrice de covariance de bruit
noise_cov = mne.make_ad_hoc_cov(info)

# Utiliser un modèle de tête standard fourni par MNE
subjects_dir = 'C:/Users/robinaki/mne_data/MNE-sample-data/subjects'


# noise_cov = mne.make_ad_hoc_cov(info)
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
bem = mne.read_bem_solution(
'C:/Users/robinaki/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'
    )
src = mne.setup_source_space("fsaverage", spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)
fwd = mne.make_forward_solution(
        info,
        trans="fsaverage",
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=None,
        verbose=False,
    )




# Calculer la solution inverse
# %%
inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
# stc = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=1.0 / 9.0, method='sLORETA')

# %%

evokedL = epochsL.average()
evokedR = epochsL.average()
stcR, residualR = stc = mne.minimum_norm.apply_inverse(
    evokedR,
    inverse_operator,
    lambda2 = 1/9,
    method='sLORETA',
    pick_ori=None,
    return_residual=True,
    verbose=True,
)
stcR, residualR = mne.minimum_norm.apply_inverse(
    evokedR,
    inverse_operator,
    lambda2 = 1/9,
    method='sLORETA',
    pick_ori=None,
    return_residual=True,
    verbose=True,
)
# evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5))



# %%
stcR.plot()
