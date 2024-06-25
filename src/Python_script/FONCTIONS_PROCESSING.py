
def extract_temporal_features(X):
    """
    Extact mean, std and higher order moments
    """
    features = []
    for epoch in X:
        mean = np.mean(epoch, axis=-1)
        std = np.std(epoch, axis=-1)
        features.append([mean,std])
    return features
def find_gaming_set_files(directory):
    '''Find gaming .set files '''
    # Liste pour stocker les chemins des fichiers correspondants
    matching_files = []

    # Parcourir tous les fichiers dans le répertoire donné
    for root, _, files in os.walk(directory):
        for file in files:
            # Vérifier si le fichier commence par "GAMING" et se termine par ".set"
            if file.startswith("GAMING") and file.endswith(".set"):
                # Ajouter le chemin complet du fichier à la liste
                matching_files.append(os.path.join(root, file))
    
    return matching_files
def Inverse_calculator(epochs , method = 'dSPM'):
    '''Calcul of the inverse operator and apply it on epochs to return source'''
    info = epochs.info

    # Calculer la matrice de covariance de bruit
    noise_cov = mne.compute_covariance(epochs, tmax=0.0, method="auto", rank=None)
    # Utiliser un modèle de tête standard fourni par MNE
    subjects_dir = 'C:/Users/robinaki/mne_data/MNE-sample-data/subjects'


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

    inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)


    stc= mne.minimum_norm.apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method=method,
    pick_ori=None,
    verbose=True,
    )
    return stc
def src_data_to_X_data(stc):
    '''extract data from source objects'''
    X_src = []
    for epoch in stc:
        X_src += [epoch.data]
    return X_src
def pre_process_run(raw_fnames ,tmin = -0.5 , tmax = 1 ,f_low = 1, L_H_freq = [(8,30)],Brain_visu = True,baseline = (None, 0),Notchs_freq = (50,60),eeg_reject = 150e-3):
    """ process data in .set format and do source localisation algorithm """
    
    # Initialiser les listes pour les données
    X_all_runs = []
    y_all_runs = []


    reject = dict(eeg=eeg_reject)


    # Charger les fichiers et créer des époques pour chaque run
    N_dir = len(raw_fnames)
    N_freq = len(L_H_freq)

    j = 0
    for raw_fname in raw_fnames:
        raw = mne.io.read_raw_eeglab(raw_fname,preload=True)
        raw.filter( f_low, None)
        raw.notch_filter(Notchs_freq)
        # Appliquer ASR pour nettoyer les artefacts
        # asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
        # asr.fit(raw)
        # raw = asr.transform(raw)
        # ica = ICA()
        # ica.fit(raw)
        # raw= ica.apply(raw)
        X_run = []
        montage = mne.channels.make_standard_montage('standard_1020')
        
        raw.set_montage(montage)
        raw.set_eeg_reference('average', projection=True)

        i = 0
        for l_freq, h_freq in L_H_freq:
        
            print('avancement',i/N_freq,j/N_dir)
            raw2 = raw.copy()
            raw2.filter(l_freq, h_freq)       
            events , event_id= mne.events_from_annotations(raw2)
            event_id_RL= {'right' : event_id['right'] , 'left' : event_id['left']}
            events =mne.pick_events(events, include=[event_id['right'],event_id['left']])
            epochs = mne.Epochs(raw2, events, event_id_RL, tmin, tmax, proj=True, baseline=baseline, reject=reject, preload=True)
            if Brain_visu:
                src = Inverse_calculator(epochs)
                X_src= src_data_to_X_data(src)
                X_run += [X_src]
                del X_src
            else:
                X_run += [epochs.get_data(copy=True)]
            del raw2
            i+=1
        del raw
        j+=1
        X_run = np.moveaxis(X_run, 0, 1)
        y_run = epochs.events[:, -1] == event_id['right']


        X_all_runs.append(X_run)
        y_all_runs.append(y_run)
        del X_run
    return X_all_runs , y_all_runs
def make_featuring(X_features , y_all_runs): 
    '''call extract_temporal_features and reshape the feature in a 2 dim files of size (N_epochs,N_features ) for classification '''
    
    X_ccat = np.moveaxis(X_features,2,0)
    SHAPE = np.shape(X_ccat)
    X_feat_reshaped = np.reshape(X_ccat,(SHAPE[0], SHAPE[1]* SHAPE[2]*SHAPE[3]*SHAPE[4]))
    y = np.concatenate(y_all_runs, axis=0)
    return X_feat_reshaped , y
def Sores_RandomForest(X_train, X_test, y_train, y_test,n_estimators = 100 , model = RandomForestClassifier()):
    selector = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators))
    selector.fit(X_train, y_train)
    X_train_reduced = selector.transform(X_train)
    X_test_reduced = selector.transform(X_test)

    model.fit(X_train_reduced, y_train)
    y_pred = model.predict(X_test_reduced)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy 
def Scores_feature_select(X_train, X_test, y_train, y_test ,model_feature_select=PCA(n_components=50),model = RandomForestClassifier()):
    """ Input: Train and test data for training selected classifier with selected features selection
        Output : Accuracy for the test set """

    model_select = model_feature_select

    X_train_select = model_select.fit_transform(X_train,y_train)
    X_test_select = model_select.transform(X_test)

    # Entraîner un modèle de classification avec GradientBoostingClassifier
    model.fit(X_train_select, y_train)

    # Prédire et évaluer le modèle
    y_pred = model.predict(X_test_select)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    return accuracy
def Scores_no_feature_select(X_train, X_test, y_train, y_test ,model = RandomForestClassifier()):
    """ Input: Train and test data for training selected classifier with no  features selection Output : Accuracy for the test set """
    model.fit(X_train, y_train)
    # Prédire et évaluer le modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    return accuracy