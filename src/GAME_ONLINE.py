# %% Library importation 
import numpy as np
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import pygame
import sys
import random
import time
import pickle
from joblib import dump, load
import logging
import threading
from pythonosc import dispatcher, osc_server
from pylsl import StreamInfo, StreamOutlet
import os
import stat
import numpy as np
import mne
import asrpy
import gc
import numpy as np
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, LeaveOneOut, LeavePOut, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.feature_selection import SelectFromModel
# %% definition of processing data function 
def extract_temporal_features_live(X):
    """
    Extact mean, std of epochs from one raw data
    """
    mean = np.mean(X, axis=-1)
    std = np.std(X, axis=-1)
    return [mean,std]
def extract_temporal_features(X):
    """
    Extact mean, std of epochs from a list of raw data 
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
def Inverse_calculator(info):
    '''Calcul of the inverse operator and apply it on epochs to return source'''

    # Calculer la matrice de covariance de bruit
    noise_cov  = mne.make_ad_hoc_cov(info)
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
    return inverse_operator
def epochs_to_stc(epochs,inverse_operator,method = 'dSPM'):
    stc= mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=1.0 / 9.0,
        method=method,
        pick_ori=None,
        verbose=False,
        )
    return stc


def pre_process_run_game(raw_fnames ,info,inverse_operator,tmin = -0.5 , tmax = 1 ,f_low = 1, L_H_freq = [(8,12),(12,30)],Brain_visu = True,baseline = (-0.5, 0),Notchs_freq = (50,60),eeg_reject = 150e-3):
    """ process data in .set format and do source localisation algorithm """
    
    # Initialiser les listes pour les données
    X_run = []
    y_run = []
    reject = dict(eeg=eeg_reject)
    

    # Charger les fichiers et créer des époques pour chaque run
    N_dir = len(raw_fnames)
    N_freq = len(L_H_freq)

    j = 0
    for raw_fname in raw_fnames:
        print('pre-processing of file:',raw_fname)
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
        
        montage = mne.channels.make_standard_montage('standard_1020')
        
        raw.set_montage(montage)
        raw.set_eeg_reference('average', projection=True)
        events , event_id= mne.events_from_annotations(raw)
        event_id_RL= {'right' : event_id['right'] , 'left' : event_id['left']}
        events =mne.pick_events(events, include=[event_id['right'],event_id['left']])
    

        i = 0
        for l_freq, h_freq in L_H_freq:
            print('avancement',i/N_freq,j/N_dir)
            raw2 = raw.copy()
            raw2.filter(l_freq, h_freq)       
            epochs = mne.Epochs(raw2, events, event_id_RL, tmin, tmax, proj=True, baseline=baseline, reject=reject, preload=True)
            
            y_run =list(epochs.events[:, -1] == event_id['right'])
            if Brain_visu:
                src = epochs_to_stc(epochs,inverse_operator,method = 'dSPM')
                X_src= src_data_to_X_data(src)
                X_features = extract_temporal_features(X_src)
                del X_src
                
                shape = np.shape(X_features)
                X_features = np.reshape(X_features,(shape[0],shape[1]*shape[2]))
                
            else:
                X_chann= epochs.get_data(copy=True)
                X_features = extract_temporal_features(X_chann)
                del X_chann
                
            del raw2

            if i==0:
                X_run = X_features
            else: 
                X_run = np.concatenate([X_run,X_features],axis= 1)
            gc.collect()
        
            i+=1
        del raw
        gc.collect()
        if j==0:
            X_all_run = X_run
            y_all_runs = y_run
        else: 
            X_all_run = np.concatenate([X_all_run,X_run],axis= 0)
            y_all_runs +=y_run
        print('c',np.shape(X_all_run))

        j+=1


    print('coucou')
    return X_all_run , y_all_runs, inverse_operator, info

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
            info = epochs.info
            if Brain_visu:
                inverse_operator = Inverse_calculator(info)
                src = epochs_to_stc(epochs,inverse_operator,method = 'dSPM')
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
    return X_all_runs , y_all_runs, inverse_operator, info
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
def find_model(path, keyword):
    """
    Trouve le nom entier d'un fichier dans le chemin spécifié dont le nom contient le mot clé donné.

    :param path: Chemin du répertoire à parcourir.
    :param keyword: Mot clé à rechercher dans les noms de fichiers.
    :return: Nom complet du fichier si trouvé, sinon None.
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            print(root)
            if keyword in file:
                return os.path.join(root, file)
    print("no model found")
    return None
def create_mne_info_from_lsl(inlet_info,ch_names):
    """
    Create an mne.Info object from LSL stream information.

    Parameters:
    inlet_info: pylsl.StreamInfo
        The stream information object from a pylsl.StreamInlet.

    Returns:
    mne.Info
        The MNE Info object created from the inlet information.
    """
    # Extracting necessary information from the inlet_info object
    n_channels = inlet_info.channel_count()
    sfreq = inlet_info.nominal_srate()

    # Define standard channel types, assuming EEG for example purposes
    ch_types = ['eeg'] * n_channels

    # Create the info structure needed by MNE
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Load standard montage (here we use 'standard_1020' as an example)
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    # Get the position of the reference electrode (A1)
    ref_pos = None
    if 'A1' in montage.ch_names:
        ref_index = montage.ch_names.index('A1')
        ref_pos = montage.dig[ref_index + 3]['r']  # +3 to skip fiducials and nasion

    # Find and set locations for each channel if available in the montage
    montage_ch_names = montage.ch_names
    for i, ch_name in enumerate(ch_names):
        loc = np.zeros(12)  # Initialize with zeros
        if ch_name in montage_ch_names:
            ch_index = montage_ch_names.index(ch_name)
            pos = montage.dig[ch_index + 3]['r']  # +3 to skip fiducials and nasion
            loc[:3] = pos  # Set the nominal channel position
            if ref_pos is not None:
                loc[3:6] = ref_pos  # Set the reference channel position
        info['chs'][i]['loc'] = loc
    
    return info
# %% Path work

# Absolute path of the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
Save_model = True
record_game_directory = os.path.join(script_directory, 'RECORDS')
raw_fnames_dir = find_gaming_set_files(record_game_directory)
print(record_game_directory)
# Change the current working directory to 'src/APPLE_GAME'

raw_fnames=[ 'C:/recordings/Game_recordings_test/TOUT_4_second_apple/GAMING_APPLE_2_cheat_2.set' ]
training_file =  ['C:/Users/robinaki/Documents/NTNU-EEG/src/Data_games/TRAINING_data_1,0_6_27_14']

training_file = ['C:/Users/robinaki/Documents/NTNU-EEG/src/Data_games/TRAINING_data_0,0_6_27_16']
training_file = 0


# Construct the absolute path to 'src/APPLE_GAME'
apple_game_directory = os.path.join(script_directory, 'APPLE_GAME')
# Change the current working directory to 'src/APPLE_GAME'
os.chdir(apple_game_directory)
Brain_visu = True
directory_path = "C:/recordings/Game_recordings_test/RECORDS"


# %% Initializing game eeg live
 
# Trouver un flux EEG disponible (par exemple, par son type)
print("Recherche d'un flux EEG...")
streams = resolve_stream('name', 'RobinEEG')
inlet = StreamInlet(streams[0],)
Inlet_info = inlet.info() 

Sfreq = Inlet_info.nominal_srate()
print(Sfreq)
n_chan = min(32, Inlet_info.channel_count())  # Ne traquer que les 32 premiers canaux
description = Inlet_info.desc()
ch_names = ['C5', 'C6', 'C3', 'C1', 'FC3', 'C4', 'C2', 'FC4']

# Logger setup for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a new stream for markers
load_time_seconds_before = 2  # Time between each apple drop
load_time_seconds_marker = 1
load_time_seconds = 6
info = StreamInfo('markers', 'Markers', 1, 1/load_time_seconds, 'string', 'MyMarkerStream')
outlet = StreamOutlet(info)
sample,timestamp = inlet.pull_chunk(timeout=1.0)
# Constants
TRAINING_MODE = 2
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PLAYER_WIDTH = 150
PLAYER_HEIGHT = 150
APPLE_SIZE = 80
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_COLOR = (0, 0, 0)
APPLE_COLOR = (255, 0, 0)
LEFT_HAND_OPEN_PATH = "left_hand_open.png"  # Path to the left hand open image
LEFT_HAND_CLOSED_PATH = "left_hand_closed.png"  # Path to the left hand closed image
RIGHT_HAND_OPEN_PATH = "right_hand_open.png"  # Path to the right hand open image
RIGHT_HAND_CLOSED_PATH = "right_hand_closed.png"  # Path to the right hand closed image
APPLE_IMAGE_PATH = "apple.png"  # Path to the apple image
TREE_IMAGE_PATH = "treee.png"
LOAD_BAR_HEIGHT = 20
LOAD_BAR_COLOR = (0, 255, 0)  # Green color for the load bar
MARKER_BAR_COLOR = (255, 128, 0)  # Orange color for the marker line
FPS = 30

# Initialize global variable

class Game:
    def __init__(self,PLAYER_HEIGHT,PLAYER_WIDTH,SCREEN_WIDTH,LOAD_BAR_HEIGHT,SCREEN_HEIGHT,APPLE_SIZE,TRAINING_MODE,load_time_seconds_before,load_time_seconds_marker,load_time_seconds,training_file,Save_model,Inlet_info,ch_names= ch_names):
        pygame.init()
        self.Save_model =Save_model  
        # definition of pre-processing constant 
        self.time_of_window = 4
        self.L_H_freq = [(8,30)]
        self.f_low = 1
        
        self.Notchs_freq = (50,60)
        self.TABLE_score = []
        self.time_crop = load_time_seconds_marker
        self.ldtb = load_time_seconds_before
        self.ldtm = load_time_seconds_marker
        self.ldt = load_time_seconds
        self.TRAINING_MODE = TRAINING_MODE
        self.initialize_model(training_file,Inlet_info,ch_names)
        if TRAINING_MODE:
            self.X_F_stock =[]
            self.Y_stock = []
        self.score = 0
        self.failures = 0
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + LOAD_BAR_HEIGHT))
        pygame.display.set_caption("Apple Catcher Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.marker_sent = False
        self.player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - PLAYER_HEIGHT]
        self.apple_speed = (SCREEN_HEIGHT -APPLE_SIZE/2  )/ (load_time_seconds * FPS)
        self.start_time = time.time()
        self.input_processed= True
        self.tree_image = pygame.image.load(TREE_IMAGE_PATH)
        self.marker_not_finished = True
        self.prob = 0.2
        self.p = 0.5
        # Load and scale images
        self.left_hand_open = pygame.image.load(LEFT_HAND_OPEN_PATH)
        self.left_hand_open = pygame.transform.scale(self.left_hand_open, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.left_hand_closed = pygame.image.load(LEFT_HAND_CLOSED_PATH)
        self.left_hand_closed = pygame.transform.scale(self.left_hand_closed, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.right_hand_open = pygame.image.load(RIGHT_HAND_OPEN_PATH)
        self.right_hand_open = pygame.transform.scale(self.right_hand_open, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.right_hand_closed = pygame.image.load(RIGHT_HAND_CLOSED_PATH)
        self.right_hand_closed = pygame.transform.scale(self.right_hand_closed, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.apple_image = pygame.image.load(APPLE_IMAGE_PATH)
        self.apple_image = pygame.transform.scale(self.apple_image, (APPLE_SIZE, APPLE_SIZE))
        self.hand_status = {"left": "closed", "right": "closed"}



    def CREATE_EPOCHS(self,raw):
        # Assurez-vous que les timestamps et samples ont des formes compatibles
        X_featured = []
        for l_freq, h_freq in self.L_H_freq:
            raw2 = raw.copy()
            raw2.filter(l_freq, h_freq) 
            # Définir les périodes d'intérêt pour couvrir tout le timestamp
            tmin, tmax = -0.5, self.time_crop # en secondes
            # Créer des epochs
            epochs_ = mne.Epochs(raw2, self.events,  tmin=tmin, tmax=tmax,baseline=(-0.5,0), preload=True)#
            src = epochs_to_stc(epochs_,self.inverse_operator,method = 'dSPM')
            X_src = src[0].data
            print(np.shape(np.array(X_src)))
            X_featured += [extract_temporal_features_live(X_src)]
        return X_featured
    def stock_features(self,y):
        self.X_F_stock.append(self.X_featured[0]) 
        self.Y_stock.append(y) 
    def pre_process_run_update(self):
        # Convertir les timestamps en secondes (s'ils sont en millisecondes)
        TIME_STAMPLE= int((self.time_of_window-self.time_crop)*self.info['sfreq'])
        print('TIME_STAMPLE',TIME_STAMPLE)
        self.events = np.array([[TIME_STAMPLE, 0, 1]])  # Un seul événement couvrant tout l'epoch A CHANGER POUR RIGHT ET LEFT
        
        # Créer un RawArray
        raw = mne.io.RawArray(np.moveaxis(self.sample,1,0), self.info)
        raw.filter( self.f_low, None)
        raw.notch_filter(self.Notchs_freq)
        montage = mne.channels.make_standard_montage('standard_1020')
        
        raw.set_montage(montage)
        raw.set_eeg_reference('average', projection=True)

        X_featured = self.CREATE_EPOCHS(raw)
        X_featured = np.array(X_featured)
        print('shape',np.shape(X_featured))
        X_featured = X_featured.reshape(-1)
        X_featured = X_featured.reshape(1, -1)

        return X_featured
    def initialize_model(self, training_file,Inlet_info,ch_names):
        self.info = create_mne_info_from_lsl(Inlet_info,ch_names)
        self.inverse_operator = Inverse_calculator(self.info)
        if type(training_file) == type([]):
            if '.set' in training_file[0]:# We verify if training file is a .set files
                X_all_runs , y_all_runs = pre_process_run_game(training_file,self.info,self.inverse_operator)
                self.mean_init = np.mean(X_all_runs,axis = 0)
                self.std_init = np.std(X_all_runs,axis = 0)
                X_z_score  = (X_all_runs- self.mean_init)/self.std_init# estimation for unit variance and meaned data
                self.model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
                # Apprendre de manière incrémentale sur les mini-lots de données
                batch_size = 10
                for i in range(0, len(X_z_score), batch_size):
                    X_batch = X_z_score[i:i + batch_size]
                    y_batch = y_all_runs[i:i + batch_size]
                    self.model.partial_fit(X_batch, y_batch, classes=[False,True])
                print("Model trained succesfully")
                if self.Save_model:
                    Date = time.localtime()
                    dump(self.model,filename =(os.path.join(script_directory, 'Saved_models','init_'+str(Date[1])+'_'+str(Date[2])+'_'+str(Date[3])+':'+str(Date[4]) ) ) )
                    print("Model saved succesfully")
                if True:
                    pipeline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))
                    cv = StratifiedKFold(5, shuffle=True)
                    scores = cross_val_score(pipeline, X_all_runs,y_all_runs, cv = cv)
                    print(f"Cross validation scores: {scores}")
                    print(f"Mean: {scores.mean()}")
            if 'Data_games' in training_file[0]:
                self.model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
                for training_set in training_file:
                    [X_z_score , y_all_runs, Table_score] = [None , None , None]
                    with open(training_set, 'rb') as file:
                        # Deserialize and retrieve the variable from the file
                        [X_z_score , y_all_runs, Table_score] = pickle.load(file)
                    self.model.partial_fit(X_z_score, y_all_runs, classes=[False,True])
                print("Model trained succesfully")
        if type(training_file) == type(''):
            self.model = load(training_file)
            print("Model loaded succesfully")
        if training_file == 1:
            training_file_find = find_model(os.path.join(script_directory,'Saved_models'), 'TRAINING_SET')
            if training_file_find==None:
                training_file = 0
            else:
                self.model = load(training_file_find)
                print("Model loaded succesfully")
        if training_file == 0:
            self.model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
            print('No model or data foud, let s create it!')
            self.TRAINING_MODE = 1
    def data_to_p(self):
        self.X_featured = self.pre_process_run_update()
        if self.TRAINING_MODE==0:
            self.X_z_score  = (self.X_featured- self.mean_init)/self.std_init# estimation for unit variance and meaned data
            self.p = self.model.predict(self.X_z_score)
        return self.p
    def get_random_apple_position(self):
        Choice_apple = self.APPLE_ORDER[self.score+self.failures]
        falling_pos = [SCREEN_WIDTH // 4 - APPLE_SIZE // 2, 3 * SCREEN_WIDTH // 4 - APPLE_SIZE // 2]
        return falling_pos[Choice_apple]
    def update_model(self,y):
        # Apprendre de manière incrémentale sur les mini-lots de données
        self.model.partial_fit(self.X_z_score, y, classes=[False , True ])
        return True    
    def draw_player(self):
        if self.hand_status["left"] == "open":
            self.screen.blit(self.left_hand_open, (self.player_pos[0] - PLAYER_WIDTH, self.player_pos[1]))
        else:
            self.screen.blit(self.left_hand_closed, (self.player_pos[0] - PLAYER_WIDTH, self.player_pos[1]))

        if self.hand_status["right"] == "open":
            self.screen.blit(self.right_hand_open, (self.player_pos[0], self.player_pos[1]))
        else:
            self.screen.blit(self.right_hand_closed, (self.player_pos[0], self.player_pos[1]))

    def draw_apple(self):
        self.screen.blit(self.apple_image, (self.apple_pos[0], self.apple_pos[1]))

    def draw_background(self):
        self.screen.blit(self.tree_image, (0, 0))

    def update_apple(self):
        self.apple_pos[1] += self.apple_speed
        if self.apple_pos[1] >= SCREEN_HEIGHT:
            self.apple_pos = [self.get_random_apple_position(), 0]
            self.failures += 1
            self.TABLE_score += [self.score/(self.score + self.failures)]
            if not(self.TRAINING_MODE):
                self.update_model([self.apple_pos[0] >= SCREEN_WIDTH // 2])
            
            self.stock_features(self.apple_pos[0] >= SCREEN_WIDTH // 2)
            self.input_processed= True
            self.start_time = time.time()
            self.marker_sent = False
            self.hand_status["left"] = "closed"
            self.hand_status["right"] = "closed"

    def handle_input(self):
        if self.input_processed == False:
            if self.TRAINING_MODE:
                self.prob = int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 )
                self.input_processed = True
            if self.TRAINING_MODE==2:
                self.prob = random.choice([int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 ),random.random(),int(self.apple_pos[0]/(SCREEN_WIDTH // 2) + 0.5 )])
                self.input_processed = True
            if self.TRAINING_MODE ==  False :
                self.prob = self.p 
        if self.prob < 0.5 and self.marker_finished:  # Open left hand
            self.hand_status["left"] = "open"
            self.hand_status["right"] = "closed"
            self.check_catch("left")
        if self.prob >= 0.5 and self.marker_finished:  # Open right hand
            self.hand_status["right"] = "open"
            self.hand_status["left"] = "closed"
            self.check_catch("right")

    def check_catch(self, hand):
        if hand == "left" and self.apple_pos[0] < SCREEN_WIDTH // 2:
            if self.apple_pos[1]  >= self.player_pos[1]:
                self.score += 1
                self.TABLE_score += [self.score/(self.score + self.failures)]
                if not(self.TRAINING_MODE):
                    self.update_model([0])
                self.stock_features(0)
                self.hand_status["left"] = "closed"
                self.apple_pos = [self.get_random_apple_position(), 0]
                self.start_time = time.time()
                self.marker_sent = False

        elif hand == "right" and self.apple_pos[0] >= SCREEN_WIDTH // 2:
            if self.apple_pos[1]  >= self.player_pos[1]:
                self.score += 1
                self.TABLE_score += [self.score/(self.score + self.failures)]
                if not(self.TRAINING_MODE):
                    self.update_model([1])

                self.stock_features(1)
                self.hand_status["right"] = "closed"
                self.apple_pos = [self.get_random_apple_position(), 0]
                self.start_time = time.time()
                self.marker_sent = False
    def draw_scoreboard(self):
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        failures_text = self.font.render(f'Failures: {self.failures}', True, (255, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(failures_text, (10, 50))

    def draw_load_bar(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.ldt:
            elapsed_time = self.ldt  # Cap the elapsed time at load_time_seconds

        load_ratio = elapsed_time / self.ldt
        load_bar_width = SCREEN_WIDTH * load_ratio
        load_bar_rect = pygame.Rect(0, SCREEN_HEIGHT, load_bar_width, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, LOAD_BAR_COLOR, load_bar_rect)

        marker_position = SCREEN_WIDTH * self.ldtb /self.ldt
        marker_position_end = SCREEN_WIDTH * (self.ldtb + self.ldtm)/self.ldt
        marker_rect = pygame.Rect(marker_position, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, marker_rect)
        marker_rect_end = pygame.Rect(marker_position_end, SCREEN_HEIGHT, 2, LOAD_BAR_HEIGHT)
        pygame.draw.rect(self.screen, MARKER_BAR_COLOR, marker_rect_end)
        if elapsed_time < self.ldtb  and not(self.marker_sent):
            self.marker_not_finished = True
            self.marker_finished = False
        if elapsed_time >= self.ldtb  and not(self.marker_sent):
            if self.apple_pos[0] < SCREEN_WIDTH // 2:
                outlet.push_sample(["left"])
            else:
                outlet.push_sample(["right"])
            self.marker_sent = True
        if elapsed_time >= self.ldtb + self.ldtm  and self.marker_not_finished:
            self.input_processed= False
            self.marker_not_finished = False
            self.sample,self.timestamp = inlet.pull_chunk(timeout=self.time_of_window)
            self.data_to_p()
        if elapsed_time >= self.ldtb + self.ldtm :
            self.marker_finished = True

    def show_menu(self):
        menu_font = pygame.font.SysFont(None, 36)
        title_font = pygame.font.SysFont(None, 54)
        title = title_font.render('Apple Catcher Game', True, (0, 0, 0))
        start_text = menu_font.render('Start Game', True, (0, 0, 0))
        quit_text = menu_font.render('Quit', True, (0, 0, 0))
        
        title_rect = title.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        start_rect = start_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 100))
        quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 150))

        mode_font = pygame.font.SysFont(None, 24)
        mode_0 = mode_font.render('Test', True, (0,0,0))
        mode_1 = mode_font.render('Training', True, (0,0,0))
        mode_2 = mode_font.render('Define', True, (0,0,0))
        mode_0_rect = mode_0.get_rect(center=(SCREEN_WIDTH/4,SCREEN_HEIGHT-100))
        mode_1_rect = mode_1.get_rect(center=(SCREEN_WIDTH/2,SCREEN_HEIGHT-100))
        mode_2_rect = mode_2.get_rect(center=((SCREEN_WIDTH*3)/4,SCREEN_HEIGHT-100))
        
        # Variables
        COULEUR_CHAMP_TEXTE = (200, 200, 200)
        COULEUR_CHAINE = (255, 255, 255)
        COULEUR_BORDURE = (0, 0, 0)
        # Police pour le texte
        imput_font = pygame.font.Font(None, 24)
        chaine_texte = ''
        actif = False
        # Rectangle du champ de texte
        input_rect = pygame.Rect( 275, SCREEN_HEIGHT-70, 50, 32)

        end_value_txt_font = pygame.font.SysFont(None, 24)
        end_value_txt = end_value_txt_font.render('End_Value', True, (0,0,0))
        end_value_txt_rect = end_value_txt.get_rect(center=(((SCREEN_WIDTH*2)/4,SCREEN_HEIGHT-55)))

        while True:
            self.screen.fill(BACKGROUND_COLOR)
            self.draw_background()
            self.screen.blit(title, title_rect)
            self.screen.blit(start_text, start_rect)
            self.screen.blit(quit_text, quit_rect)

            self.screen.blit(mode_0, mode_0_rect)
            self.screen.blit(mode_1, mode_1_rect)
            self.screen.blit(mode_2, mode_2_rect)

            self.screen.blit(end_value_txt, end_value_txt_rect)

            # Dessiner le champ de texte
            pygame.draw.rect(self.screen, COULEUR_CHAMP_TEXTE, input_rect)
            pygame.draw.rect(self.screen, COULEUR_BORDURE, input_rect, 2)
            # Texte à afficher dans le champ de texte
            txt_surface = imput_font.render(chaine_texte, True, COULEUR_CHAINE)
            # Dessiner le texte
            self.screen.blit(txt_surface, (input_rect.x + 5, input_rect.y + 10))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if start_rect.collidepoint(mouse_pos):
                        return 'start'
                    elif quit_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()
                    if mode_0_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE=False
                        mode_0 = mode_font.render('Test', True, (255,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))
                    if mode_1_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE = True
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (255,0,0))
                        mode_2 = mode_font.render('Define', True, (0,0,0))                       
                    if mode_2_rect.collidepoint(mouse_pos):
                        self.TRAINING_MODE=2
                        mode_0 = mode_font.render('Test', True, (0,0,0))
                        mode_1 = mode_font.render('Training', True, (0,0,0))
                        mode_2 = mode_font.render('Define', True, (255,0,0))

                    if input_rect.collidepoint(event.pos):
                        actif = not actif
                    else:
                        actif = False

                elif event.type == pygame.KEYDOWN:
                    if actif:
                        if event.key == pygame.K_RETURN:
                            print(chaine_texte)  # Affiche la chaîne entrée dans la console
                            self.end_value=int(chaine_texte)
                            chaine_texte = ''
                        elif event.key == pygame.K_BACKSPACE:
                            chaine_texte = chaine_texte[:-1]
                        else:
                            if event.unicode.isdigit():  # Assure que seuls les chiffres sont entrés
                                chaine_texte += event.unicode


            pygame.display.flip()
            pygame.time.Clock().tick(FPS)

    def run(self):
        running = True
        self.APPLE_ORDER = [i%2 for i in range(self.end_value)]
        random.shuffle(self.APPLE_ORDER) 
        self.APPLE_ORDER.append(-1) 
        self.apple_pos = [self.get_random_apple_position(), 0]
        while running:
            T= time.time()
            if (self.score + self.failures) >= self.end_value:
                outlet.push_sample(['STOP'])
                Date = time.localtime()
                if self.TRAINING_MODE:
                    self.mean_init = np.mean(self.X_F_stock,axis = 0)
                    self.std_init = np.std(self.X_F_stock,axis = 0)
                    self.X_z_score = (self.X_F_stock- self.mean_init)/self.std_init
                    self.update_model(self.Y_stock)
                    Score =0
                    if False:
                        pipeline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))
                        cv = StratifiedKFold(5, shuffle=True)
                        scores = cross_val_score(pipeline, self.X_F_stock,self.Y_stock, cv = cv)
                        print(f"Cross validation scores: {scores}")
                        print(f"Mean results for the training set: {scores.mean()}")
                        Score = scores.mean()

                    file_path_data = (os.path.join(script_directory,'Data_games', 'TRAINING_data'+'_'+str(int(1000*Score)/1000).replace('.', '_')+'_'+str(Date[1])+'_'+str(Date[2])+'_'+str(Date[3])+':'+str(Date[4])) )
                    file_path_model =(os.path.join(script_directory,'Saved_models', 'TRAINING_SET'+str(int(1000*Score)/1000).replace('.', '_')+'_'+str(Date[1])+'_'+str(Date[2])+'_'+str(Date[3])+':'+str(Date[4])) ) 

                else:
                    score_rounded =int(1000*self.score/(self.score + self.failures))/1000
                    file_path_data = (os.path.join(script_directory,'Data_games', 'Testing_data'+str(score_rounded).replace('.', '_')+'_'+str(Date[1])+'_'+str(Date[2])+'_'+str(Date[3])+':'+str(Date[4])) )
                    file_path_model  = (os.path.join(script_directory,'Saved_models', 'finished_score_'+str(score_rounded)+str(Date[1]).replace('.', '_')+'_'+str(Date[2])+'_'+str(Date[3])+':'+str(Date[4]) ) ) 

                # FILE_TO_SAVE=[file_path_data,file_path_model]
                np.savez(file_path_data,np.array(self.X_F_stock.copy()),np.array(self.Y_stock.copy()),np.array(self.TABLE_score.copy()))
                
                with open(file_path_model, 'x') as file:
                    dump(self.model, filename = file_path_model)
                running = False
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill(BACKGROUND_COLOR)
            self.draw_background()
            self.draw_scoreboard()
            self.draw_load_bar()
            self.handle_input()
            self.update_apple()
            self.draw_player()
            self.draw_apple()
        
            pygame.display.flip()

            t = time.time()
            while t-T < 1/FPS:
                t = time.time()


        pygame.quit()
        return None




if __name__ == "__main__":
    print(training_file)
    game = Game(PLAYER_HEIGHT,PLAYER_WIDTH,SCREEN_WIDTH,LOAD_BAR_HEIGHT,SCREEN_HEIGHT,APPLE_SIZE,TRAINING_MODE,load_time_seconds_before,load_time_seconds_marker,load_time_seconds,training_file,Save_model,Inlet_info)

    if game.show_menu()=='start':
        game.start_time=time.time()
        game.run()
print('done')
