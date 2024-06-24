function save_eeglab_set(file_data, file)
    % Create an EEGLAB EEG structure
    % C:\Users\robinaki\Downloads\Dataset\sub-P001\ses-S001\eeg
    your_data_struct_s = open(file_data)
    your_data_struct = your_data_struct_s.EEG
    EEG = struct();

    % Assign the required fields
    EEG.data = zeros(32,your_data_struct.pnts);
    DATA= your_data_struct.data;
    DATA= DATA(1:end-1,:);
    EEG.data = DATA;
    EEG.times = your_data_struct.times;
    EEG.setname = your_data_struct.setname;
    EEG.nbchan = your_data_struct.nbchan;
    EEG.pnts = your_data_struct.pnts;
    EEG.trials = your_data_struct.trials;
    EEG.srate = your_data_struct.srate;
    EEG.xmin = your_data_struct.xmin;
    EEG.xmax = your_data_struct.xmax;
    CHANNELS = your_data_struct.chanlocs; 
    EEG.chanlocs.labels = zeros(32)
    n_chan =  length(your_data_struct.chanlocs) -1
    for i = 1:n_chan
        CHAN = CHANNELS(i)
        lab = CHAN{1,1}.labels;
        EEG.chanlocs(i).labels = lab; 
    end
    % Initialize ICA fieldsZ
    EEG.icaweights = [];
    EEG.icasphere = [];
    EEG.icawinv = [];
    
    % Process event information
    events = your_data_struct.event;
    n_events = length(events);
    EEG.event = struct('latency', cell(1, n_events), 'type', cell(1, n_events));

    for i = 1:n_events
        EV = events(i)
        lat = EV{1,1}.latency;
        type= EV{1,1}.type;
        EEG.event(i).latency = lat;
        EEG.event(i).type = type;
    end

    % Save the structure as a .set file
    EEG.filename='test.set';
    EEG_ = pop_newset(EEG);
    [EEG_] = pop_chanedit(EEG_, 'load',{'C:\Users\robinaki\Downloads\ThesisRobin\ThesisRobin\Matlab Code\8Chann.loc' 'filetype' 'autodetect'});

    pop_saveset(EEG, 'TEST_data_set', file);
end
