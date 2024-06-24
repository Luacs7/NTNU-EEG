function save_eeglab_set2(file_data, file)
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
    EEG.nbchan = 32;
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
        EV = events(i);
        lat = EV{1,1}.latency;
        type= EV{1,1}.type;
        if strcmp(type,'Go_Right')
            type = 'right';
        end
        if strcmp(type,'Go_Left')
            type = 'left';
        end
        EEG.event(i).latency = lat;
        EEG.event(i).type = type;
    end

    % Save the structure as a .set file
    EEG.filename='test.set';
    EEG = pop_newset(EEG);
    [EEG] = pop_chanedit(EEG, 'load',{'C:\Users\robinaki\Downloads\eeglab_current\eeglab2024.0\plugins\dipfit\standard_BEM\elec\standard_1005.elc' 'filetype' 'autodetect'});
    pop_saveset(EEG, 'TEST_data_set', file);
end
