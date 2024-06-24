function xdftoset(filename_,output_char)
filename = strcat(filename_, output_char ,'.xdf');
EEG = pop_loadxdf( filename );
EEG.srate
% Starting EEGLAB
EEG_ = pop_newset(EEG);
[EEG_] = pop_chanedit(EEG_, 'load',{'C:\Users\robinaki\Downloads\ThesisRobin\ThesisRobin\Matlab Code\8Chann.loc' 'filetype' 'autodetect'});


% Replace with your desired storage location and filename
pop_saveset(EEG_, 'filename',strcat( output_char ,'.set'), 'filepath', char(filename_)); 

% Cleaning up the workspace
eeglab redraw
end
