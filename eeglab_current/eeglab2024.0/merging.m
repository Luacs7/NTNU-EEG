function merging(filename1,filename2,name1,name2)
[EEG1] = pop_loadset(strcat( name1,'.set'), filename1)
[EEG2] = pop_loadset(strcat( name2,'.set'), filename2)
EEG = pop_mergeset( EEG1, EEG2);
save(strcat(filename1, strcat("merge",name1,name2),'.set'), 'EEG', '-mat')
eeglab redraw;
end