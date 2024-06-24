function RUN_FILE_VOID(fil)
file=strcat(fil,'\');
xdftoset_FILE(file);
EEG_tot = Merging_sets(file);
xdftoset_FILE(strcat(file,'VOID_FILE','\'));
[EEG_VOID] = pop_loadset('VOID_DATA.set', strcat(file,'VOID_FILE'));
EEG_VOID.event = []
shuffle_data_train_test(EEG_tot,file,EEG_VOID);
end