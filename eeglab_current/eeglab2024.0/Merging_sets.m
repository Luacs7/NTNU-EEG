function EEG_tot = Merging_sets(file)
chdir(file);
LIST_set= ls('*.set');
OUTPUT_set=extractFilenamesCharArray(LIST_set);
EEG_tot=[];
for ind = 1:size(OUTPUT_set,1)
    name=strcat(OUTPUT_set(ind),'.set')
    [EEG1] = pop_loadset(name, file);
    if isempty(EEG_tot)
        EEG_tot = EEG1
    else
        EEG_tot = pop_mergeset(EEG_tot , EEG1)
    end
end

pop_saveset( EEG_tot, 'mergeddata', file)
eeglab redraw;
end