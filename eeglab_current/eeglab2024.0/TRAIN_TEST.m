function Merging_sets(file)
chdir(file)
LIST = ls('*.set')
EEG_tot=[];
for ind = 1:size(LIST,1)
    name=LIST(ind,1:end)
    [EEG1] = pop_loadset(name, file);
    if isempty(EEG_tot)
        EEG_tot = EEG1
    else
        EEG_tot = pop_mergeset(EEG_tot , EEG1)
    end
end

pop_saveset( EEG_tot, 'mergeddata', file)
eeglab redraw;
return EEG_tot