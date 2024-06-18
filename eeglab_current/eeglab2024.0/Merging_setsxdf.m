function EEG_tot = Merging_setsxdf(file)
chdir(file)
LIST = ls('*.xdf')
LIST_xdf = ls('*.xdf');
OUTPUT_xdf = extractFilenamesCharArray(LIST_xdf);
i=1
output = OUTPUT_xdf(i);
output
filename = strcat(char(output), '.xdf');
EEG_tot = pop_loadxdf(char(filename));
% 
% for ind = 1:size(LIST_xdf,1)
%     name=LIST(ind,1:end)
%     [EEG1] = pop_loadxdf(name, file);
%     if isempty(EEG_tot)
%         EEG_tot = EEG1
%     else
%         EEG_tot = pop_mergeset(EEG_tot , EEG1)
%     end
% end

eeglab redraw;
end