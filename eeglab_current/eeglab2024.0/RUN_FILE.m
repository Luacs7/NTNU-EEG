function RUN_FILE(fil)
file=strcat(fil,'\');
xdftoset_FILE(file);
EEG_tot = Merging_sets(file);
% Prop_train = [0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
Prop_train=[0.6]
for j = 1:length(Prop_train)
    prop_train = Prop_train(j)
    shuffle_data_train_test(EEG_tot,file,[],prop_train);
end
end