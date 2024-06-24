function shuffle_data_train_test(EEG,file,EEG_void,prop_train)
% prop_train=0.8;
N_events=size(EEG.event(1,:,1));
N_events=N_events(2);

N_events_train= int64(N_events*prop_train);
N_events_test = N_events - N_events_train

RANDOM_ORDER= randperm(N_events);
% RANDOM_ORDER = 1:N_events
NEWEVENTS=EEG.event(RANDOM_ORDER);
t_marker=20;%define the time between each marker in seconds
t_post_marker = 30;
T_tot_train= (t_post_marker+t_marker) * N_events_train
N_tot_train=int64(T_tot_train * EEG.srate) + 1;
%We create the structure of train and test datasets
NEWEEG_train=struct();
NEWEEG_train.event=[];
NEWEEG_train.setname="NEWEEG_train";
NEWEEG_train.data=zeros(EEG.nbchan,N_tot_train);
NEWEEG_train.icaweights=[];
NEWEEG_train.icawinv=[];
NEWEEG_train.times=(0:N_tot_train-1)*1000/EEG.srate;
NEWEEG_train.icasphere=[];
NEWEEG_train.nbchan=EEG.nbchan;

T_tot_test= (t_post_marker+t_marker) * N_events_test
N_tot_test= int64(T_tot_test * EEG.srate) + 1;
NEWEEG_test=struct();
NEWEEG_test=struct();
NEWEEG_test.event=[];
NEWEEG_test.setname="NEWEEG_test";
NEWEEG_test.data=zeros(EEG.nbchan,N_tot_test);
NEWEEG_test.icaweights=[];
NEWEEG_test.icawinv=[];
NEWEEG_test.times=(0:N_tot_test-1)*1000/EEG.srate;
NEWEEG_test.icasphere=[];
NEWEEG_test.nbchan=EEG.nbchan;
N_points=t_marker*EEG.srate;
N_points_post = t_post_marker*EEG.srate;

i1=1;
N_events_train
for j= 1:N_events_train
    event_=NEWEVENTS(j);
    if  strcmp(event_.type,'right') || strcmp(event_.type,'left')
        N_missing=0;
        N_missing_end = 0;
        i1
        Point_ind=1+(i1-1)*(N_points+ N_points_post);
        NEWEEG_train.pnts=length(NEWEEG_train.data);
        NEWEEG_train.event(i1).type= event_.type;
        NEWEEG_train.event(i1).latency= Point_ind + N_points_post;
        NEWEEG_train.event(i1).duration= 1;
        if event_.latency <=  N_points_post 
            N_missing = N_points_post- event_.latency+1
            NEWEEG_train.data(:,Point_ind:Point_ind + N_missing-1) = zeros(EEG.nbchan,N_missing)
        end
        if event_.latency+N_points -1 > EEG.pnts
            N_missing_end = event_.latency+N_points -1  -  EEG.pnts
            NEWEEG_train.data(:,Point_ind + N_points + N_points_post -N_missing_end :Point_ind + N_points + N_points_post -1) = zeros(EEG.nbchan,N_missing_end)
        end
        N_missing
        N_missing_end
        NEWEEG_train.data(:,Point_ind+N_missing:Point_ind + N_points + N_points_post -1-N_missing_end) = EEG.data(:,event_.latency +N_missing -  N_points_post:event_.latency+N_points-1-N_missing_end);
        %NEWEEG_train.data(:,Point_ind:Point_ind + N_points-1) = NEWEEG_train.data(:,Point_ind:Point_ind + N_points-1); - mean(NEWEEG_train.data(:,Point_ind:Point_ind + N_points-1),2)

        i1=i1+1;
    end 
    
end
i=1;
for j= 1:N_events_test
    event_=NEWEVENTS(j + N_events_train );
    if  strcmp(event_.type,'right') || strcmp(event_.type,'left')
        N_missing=0
        N_missing_end = 0;
        Point_ind=1+(i-1)*(N_points+ N_points_post);
        NEWEEG_test.pnts=length(NEWEEG_test.data);

        NEWEEG_test.event(i).type= event_.type;
        NEWEEG_test.event(i).latency= Point_ind+ N_points_post;
        NEWEEG_test.event(i).duration= 1;
        if event_.latency <=  N_points_post 
            N_missing = N_points_post- event_.latency+1
            NEWEEG_test.data(:,Point_ind:Point_ind + N_missing-1) = zeros(EEG.nbchan,N_missing)
        end
        if event_.latency+N_points -1 > EEG.pnts
            N_missing_end = event_.latency+N_points -1  - int64(EEG.pnts)
            NEWEEG_test.data(:,Point_ind + N_points + N_points_post -N_missing_end :Point_ind + N_points + N_points_post -1) = zeros(EEG.nbchan,N_missing_end)

        end
        N_missing
        N_missing_end
        NEWEEG_test.data(:,Point_ind+N_missing:Point_ind + N_points + N_points_post -1-N_missing_end) = EEG.data(:,event_.latency +N_missing -  N_points_post:event_.latency+N_points-1-N_missing_end);
   
        i=i+1;
    end 
    
end

%NEWEEG_train.event(i1).type = 'calib-end'
%NEWEEG_train.event(i1).latency = N_events_train*N_points -2
%NEWEEG_train.event(i1).duration=1


NEWEEG_train.pnts=length(NEWEEG_train.data);
NEWEEG_train.trials=EEG.trials;
NEWEEG_train.xmin=0;
NEWEEG_train.xmax=NEWEEG_train.times(:,end);
NEWEEG_train.srate=EEG.srate;
NEWEEG_train.chanlocs=EEG.chanlocs;
NEWEEG_train.filename='train.set';
NEWEEG_train=pop_newset(NEWEEG_train);
if not(isempty(EEG_void))
    NEWEEG_train = pop_mergeset(EEG_void , NEWEEG_train)
end
filename_train = char(strcat('train_data',string(prop_train*100)));
pop_saveset( NEWEEG_train, filename_train, file);
NEWEEG_test.pnts=length(NEWEEG_test.data);
NEWEEG_test.trials=EEG.trials;
NEWEEG_test.xmin=0;
NEWEEG_test.xmax=NEWEEG_test.times(:,end);
NEWEEG_test.srate=EEG.srate;
NEWEEG_test.chanlocs=EEG.chanlocs;
NEWEEG_test.filename='test.set';
NEWEEG_test=pop_newset(NEWEEG_test);
filename_test = char(strcat('test_data',string(prop_train*100)));
pop_saveset( NEWEEG_test,filename_test, file);

end