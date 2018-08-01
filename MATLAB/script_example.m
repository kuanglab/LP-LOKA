clear

%% loading data
load('test/S');
load('test/F0');
[H,~] = fastaread('test/100_seqs.fa');

%% call nystrom
F = LPLOKA_Nystrom(S);

%% call label propagation
Fhat = LPLOKA(F, F0, 'Alpha', 0.1);

%% get results
SeqIds = LPLOKA_GetRankedSequenceID(Fhat, H);