%% STAGE 26 - PAIRED DAY-BLOCK BOOTSTRAP (DESCRIPTIVE AUDIT)
% Flight delay manuscript revision, January 2025.
%
% GOAL: Compare the numerically corrected Robust Linear model (Stage 25)
% against FROZEN comparator predictions on precisely the same 131,891
% holdout flights. NO TRAINING, no tuning, no changes to the 23-preset table.
%
% IMPORTANT: The Jan 25-31 holdout has already been inspected during
% model development. This is a descriptive uncertainty/sensitivity audit,
% NOT an untouched/blind confirmatory model-selection experiment.
% Seven day clusters are too few for definitive statistical inference.
%
% Required files, all in one folder:
%   Stage25_RobustLinear_QR_Audit.mat
%   Stage23_23Presets_Checkpoint.mat
%   FlightDelay_Stage16_FinalTest.mat
%
% Outputs:
%   Stage26_PairedBootstrap_ModelCI.csv
%   Stage26_PairedBootstrap_Differences.csv
%   Stage26_DailyModelMetrics.csv
%   Stage26_PairedBootstrap.mat
%
% Standard MATLAB (no new toolboxes beyond those used earlier).

clear; clc; format compact;
rng(42,'twister');

%% 1. Load completed prediction files (NOT Excel)
[fileName,folderPath] = uigetfile('Stage25_RobustLinear_QR_Audit.mat', ...
    'Select Stage25_RobustLinear_QR_Audit.mat');
if isequal(fileName,0), error('No file selected.'); end

S25 = load(fullfile(folderPath,fileName),'P6','P12');
S23 = load(fullfile(folderPath,'Stage23_23Presets_Checkpoint.mat'), ...
    'P','modelNames','yVal');
S16 = load(fullfile(folderPath,'FlightDelay_Stage16_FinalTest.mat'), ...
    'P','modelNames','yTest','testDays');

y = double(S23.yVal(:));
assert(numel(y)==131891, 'Unexpected holdout sample length.');
assert(isequal(y,double(S16.yTest(:))), ...
    'Stage 16 and Stage 23 target/order mismatch: STOP.');

day = double(S16.testDays(:));
assert(numel(day)==numel(y), 'Day vector length mismatch.');
days = unique(day);
assert(isequal(days(:),(25:31)'), ...
    'Expected seven days Jan 25-31: STOP.');

robust = double(S25.P6{2}(:));
robust12 = double(S25.P12{2}(:));
assert(numel(robust)==numel(y) && all(isfinite(robust)), ...
    'Stage25 corrected predictions invalid.');

name23 = string(S23.modelNames(:));
name16 = string(S16.modelNames(:));
get23 = @(s) find(name23==s,1,'first');
get16 = @(s) find(name16==s,1,'first');

originals = ["Linear"; "Bagged Trees"; "Boosted Trees"];
references = ["Linear Regression"; "Residual Bagged Trees"; "Residual LSBoost"];
for j=1:numel(originals)
    assert(~isempty(get23(originals(j))), ...
        'Stage23 model missing: %s',char(originals(j)));
end
for j=1:numel(references)
    assert(~isempty(get16(references(j))), ...
        'Stage16 model missing: %s',char(references(j)));
end

modelNames = ["Robust Linear (6-pass QR)"; ...
              "Linear (original preset)"; ...
              "Bagged Trees (original preset)"; ...
              "Boosted Trees (original preset)"; ...
              "Linear Regression (Stage16)"; ...
              "Residual Bagged Trees (Stage16)"; ...
              "Residual LSBoost (Stage16)"];

P = [robust, ...
     double(S23.P(:,get23("Linear"))), ...
     double(S23.P(:,get23("Bagged Trees"))), ...
     double(S23.P(:,get23("Boosted Trees"))), ...
     double(S16.P(:,get16("Linear Regression"))), ...
     double(S16.P(:,get16("Residual Bagged Trees"))), ...
     double(S16.P(:,get16("Residual LSBoost")))];

assert(all(size(P)==[numel(y),numel(modelNames)]), ...
    'Prediction matrix shape incorrect.');
assert(all(isfinite(P(:))), 'Nonfinite predictions found.');

fprintf('\n===== STAGE 26: MATCHED PREDICTIONS =====\n');
fprintf('Holdout flights: %d\n',numel(y));
fprintf('Days: %s\n',mat2str(days'));
fprintf('Compared prediction columns: %d\n',numel(modelNames));
fprintf('Max 6-to-12-pass prediction change (diagnostic): %.4f min\n', ...
    max(abs(robust12-robust)));

%% 2. Compute day-level error sums (same blocks for every model)
K = numel(days);
M = numel(modelNames);
N = zeros(K,1);
SSE = zeros(K,M);
SAE = zeros(K,M);
DAILY_RMSE = zeros(K,M);
DAILY_MAE = zeros(K,M);
DailyMetrics = table();

for k=1:K
    ix = day==days(k);
    N(k) = sum(ix);
    err = y(ix)-P(ix,:);
    SSE(k,:) = sum(err.^2,1);
    SAE(k,:) = sum(abs(err),1);
    DAILY_RMSE(k,:) = sqrt(SSE(k,:)/N(k));
    DAILY_MAE(k,:) = SAE(k,:)/N(k);
    for j=1:M
        row = table(days(k),N(k),modelNames(j), ...
            DAILY_RMSE(k,j),DAILY_MAE(k,j), ...
            'VariableNames',{'Day','N','Model','RMSE_min','MAE_min'});
        DailyMetrics = [DailyMetrics;row]; %#ok<AGROW>
    end
end

assert(sum(N)==numel(y),'Day counts do not sum to total holdout.');

rmse = sqrt(sum(SSE,1)/sum(N));
mae = sum(SAE,1)/sum(N);
fullSST = sum((y-mean(y)).^2);
r2 = 1-sum(SSE,1)/fullSST;

%% 3. Day-cluster paired bootstrap: exploratory only
B = 5000;
bootRMSE = zeros(B,M);
bootMAE = zeros(B,M);
for b=1:B
    chosen = randi(K,K,1); % sample DAYS, not flights
    nb = sum(N(chosen));
    bootRMSE(b,:) = sqrt(sum(SSE(chosen,:),1)/nb);
    bootMAE(b,:) = sum(SAE(chosen,:),1)/nb;
end

ciRMSE = prctile(bootRMSE,[2.5 97.5],1);
ciMAE = prctile(bootMAE,[2.5 97.5],1);

ModelCI = table(modelNames,rmse',ciRMSE(1,:)',ciRMSE(2,:)', ...
    mae',ciMAE(1,:)',ciMAE(2,:)',r2', ...
    'VariableNames',{'Model','RMSE_min','RMSE_CI_Low','RMSE_CI_High', ...
                     'MAE_min','MAE_CI_Low','MAE_CI_High','R2'});

%% 4. Paired differences vs corrected Robust Linear (first column)
% Difference = competitor error - Robust Linear error.
% Positive => Robust Linear has LOWER error.
% Negative => competitor has LOWER error.

compNames = modelNames(2:end);
nComp = numel(compNames);
DeltaRMSE = zeros(nComp,1);
DeltaMAE = zeros(nComp,1);
RMSE_Low = zeros(nComp,1);
RMSE_High = zeros(nComp,1);
MAE_Low = zeros(nComp,1);
MAE_High = zeros(nComp,1);
DaysRobustBetter_RMSE = zeros(nComp,1);
DaysRobustBetter_MAE = zeros(nComp,1);
RMSE_CI_ExcludesZero = false(nComp,1);
MAE_CI_ExcludesZero = false(nComp,1);

for j=1:nComp
    c=j+1;
    DeltaRMSE(j)=rmse(c)-rmse(1);
    DeltaMAE(j)=mae(c)-mae(1);
    diffRMSE=bootRMSE(:,c)-bootRMSE(:,1);
    diffMAE=bootMAE(:,c)-bootMAE(:,1);
    rCI=prctile(diffRMSE,[2.5 97.5]);
    aCI=prctile(diffMAE,[2.5 97.5]);
    RMSE_Low(j)=rCI(1); RMSE_High(j)=rCI(2);
    MAE_Low(j)=aCI(1); MAE_High(j)=aCI(2);
    DaysRobustBetter_RMSE(j)=sum(DAILY_RMSE(:,1)<DAILY_RMSE(:,c));
    DaysRobustBetter_MAE(j)=sum(DAILY_MAE(:,1)<DAILY_MAE(:,c));
    RMSE_CI_ExcludesZero(j)=(rCI(1)>0 || rCI(2)<0);
    MAE_CI_ExcludesZero(j)=(aCI(1)>0 || aCI(2)<0);
end

PairedDifferences = table(compNames,DeltaRMSE,RMSE_Low,RMSE_High, ...
    RMSE_CI_ExcludesZero,DeltaMAE,MAE_Low,MAE_High, ...
    MAE_CI_ExcludesZero,DaysRobustBetter_RMSE,DaysRobustBetter_MAE, ...
    'VariableNames',{'ComparedModel','DeltaRMSE_min','RMSE_CI_Low', ...
    'RMSE_CI_High','RMSE_CI_ExcludesZero','DeltaMAE_min', ...
    'MAE_CI_Low','MAE_CI_High','MAE_CI_ExcludesZero', ...
    'DaysRobustBetter_RMSE','DaysRobustBetter_MAE'});

%% 5. Present and save outputs (never overwrite prior stages)
fprintf('\n===== STAGE 26 MODEL METRICS AND APPROXIMATE CIs =====\n');
disp(ModelCI);
fprintf('\n===== STAGE 26 PAIRED DIFFERENCES VS ROBUST LINEAR =====\n');
fprintf('Positive delta = corrected Robust Linear has lower error.\n');
disp(PairedDifferences);

fprintf('\nCAUTIONS:\n');
fprintf('1. Seven calendar-day clusters only; bootstrap CIs are exploratory.\n');
fprintf('2. Holdout has been previously inspected; do not assert blind selection.\n');
fprintf('3. This script DOES NOT train or tune any model.\n');

writetable(ModelCI,fullfile(folderPath, ...
    'Stage26_PairedBootstrap_ModelCI.csv'));
writetable(PairedDifferences,fullfile(folderPath, ...
    'Stage26_PairedBootstrap_Differences.csv'));
writetable(DailyMetrics,fullfile(folderPath, ...
    'Stage26_DailyModelMetrics.csv'));
save(fullfile(folderPath,'Stage26_PairedBootstrap.mat'), ...
    'ModelCI','PairedDifferences','DailyMetrics','days','N', ...
    'rmse','mae','r2','bootRMSE','bootMAE','B','-v7.3');

fprintf('\nSTAGE 26 COMPLETED: files saved to %s\n',folderPath);
