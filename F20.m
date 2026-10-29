%% STAGE 00: SIMPLE CHRONOLOGICAL REFERENCE PREDICTORS
% January 2025 BTS arrival delay (minutes), post-departure setting.
% Reconstruct training-set mean, training-set median, and direct DEP_DELAY
% reference models on the authors' Jan 25-31 supplementary test period.
% No ML fitting, no optimization, no target-derived delay attribution.
% This standalone script has no earlier checkpoint dependency.
%
% The test interval has already been inspected in the research workflow;
% these results are descriptive and not a blind external validation.
% MATLAB R2024b+; source Excel must match the exact study snapshot.

clear; clc; format compact;
[fileName,folderPath] = uigetfile({'*.xlsx;*.xls','Excel Files'}, ...
    'Choose TranstatsData_Ayse4.xlsx');
if isequal(fileName,0), error('No Excel file selected.'); end
sourcePath = fullfile(folderPath,fileName);

T = readtable(sourcePath,'Sheet','TranstatsData_Ayse');
required = {'CANCELLED','DIVERTED','ARR_DELAY','DEP_DELAY','DAY_OF_MONTH'};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'Source workbook missing required BTS fields.');
keep = T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY) & ...
    ~ismissing(T.DAY_OF_MONTH);
U = T(keep,{'ARR_DELAY','DEP_DELAY','DAY_OF_MONTH'});
assert(height(U)==578957, ...
    'Expected 578957 eligible records; verify frozen data extract.');

day = double(U.DAY_OF_MONTH);
y = double(U.ARR_DELAY);
depDelay = double(U.DEP_DELAY);
tr = day>=1 & day<=24;
te = day>=25 & day<=31;
assert(nnz(tr)==447066 && nnz(te)==131891, ...
    'Chronological sample counts differ from the study.');
yTrain = y(tr); yTest = y(te);

yhat = [repmat(mean(yTrain),numel(yTest),1), ...
        repmat(median(yTrain),numel(yTest),1), ...
        depDelay(te)];
name = ["Training mean";"Training median";"Observed departure delay"];
rmse = zeros(3,1);mae = zeros(3,1);mse = zeros(3,1);r2 = zeros(3,1);
for k=1:3
    err = yTest - yhat(:,k);
    mse(k) = mean(err.^2);
    rmse(k) = sqrt(mse(k));
    mae(k) = mean(abs(err));
    r2(k) = 1-sum(err.^2)/sum((yTest-mean(yTest)).^2);
end
Result = table(name,rmse,mae,mse,r2, ...
    'VariableNames',{'Reference','RMSE_min','MAE_min','MSE_min2','R2'});
disp(Result);
outDir = fullfile(folderPath,'Stage00_ReferenceModels');
if ~isfolder(outDir),mkdir(outDir);end
writetable(Result,fullfile(outDir,'Stage00_Baselines.csv'));
fprintf('Saved: %s\n',fullfile(outDir,'Stage00_Baselines.csv'));
