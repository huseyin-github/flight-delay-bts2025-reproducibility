%% STAGE 21 - ORIGINAL TABLE 9 TOP PRESETS: 5-FOLD TEMPORAL CV
% Frontiers in Built Environment - revision
% Candidate models were shortlisted using Jan 20-24 validation (Stage 20).
% This is DEVELOPMENT cross-validation, not a new independent test.
% No records from Jan 25-31 are used in fitting, scoring, or selection.
%
% Compared original presets: Bagged Trees, Linear, Boosted Trees, Coarse Tree
% Earlier baselines (Stage 11) and residual models (Stages 11,13) are
% loaded from OOF files for IDENTICAL-FLIGHT comparisons if available.
%
% All four selected original presets train on ALL available fold TRAIN rows.
% No target-derived delay attribution variables are used.
% TARGET = ARR_DELAY (minutes). Prediction happens AFTER departure.
%
% Script checkpoints after EACH MODEL; rerun with same Excel file to resume.
% Requires MATLAB Statistics and Machine Learning Toolbox.

clear; clc; format compact;
rng(42,'twister');

%% 1. INPUT
[fileName,folderPath] = uigetfile({'*.xlsx;*.xls','Excel Files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fileName,0), error('Excel file was not selected.'); end
sourcePath = fullfile(folderPath,fileName);
fprintf('\nReading data: %s\n',sourcePath);
T = readtable(sourcePath,'Sheet','TranstatsData_Ayse');

%% 2. SAMPLE - EXACTLY SAME AS PRIOR REVISED ANALYSES
keep = T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
vars = {'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_DEP_TIME', ...
    'CRS_ARR_TIME','CRS_ELAPSED_TIME','DISTANCE', ...
    'DEP_DELAY','ARR_DELAY'};
M = rmmissing(T(keep,vars));
assert(height(M)==578957,'Unexpected sample count.');
clear T;

allDay = double(M.DAY_OF_MONTH);
yAll = double(M.ARR_DELAY);
depAll = double(M.DEP_DELAY);

X = M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
    'DISTANCE','DEP_DELAY'});

depHHMM = double(M.CRS_DEP_TIME);
arrHHMM = double(M.CRS_ARR_TIME);
depMin = floor(depHHMM/100)*60+mod(depHHMM,100);
arrMin = floor(arrHHMM/100)*60+mod(arrHHMM,100);
assert(all(depMin>=0 & depMin<1440) && all(arrMin>=0 & arrMin<1440), ...
    'Invalid HHMM in input data.');
X.DepTimeSin = sin(2*pi*depMin/1440);
X.DepTimeCos = cos(2*pi*depMin/1440);
X.ArrTimeSin = sin(2*pi*arrMin/1440);
X.ArrTimeCos = cos(2*pi*arrMin/1440);

catVars = {'DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT','DEST_AIRPORT'};
for k=1:numel(catVars)
    v = catVars{k};
    % Only the development period defines permissible category labels.
    raw = string(X.(v));
    labels = unique(raw(allDay<=24));
    X.(v) = categorical(raw,labels);
end

trainEnd = [9 12 15 18 21];
valStart = [10 13 16 19 22];
valEnd   = [12 15 18 21 24];
K = numel(trainEnd);

modelNames = ["Bagged Trees";"Linear";"Boosted Trees";"Coarse Tree"];
J = numel(modelNames);
N = height(X);
OOF = nan(N,J);
State = repmat("NOT_RUN",K,J);
TrainSec = nan(K,J);
FitCount = nan(K,J);
LinearLSQRFlag = nan(K,J);
Errors = repmat("",K,J);

fprintf('\nTOTAL N=%d; development period: 1-24 January\n',N);
fprintf('Selected original Table 9 presets: %d\n',J);
fprintf('Validation flights, Jan 10-24: %d\n',sum(allDay>=10 & allDay<=24));

%% 3. CHECKPOINT / RESUME
ckPath = fullfile(folderPath,'Stage21_OriginalTopPresets_CV_Checkpoint.mat');
if isfile(ckPath)
    S = load(ckPath);
    assert(strcmp(string(S.sourcePath),string(sourcePath)), ...
        'Checkpoint corresponds to a different input path.');
    assert(isequal(string(S.modelNames),modelNames), ...
        'Checkpoint models differ.');
    assert(isequal(S.trainEnd,trainEnd) && isequal(S.valStart,valStart) ...
        && isequal(S.valEnd,valEnd), 'Checkpoint fold periods differ.');
    assert(isequal(double(S.yOOFCheck(:)),yAll(allDay>=10 & allDay<=24)), ...
        'Checkpoint row order/response does not match.');
    OOF = S.OOF; State = S.State; TrainSec = S.TrainSec;
    FitCount = S.FitCount; Errors = S.Errors;
    if isfield(S,'LinearLSQRFlag'), LinearLSQRFlag=S.LinearLSQRFlag; end
    fprintf('Checkpoint found; completed fold-model pairs will be skipped.\n');
end

%% 4. EXPANDING-WINDOW TEMPORAL CV
for f=1:K
    tr = allDay>=1 & allDay<=trainEnd(f);
    va = allDay>=valStart(f) & allDay<=valEnd(f);
    XTrain = X(tr,:);
    XVal = X(va,:);
    yTrain = yAll(tr);
    yVal = yAll(va);
    fprintf('\n=========================================\n');
    fprintf('FOLD %d/5, train Jan 1-%d (N=%d), val Jan %d-%d (N=%d)\n', ...
        f,trainEnd(f),numel(yTrain),valStart(f),valEnd(f),numel(yVal));

    for j=1:J
        if State(f,j)=="OK"
            fprintf('SKIP: %s (already completed)\n',modelNames(j));
            continue;
        end
        rng(8100+100*f+j,'twister');
        fprintf('\nTraining %s ...\n',modelNames(j));
        try
            startT = tic;
            flag = NaN;
            switch j
                case 1 % Original 'Bagged Trees'
                    learner = templateTree('MinLeafSize',8, ...
                        'NumVariablesToSample','all');
                    mdl = fitrensemble(XTrain,yTrain, ...
                        'Method','Bag','NumLearningCycles',30, ...
                        'Learners',learner, ...
                        'CategoricalPredictors',catVars);
                    yp = stage21PredictBatched(mdl,XVal,2000);
                    clear mdl;
                case 2 % Original 'Linear', same one-hot/LSQR method as Stage 20
                    [A,AV] = stage21SparseDesign(XTrain,XVal,catVars);
                    [beta,flag] = lsqr(A,yTrain,1e-6,350);
                    yp = full(AV*beta);
                    clear A AV beta;
                    if flag~=0
                        fprintf(['NOTE: LSQR flag=%d. This is a numerical ' ...
                            'approximation; inspect before journal reporting.\n'],flag);
                    end
                case 3 % Original 'Boosted Trees'
                    learner = templateTree('MinLeafSize',8, ...
                        'NumVariablesToSample','all');
                    mdl = fitrensemble(XTrain,yTrain, ...
                        'Method','LSBoost','NumLearningCycles',30, ...
                        'LearnRate',0.1,'Learners',learner, ...
                        'CategoricalPredictors',catVars);
                    yp = stage21PredictBatched(mdl,XVal,2000);
                    clear mdl;
                case 4 % Original 'Coarse Tree'
                    mdl = fitrtree(XTrain,yTrain, ...
                        'MinLeafSize',36,'Surrogate','off', ...
                        'CategoricalPredictors',catVars);
                    yp = stage21PredictBatched(mdl,XVal,2000);
                    clear mdl;
            end
            yp = double(yp(:));
            assert(numel(yp)==numel(yVal) && all(isfinite(yp)), ...
                'Predictions are incomplete or nonfinite.');
            TrainSec(f,j) = toc(startT);
            FitCount(f,j) = numel(yTrain);
            LinearLSQRFlag(f,j) = flag;
            OOF(va,j) = yp;
            State(f,j)="OK";
            Errors(f,j)="";
            e = yVal-yp;
            fprintf(['OK | RMSE=%.4f | MAE=%.4f | R2=%.5f | ' ...
                'seconds=%.1f\n'], ...
                sqrt(mean(e.^2)),mean(abs(e)), ...
                1-sum(e.^2)/sum((yVal-mean(yVal)).^2),TrainSec(f,j));
        catch ME
            State(f,j)="FAILED";
            Errors(f,j)=string(regexprep(ME.message,'[\r\n]+',' '));
            fprintf(2,'FAILED (%s): %s\n',modelNames(j),ME.message);
        end
        % Write successful predictions after EACH preset.
        yOOFCheck=yAll(allDay>=10 & allDay<=24); %#ok<NASGU>
        save(ckPath,'sourcePath','modelNames','trainEnd', ...
            'valStart','valEnd','yOOFCheck','OOF','State', ...
            'TrainSec','FitCount','LinearLSQRFlag','Errors','-v7.3');
        fprintf('Checkpoint saved: fold %d, model %d/%d.\n',f,j,J);
    end
end

%% 5. PERFORMANCE: FOLD TABLE AND POOLED OOF TABLE
foldRows = table();
for f=1:K
    va = allDay>=valStart(f) & allDay<=valEnd(f);
    ya = yAll(va);
    for j=1:J
        yp = OOF(va,j);
        if State(f,j)=="OK" && all(isfinite(yp))
            e=ya-yp;
            row=table(f,modelNames(j),numel(ya), ...
                sqrt(mean(e.^2)),mean(abs(e)), ...
                1-sum(e.^2)/sum((ya-mean(ya)).^2), ...
                mean(yp-ya),100*mean(abs(e)<=15), ...
                TrainSec(f,j),FitCount(f,j),LinearLSQRFlag(f,j), ...
                'VariableNames',{'Fold','Model','NVal','RMSE_min', ...
                'MAE_min','R2','Bias_min','Within15_pct', ...
                'FitPredict_sec','NTrain','LSQRFlag'});
            foldRows = [foldRows;row]; %#ok<AGROW>
        end
    end
end

maskOOF = allDay>=10 & allDay<=24;
yOOF = yAll(maskOOF);
baseNames = modelNames;
baseP = OOF(maskOOF,:);

%% 6. INCLUDE PREVIOUSLY SAVED OOF COMPARATORS, IF AVAILABLE
prior11 = fullfile(folderPath,'FlightDelay_Stage11_ResidualTimeCV.mat');
if isfile(prior11)
    S11=load(prior11,'predOOF');
    assert(size(S11.predOOF,1)==N && size(S11.predOOF,2)>=4, ...
        'Stage 11 OOF size/order is not compatible.');
    assert(all(abs(S11.predOOF(maskOOF,1)-depAll(maskOOF))<1e-8), ...
        'Stage 11 data order does not match (departure delay).');
    baseNames=[baseNames;"Departure Delay baseline"; ...
        "Constant Recovery baseline";"Linear Regression baseline"; ...
        "Residual Bagged Trees (earlier)"];
    baseP=[baseP,S11.predOOF(maskOOF,1:4)];
    fprintf('\nStage 11 baseline/residual OOF predictions loaded.\n');
else
    fprintf('\nStage 11 OOF not found: original presets only.\n');
end
prior13=fullfile(folderPath,'FlightDelay_Stage13_LSBoostTimeCV.mat');
if isfile(prior13)
    S13=load(prior13,'predBoostOOF');
    assert(numel(S13.predBoostOOF)==N, ...
        'Stage 13 OOF size/order is incompatible.');
    baseNames=[baseNames;"Residual LSBoost (earlier)"];
    baseP=[baseP,S13.predBoostOOF(maskOOF)];
    fprintf('Stage 13 LSBoost OOF predictions loaded.\n');
end

pooledRows=table();
for j=1:numel(baseNames)
    yp=double(baseP(:,j));
    if all(isfinite(yp))
        e=yOOF-yp;
        row=table(baseNames(j),numel(yOOF), ...
            sqrt(mean(e.^2)),mean(abs(e)), ...
            1-sum(e.^2)/sum((yOOF-mean(yOOF)).^2), ...
            mean(yp-yOOF),100*mean(abs(e)<=15), ...
            100*mean(abs(e)<=30), ...
            'VariableNames',{'Model','NVal','RMSE_min','MAE_min', ...
            'R2','Bias_min','Within15_pct','Within30_pct'});
        pooledRows=[pooledRows;row]; %#ok<AGROW>
    else
        fprintf('Not pooled (missing fold predictions): %s\n',baseNames(j));
    end
end
pooledRows=sortrows(pooledRows,'RMSE_min','ascend');

fprintf('\n=========== STAGE 21: ALL FOLDS ===========\n');
disp(foldRows);
fprintf('\n=========== STAGE 21: POOLED OOF RANKING ===========\n');
disp(pooledRows);
fprintf('\nSuccessful fold-model runs: %d of %d\n', ...
    sum(State=="OK",'all'),numel(State));

%% 7. EXPORT AND SAVE
writetable(foldRows,fullfile(folderPath,'Stage21_OriginalPresets_FoldCV.csv'));
writetable(pooledRows,fullfile(folderPath,'Stage21_OriginalPresets_PooledOOF.csv'));
finalPath=fullfile(folderPath,'Stage21_OriginalTopPresets_TimeCV.mat');
save(finalPath,'modelNames','trainEnd','valStart','valEnd', ...
    'OOF','foldRows','pooledRows','State','TrainSec', ...
    'FitCount','LinearLSQRFlag','Errors','-v7.3');
fprintf('\n===== STAGE 21 COMPLETED =====\n');
fprintf('Saved: %s\n',finalPath);

%% LOCAL FUNCTION: BATCH PREDICT
function yp=stage21PredictBatched(mdl,XV,batchSize)
    nv=height(XV);
    yp=nan(nv,1);
    for start=1:batchSize:nv
        ending=min(start+batchSize-1,nv);
        yp(start:ending)=double(predict(mdl,XV(start:ending,:)));
    end
end

%% LOCAL FUNCTION: SPARSE NUMERIC + ONE-HOT LINEAR DESIGN
function [A,AV]=stage21SparseDesign(XT,XV,categoriesVars)
    vn=XT.Properties.VariableNames;
    nT=height(XT);
    nV=height(XV);
    A=sparse(ones(nT,1));
    AV=sparse(ones(nV,1));
    for k=1:numel(vn)
        v=vn{k};
        if ismember(v,categoriesVars)
            a=double(XT.(v));
            b=double(XV.(v));
            nl=numel(categories(XT.(v)));
            if nl>1
                ii=find(a>1 & ~isnan(a));
                jj=find(b>1 & ~isnan(b));
                A=[A sparse(ii,a(ii)-1,1,nT,nl-1)]; %#ok<AGROW>
                AV=[AV sparse(jj,b(jj)-1,1,nV,nl-1)]; %#ok<AGROW>
            end
        else
            a=double(XT.(v));
            b=double(XV.(v));
            m=mean(a);
            s=std(a);
            if s==0, s=1; end
            A=[A sparse((a-m)/s)]; %#ok<AGROW>
            AV=[AV sparse((b-m)/s)]; %#ok<AGROW>
        end
    end
end
