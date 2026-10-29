%% STAGE 20: ORIGINAL 23 REGRESSION-LEARNER PRESETS - SINGLE RUN
% Frontiers in Built Environment / January 2025 / revision
% DESIGN: 1-19 January TRAIN; 20-24 January DEVELOPMENT VALIDATION.
% 25-31 January is NOT touched by this script.
%
% PURPOSE: retain the same 23 model/preset names in the original Table 9.
% RESPONSE: ARR_DELAY in MINUTES, not a normalized score.
% PREDICTION TIME: post-departure, because DEP_DELAY is included.
% No leakage variables (CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY,
% SECURITY_DELAY, LATE_AIRCRAFT_DELAY, TAXI_IN/OUT, ACTUAL_ELAPSED_TIME).
%
% IMPORTANT DIFFERENCES IN LARGE-DATA IMPLEMENTATION:
% - 5 exact nonlinear SVM presets use 12,000 common training rows (subsample).
% - 2 GPR presets use all training rows, but sparse 'sr' approximation
%   with a 500-observation active set (not exact/full GPR).
% - 2 kernel-approximation models use 1024 random expansion dimensions
%   in place of MATLAB 'auto' to limit memory.
% - Linear/Robust Linear use sparse least-squares/robust IRLS solvers.
% Every difference is recorded in the result's FitImplementation column.
% Thus: all 23 presets are INCLUDED, but not all are trained exactly
% on the full 354,991 training rows. Never claim otherwise.
%
% MATLAB R2024b or later; Statistics and Machine Learning Toolbox.
% MODEL CHECKPOINT: after every completed/failed model, written to disk.
% Interrupt with Ctrl+C; run again and choose the SAME Excel file to resume.

clear; clc; format compact;
rng(42,'twister');

%% 0. RESOURCE SETTINGS (record any changes in Methods)
MAX_SVM_TRAIN = 12000;  % Exact kernel SVM: manageable fixed TRAIN subset
GPR_ACTIVE_SET = 500;  % Sparse Gaussian process approximation
KERNEL_EXPANSION_DIMS = 1024; % Random Fourier features / KAR (instead of auto)
KERNEL_MAX_ITER = 1000;  % Same iteration limit as old paper
NN_MAX_ITER = 1000;      % Same iteration limit as old paper
PREDICTION_BATCH = 1500; % Predict on entire 92,075-row validation set
SAVE_EVERY_MODEL = true;

%% 1. LOAD THE ORIGINAL DATA FILE
[fileName,folderPath] = uigetfile({'*.xlsx;*.xls','Excel Files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fileName,0), error('File not selected.'); end
sourcePath = fullfile(folderPath,fileName);
fprintf('\nReading data: %s\n',sourcePath);
T = readtable(sourcePath,'Sheet','TranstatsData_Ayse');

%% 2. ELIGIBLE OBSERVATIONS - DO NOT FILTER DELAYED FLIGHTS
mask = (T.CANCELLED==0) & (T.DIVERTED==0) & ...
       ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
rawVars = {'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_DEP_TIME', ...
    'CRS_ARR_TIME','CRS_ELAPSED_TIME','DISTANCE', ...
    'DEP_DELAY','ARR_DELAY'};
M = rmmissing(T(mask,rawVars));
assert(height(M)==578957, 'Unexpected sample size; check Excel data.');

day = double(M.DAY_OF_MONTH);
fitMask = day>=1 & day<=19;
valMask = day>=20 & day<=24;
assert(sum(fitMask)==354991 && sum(valMask)==92075, ...
    'Date-split sample sizes differ from previous stages.');

%% 3. SAME MODEL-B FEATURES AS EARLIER STAGES
X = M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
    'DISTANCE','DEP_DELAY'});

depHHMM = double(M.CRS_DEP_TIME);
arrHHMM = double(M.CRS_ARR_TIME);
depMinutes = floor(depHHMM/100)*60 + mod(depHHMM,100);
arrMinutes = floor(arrHHMM/100)*60 + mod(arrHHMM,100);
assert(all(depMinutes>=0 & depMinutes<1440));
assert(all(arrMinutes>=0 & arrMinutes<1440));
X.DepTimeSin = sin(2*pi*depMinutes/1440);
X.DepTimeCos = cos(2*pi*depMinutes/1440);
X.ArrTimeSin = sin(2*pi*arrMinutes/1440);
X.ArrTimeCos = cos(2*pi*arrMinutes/1440);

catVars = {'DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT','DEST_AIRPORT'};
for z = 1:numel(catVars)
    v = catVars{z};
    rawCategory = string(X.(v));
    levels = unique(rawCategory(fitMask));
    X.(v) = categorical(rawCategory,levels);
    encodedCategory = X.(v);
    assert(~any(isundefined(encodedCategory(valMask))), ...
        'Validation category not in training data: %s',v);
end

XFit = X(fitMask,:);
XVal = X(valMask,:);
yFit = double(M.ARR_DELAY(fitMask));
yVal = double(M.ARR_DELAY(valMask));
clear T M X;

fprintf('\nTotal eligible flights: 578957\n');
fprintf('Development training: %d (Jan 1-19)\n',numel(yFit));
fprintf('Development validation: %d (Jan 20-24)\n',numel(yVal));
fprintf('Response: actual ARR_DELAY, minutes\n');
fprintf('Prediction timing: after departure (DEP_DELAY included)\n');

%% 4. ORIGINAL TABLE 9 ORDER: EXACTLY 23 NAMES
modelNames = [ ...
    "Trilayered NN";
    "Bilayered NN";
    "Wide NN";
    "Medium NN";
    "Narrow NN";
    "Rational Quadratic GPR";
    "Exponential GPR";
    "Medium Gaussian SVM";
    "Coarse Gaussian SVM";
    "Quadratic SVM";
    "Linear SVM";
    "Cubic SVM";
    "SVM Kernel";
    "Least Squares Regression Kernel";
    "Boosted Trees";
    "Bagged Trees";
    "Coarse Tree";
    "Fine Tree";
    "Medium Tree";
    "Linear";
    "Robust Linear";
    "Efficient Linear Least Squares";
    "Efficient Linear SVM"];
assert(numel(modelNames)==23);

nModels = numel(modelNames);
NUsed = nan(nModels,1);
TrainSec = nan(nModels,1);
PredictSec = nan(nModels,1);
RMSE = nan(nModels,1);
MAE = nan(nModels,1);
MSE = nan(nModels,1);
R2 = nan(nModels,1);
Bias = nan(nModels,1);
MAPE_nonzero_pct = nan(nModels,1);
Status = repmat("NOT_RUN",nModels,1);
FitImplementation = repmat("",nModels,1);
ErrorDetails = repmat("",nModels,1);
P = nan(numel(yVal),nModels);

%% 5. COMMON, SEEDED TRAINING SUBSET FOR EXACT SVM PRESETS
% Stratified on day (not on outcome); all models 8-12 use SAME subset.
rng(2025,'twister');
trainDays = day(fitMask);
svmRows = zeros(0,1);
for d = 1:19
    rows = find(trainDays==d);
    nPick = max(1,round(MAX_SVM_TRAIN*numel(rows)/numel(trainDays)));
    nPick = min(nPick,numel(rows));
    svmRows = [svmRows; rows(randperm(numel(rows),nPick))]; %#ok<AGROW>
end
if numel(svmRows)>MAX_SVM_TRAIN
    svmRows = svmRows(1:MAX_SVM_TRAIN);
end
svmRows = sort(svmRows);
fprintf('Exact-SVM training subset: %d rows (shared).\n',numel(svmRows));

%% 6. RESUME CHECKPOINT
ckFile = fullfile(folderPath,'Stage20_23Presets_Checkpoint.mat');
if isfile(ckFile)
    S = load(ckFile);
    assert(strcmp(string(S.sourcePath),string(sourcePath)), ...
        'Existing checkpoint points to another Excel file.');
    assert(isequal(string(S.modelNames),modelNames), ...
        'Checkpoint preset names/order differ.');
    assert(isequal(double(S.yVal(:)),yVal), ...
        'Checkpoint has different validation response/order.');
    NUsed=S.NUsed; TrainSec=S.TrainSec; PredictSec=S.PredictSec;
    RMSE=S.RMSE; MAE=S.MAE; MSE=S.MSE; R2=S.R2;
    Bias=S.Bias; MAPE_nonzero_pct=S.MAPE_nonzero_pct;
    Status=S.Status; FitImplementation=S.FitImplementation;
    ErrorDetails=S.ErrorDetails; P=S.P;
    fprintf('\nCheckpoint loaded. Finished models will be skipped.\n');
end

%% 7. TRAIN 23 PRESETS, ONE BY ONE
for k = 1:nModels
    if Status(k)=="OK"
        fprintf('Skipping %02d/23: %s [%s]\n', ...
            k,modelNames(k),Status(k));
        continue;
    end

    fprintf('\n-----------------------------------------------\n');
    fprintf('%02d / 23: %s\n',k,modelNames(k));
    rng(4200+k,'twister');
    try
        t0 = tic;
        [prediction,nTrain,implementation] = trainOriginalPreset( ...
            k,XFit,yFit,XVal,catVars,svmRows, ...
            GPR_ACTIVE_SET,KERNEL_EXPANSION_DIMS, ...
            KERNEL_MAX_ITER,NN_MAX_ITER,PREDICTION_BATCH);
        trTime = toc(t0);
        % PredictSec is included in TrainSec for these function-local models.
        % Full implementation time is reported; predict-only time is NA.
        prediction = double(prediction(:));
        assert(numel(prediction)==numel(yVal));
        assert(all(isfinite(prediction)),'Nonfinite predictions found.');

        e = yVal-prediction;
        NUsed(k) = nTrain;
        TrainSec(k) = trTime;
        RMSE(k) = sqrt(mean(e.^2));
        MSE(k) = mean(e.^2);
        MAE(k) = mean(abs(e));
        R2(k) = 1 - sum(e.^2)/sum((yVal-mean(yVal)).^2);
        Bias(k) = mean(prediction-yVal);
        nz = abs(yVal)>eps;
        MAPE_nonzero_pct(k) = 100*mean(abs(e(nz)./yVal(nz)));
        Status(k) = "OK";
        FitImplementation(k) = string(implementation);
        ErrorDetails(k) = "";
        P(:,k) = prediction;

        fprintf(['OK: RMSE=%.4f MAE=%.4f R2=%.5f ' ...
             'Bias=%.3f NFit=%d elapsed=%.1f s\n'], ...
             RMSE(k),MAE(k),R2(k),Bias(k),NUsed(k),trTime);
        fprintf('Method: %s\n',char(FitImplementation(k)));

    catch ME
        Status(k) = "FAILED";
        FitImplementation(k) = "Failed - no metric should be reported";
        ErrorDetails(k) = string(regexprep(ME.message,'[\r\n]+',' '));
        fprintf(2,'FAILED: %s\n',ME.message);
        if strcmp(ME.identifier,'MATLAB:nomem')
            fprintf(2,'Memory error: data remain unchanged.\n');
        end
    end

    if SAVE_EVERY_MODEL
        save(ckFile,'sourcePath','modelNames','yVal','NUsed', ...
            'TrainSec','PredictSec','RMSE','MSE','MAE','R2','Bias', ...
            'MAPE_nonzero_pct','Status','FitImplementation', ...
            'ErrorDetails','P','svmRows', ...
            'GPR_ACTIVE_SET','KERNEL_EXPANSION_DIMS', ...
            'MAX_SVM_TRAIN','NN_MAX_ITER','KERNEL_MAX_ITER','-v7.3');
        writeCurrentResults(folderPath,modelNames,NUsed,TrainSec, ...
            RMSE,MSE,MAE,R2,Bias,MAPE_nonzero_pct, ...
            Status,FitImplementation,ErrorDetails);
        fprintf('Checkpoint saved (%02d/23).\n',k);
    end
end

%% 8. OUTPUT TABLES AND RANKING
Results = writeCurrentResults(folderPath,modelNames,NUsed, ...
    TrainSec,RMSE,MSE,MAE,R2,Bias,MAPE_nonzero_pct, ...
    Status,FitImplementation,ErrorDetails);

fprintf('\n=========== ALL 23 PRESETS ===========\n');
disp(Results(:,{'Preset','NTrainUsed','RMSE_min','MAE_min', ...
    'R2','Status','FitImplementation'}));

success = Status=="OK";
if any(success)
    Ranking = sortrows(Results(success,:),'RMSE_min','ascend');
    fprintf('\n=========== BEST DEVELOPMENT MODELS ===========\n');
    disp(Ranking(1:min(10,height(Ranking)), ...
        {'Preset','RMSE_min','MAE_min','R2','NTrainUsed', ...
         'FitImplementation'}));
end

fprintf('\nTraining failures: %d of 23\n',sum(Status=="FAILED"));
fprintf('Successes: %d of 23\n',sum(success));
fprintf('Important: 20-24 January is development validation, NOT final test.\n');
fprintf('Models marked subsampled/sparse/approx are NOT exact full-data presets.\n');
fprintf('MAT file: %s\n',ckFile);
fprintf('CSV file: %s\n',fullfile(folderPath,'Stage20_23Presets_Results.csv'));

%% LOCAL FUNCTIONS
function [p,nUsed,impl] = trainOriginalPreset( ...
    k,XTr,yTr,XVa,cats,svmRows,gprActive,kExp,kIter,nnIter,batchN)

    nUsed = height(XTr);
    switch k
        case {1,2,3,4,5} % ORIGINAL 5 NN PRESETS
            switch k
                case 1
                    sz=[10 10 10];
                case 2
                    sz=[10 10];
                case 3
                    sz=100;
                case 4
                    sz=25;
                case 5
                    sz=10;
            end
            mdl = fitrnet(XTr,yTr, ...
                'CategoricalPredictors',cats, ...
                'LayerSizes',sz,'Activations','relu', ...
                'Lambda',0,'Standardize',true, ...
                'IterationLimit',nnIter,'Verbose',0);
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Full training; NN ReLU layers=%s, iter=%d, lambda=0', ...
                mat2str(sz),nnIter);

        case {6,7} % GPR TWO KERNELS: SPARSE FULL-N APPROX
            if k==6, ker='rationalquadratic';
            else, ker='exponential'; end
            mdl = fitrgp(XTr,yTr, ...
                'CategoricalPredictors',cats, ...
                'Basis','constant','KernelFunction',ker, ...
                'Standardize',true, ...
                'FitMethod','sr','PredictMethod','sr', ...
                'ActiveSetSize',gprActive, ...
                'ActiveSetMethod','random');
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Sparse GPR (sr), all N rows; active set %d; not exact GPR',gprActive);

        case {8,9,10,11,12} % EXACT SVM, COMMON SUBSET
            XS = XTr(svmRows,:);
            ys = yTr(svmRows);
            nUsed = numel(ys);
            switch k
                case 8
                    kv = {'KernelFunction','gaussian','KernelScale',3.6};
                case 9
                    kv = {'KernelFunction','gaussian','KernelScale',14};
                case 10
                    kv = {'KernelFunction','polynomial', ...
                          'PolynomialOrder',2,'KernelScale','auto'};
                case 11
                    kv = {'KernelFunction','linear','KernelScale','auto'};
                case 12
                    kv = {'KernelFunction','polynomial', ...
                          'PolynomialOrder',3,'KernelScale','auto'};
            end
            mdl = fitrsvm(XS,ys, ...
                'CategoricalPredictors',cats, ...
                'Standardize',true,kv{:});
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Exact SVM kernel, training subsample %d of %d; ALL validation scored', ...
                nUsed,height(XTr));

        case {13,14} % 2 KERNEL APPROXIMATION PRESETS
            if k==13, learner='svm';
            else, learner='leastsquares'; end
            mdl = fitrkernel(XTr,yTr, ...
                'CategoricalPredictors',cats, ...
                'Learner',learner,'KernelScale','auto', ...
                'NumExpansionDimensions',kExp, ...
                'Lambda','auto','Standardize',true, ...
                'IterationLimit',kIter,'BlockSize',1024);
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Full N, random kernel expansion dimension=%d (bounded; old auto changed)',kExp);

        case 15 % BOOSTED TREES; SAME PRESET PARAMETERS
            tmp = templateTree('MinLeafSize',8, ...
                'NumVariablesToSample','all');
            mdl = fitrensemble(XTr,yTr, ...
                'Method','LSBoost','NumLearningCycles',30, ...
                'LearnRate',0.1,'Learners',tmp, ...
                'CategoricalPredictors',cats);
            p = batchedPredict(mdl,XVa,batchN);
            impl = 'Full N; boosted trees 30, leaf 8, LR 0.1';

        case 16 % BAGGED TREES
            tmp = templateTree('MinLeafSize',8, ...
                'NumVariablesToSample','all');
            mdl = fitrensemble(XTr,yTr, ...
                'Method','Bag','NumLearningCycles',30, ...
                'Learners',tmp,'CategoricalPredictors',cats);
            p = batchedPredict(mdl,XVa,batchN);
            impl = 'Full N; bagged trees 30, leaf 8, all predictors';

        case {17,18,19} % THREE SINGLE TREES
            switch k
                case 17
                    leaf=36;
                case 18
                    leaf=4;
                case 19
                    leaf=12;
            end
            mdl = fitrtree(XTr,yTr, ...
                'MinLeafSize',leaf,'Surrogate','off', ...
                'CategoricalPredictors',cats);
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Full N; regression tree, MinLeafSize=%d',leaf);

        case {20,21} % LINEAR AND ROBUST LINEAR
            [A,AV] = buildSparseLinearDesign(XTr,XVa,cats);
            [beta,flag] = lsqr(A,yTr,1e-6,350);
            if flag>1, warning('Linear LSQR flag: %d',flag); end
            if k==20
                p = full(AV*beta);
                impl = sprintf('Full N; sparse OLS via LSQR (flag %d)',flag);
            else
                % Bisquare IRLS approximation to robust linear preset
                c = 4.685;
                for iter=1:6
                    r = yTr-A*beta;
                    med = median(r);
                    scale = 1.4826*median(abs(r-med));
                    scale = max(scale,0.001);
                    u = (r-med)/(c*scale);
                    w = zeros(size(u));
                    good = abs(u)<1;
                    w(good) = (1-u(good).^2).^2;
                    w = max(w,1e-5);
                    W = spdiags(sqrt(w),0,numel(w),numel(w));
                    betaNew = lsqr(W*A,sqrt(w).*yTr,1e-5,300);
                    beta = betaNew;
                end
                p = full(AV*beta);
                impl = 'Full N; sparse robust Tukey bisquare IRLS (6 passes; approximate MATLAB robustfit)';
            end

        case {22,23} % EFFICIENT LINEAR PRESETS
            if k==22, learner='leastsquares';
            else, learner='svm'; end
            mdl = fitrlinear(XTr,yTr, ...
                'CategoricalPredictors',cats, ...
                'Learner',learner, ...
                'Regularization','ridge', ...
                'BetaTolerance',1e-4);
            p = batchedPredict(mdl,XVa,batchN);
            impl = sprintf('Full N; fitrlinear, %s, ridge, BetaTol=1e-4',learner);

        otherwise
            error('Unexpected preset ID: %d',k);
    end
    p=double(p(:));
end

function p = batchedPredict(model,T,sz)
    n = height(T);
    p = nan(n,1);
    for a=1:sz:n
        b=min(a+sz-1,n);
        p(a:b)=double(predict(model,T(a:b,:)));
    end
end

function [A,AV] = buildSparseLinearDesign(XTr,XVa,catVars)
    vns = XTr.Properties.VariableNames;
    nTr = height(XTr);
    nVa = height(XVa);
    A = sparse(ones(nTr,1));
    AV = sparse(ones(nVa,1));

    for j=1:numel(vns)
        v=vns{j};
        if ismember(v,catVars)
            xTr=double(XTr.(v));
            xVa=double(XVa.(v));
            nlev=numel(categories(XTr.(v)));
            if nlev>1
                ix=find(xTr>1);
                jx=find(xVa>1);
                atr=sparse(ix,xTr(ix)-1,1,nTr,nlev-1);
                ava=sparse(jx,xVa(jx)-1,1,nVa,nlev-1);
                A=[A atr]; %#ok<AGROW>
                AV=[AV ava]; %#ok<AGROW>
            end
        else
            zTr=double(XTr.(v));
            zVa=double(XVa.(v));
            mu=mean(zTr);
            sd=std(zTr);
            if sd==0, sd=1; end
            A=[A sparse((zTr-mu)/sd)]; %#ok<AGROW>
            AV=[AV sparse((zVa-mu)/sd)]; %#ok<AGROW>
        end
    end
end

function Result=writeCurrentResults(folder,modelNames,n,secs,rmse,mse,mae,r2,bias,mape,status,impl,errs)
    Result = table((1:numel(modelNames))',modelNames,n,secs, ...
        rmse,mse,mae,r2,bias,mape,status,impl,errs, ...
        'VariableNames',{'OriginalOrder','Preset','NTrainUsed', ...
        'TotalFitAndPredict_sec','RMSE_min','MSE_min2','MAE_min', ...
        'R2','Bias_min','MAPE_nonzero_pct','Status', ...
        'FitImplementation','ErrorDetails'});
    writetable(Result,fullfile(folder,'Stage20_23Presets_Results.csv'));
    try
        writetable(Result,fullfile(folder,'Stage20_23Presets_Results.xlsx'));
    catch ME
        warning('XLSX output failed (CSV is available): %s',ME.message);
    end
end
