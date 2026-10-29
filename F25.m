%% STAGE 29 - FULL-DATA ORIGINAL QUADRATIC SVM / CONTROLLED RESUME
% Flight-delay manuscript revision | MATLAB R2024b+ | Statistics and ML Toolbox
%
% Important scientific contract:
%  1. Every attempted fit uses ALL observations of its own training fold.
%  2. No 12,000-row SVM sample, sparse/active-set GPR, or manually bounded
%     number of kernel expansion features is used.
%  3. Exact GPR is requested explicitly. If insufficient memory is allowed,
%     a GPR fit is BLOCKED (without substituted predictions).
%  4. The two KAR presets are MATLAB's ORIGINAL kernel-approximation
%     algorithms (fitrkernel); their intrinsic random-feature expansion is
%     an integral feature of the published preset, not a substitution.
%  5. Native fitlm is used for the original Linear / Robust Linear presets.
%  6. Jan 25-31 final test data is NEVER used for fitting/validation here.
%  7. Results are written after each fold/preset; restart to resume.
%  8. This targeted run enables ONLY original preset #10 (Quadratic SVM).
%     It initially runs Fold 1 only as a full-data feasibility test.
%     Change FOLDS_TO_RUN=1:5 to complete all folds if feasible.
%
% Hardware warning: an EXACT GPR Gram matrix alone needs about 238 GB
% on Fold 1 and 1.22 TB on Fold 5. Several copies may be needed. On a
% usual desktop, both GPR presets will be explicitly RESOURCE_BLOCKED.
% Exact SVMs with 170k-390k records can also take an extremely long time.
%
% This script does not promise that all 115 exact fits can finish on a
% desktop: a completed model is distinguished from an uncomputed model.

clear; clc; format compact;
rng(42,'twister');

%% CONFIGURATION - DO NOT ALTER ORIGINAL PRESET HYPERPARAMETERS
GPR_SINGLE_MATRIX_BUDGET_GIB = 128;  % Raise ONLY on appropriately large-RAM HPC
GPR_EXTRA_SAFETY_MULTIPLIER = 3;     % Estimated lower bound for working memory
SVM_CACHE_MB = 1000;                % fitrsvm documented default, bounded cache
PRED_BATCH = 256;                   % Output batching only; no training subsampling
RETRY_FAILED = false;               % Set true only to reattempt previous failures
CHECKPOINT_EVERY_MODEL = true;
PROJECT_TAG = 'FULL23_EXACT_NATIVE_5F_V1'; % MUST match existing checkpoint

%% RESUME CONTROL - NO MODEL OR HYPERPARAMETER CHANGES
% One exact full-fold MATLAB fitrsvm run, preserving earlier Stage27
% checkpoint data and ALL 23 original model rows. Stop with Ctrl+C if
% the native optimizer is prohibitively slow. No automatic fit timeout
% or background process management is claimed.
ACTIVE_PRESETS = 10;       % ONLY original preset 10: Quadratic SVM, native fitrsvm full fold
DEFER_PRESETS = [];       % Never silently downgrade or defer this targeted model
FOLDS_TO_RUN = 1;        % Benchmark only fold 1 first; change to 1:5 to finish
% After confirming feasibility, set FOLDS_TO_RUN=1:5. Completed folds
% automatically SKIP and models not selected remain untouched.
% To target full-data Linear SVM afterward: ACTIVE_PRESETS=11.
% To target another SVM: indices 8,9,11,12.
% Do not edit PROJECT_TAG or the Excel file between runs.


%% FILE SELECTION AND VERIFICATION
[fileName,folderPath] = uigetfile({'*.xlsx;*.xls','Excel Files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fileName,0), error('No Excel file was selected.'); end
sourcePath = fullfile(folderPath,fileName);
fi = dir(sourcePath);
fingerprint = sprintf('%s|%d|%.10f',sourcePath,fi.bytes,fi.datenum);

fprintf('\nReading: %s\n',sourcePath);
T = readtable(sourcePath,'Sheet','TranstatsData_Ayse');
required = {'CANCELLED','DIVERTED','ARR_DELAY','DEP_DELAY', ...
    'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT', ...
    'DEST_AIRPORT','CRS_DEP_TIME','CRS_ARR_TIME', ...
    'CRS_ELAPSED_TIME','DISTANCE'};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'Some expected raw BTS column names are missing.');

eligible = T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
rawVars = [required(5:end), {'ARR_DELAY','DEP_DELAY'}];
M = rmmissing(T(eligible,rawVars));
assert(height(M)==578957, ...
    'Expected 578957 eligible observations. Do not silently change sample.');
clear T;

% Features exactly match the previously revised post-departure Model B.
Xraw = M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
    'DISTANCE','DEP_DELAY'});

tDep = double(M.CRS_DEP_TIME);
tArr = double(M.CRS_ARR_TIME);
mDep = floor(tDep/100)*60 + mod(tDep,100);
mArr = floor(tArr/100)*60 + mod(tArr,100);
assert(all(mDep>=0 & mDep<1440) && all(mArr>=0 & mArr<1440), ...
    'Scheduled HHMM values out of range.');
Xraw.DepTimeSin = sin(2*pi*mDep/1440);
Xraw.DepTimeCos = cos(2*pi*mDep/1440);
Xraw.ArrTimeSin = sin(2*pi*mArr/1440);
Xraw.ArrTimeCos = cos(2*pi*mArr/1440);

allDay = double(M.DAY_OF_MONTH(:));
yAll = double(M.ARR_DELAY(:));
assert(all(isfinite(yAll)));
clear M;

catVars = {'DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT', ...
    'DEST_AIRPORT'};

%% FIVE EXPANDING TEMPORAL FOLDS - ORIGINAL CV DESIGN
trainEnd = [9 12 15 18 21];
valStart = [10 13 16 19 22];
valEnd   = [12 15 18 21 24];
F = 5;

assert(isequal(arrayfun(@(z)sum(allDay<=z),trainEnd), ...
    [172375 225076 280333 337480 390901]),'Training split mismatch');
assert(isequal(arrayfun(@(a,b)sum(allDay>=a & allDay<=b), ...
    valStart,valEnd),[52701 55257 57147 53421 56165]), ...
    'Validation split mismatch');

% OOF predictions are stored only for Jan 10-24, NEVER Jan 25-31.
devMask = allDay>=10 & allDay<=24;
devDays = allDay(devMask);
yDev = yAll(devMask);
assert(numel(yDev)==274691);

%% PRESET NAMES - EXACT ORIGINAL TABLE 9 ORDER
modelNames = [ ...
    "Trilayered NN"; ...
    "Bilayered NN"; ...
    "Wide NN"; ...
    "Medium NN"; ...
    "Narrow NN"; ...
    "Rational Quadratic GPR"; ...
    "Exponential GPR"; ...
    "Medium Gaussian SVM"; ...
    "Coarse Gaussian SVM"; ...
    "Quadratic SVM"; ...
    "Linear SVM"; ...
    "Cubic SVM"; ...
    "SVM Kernel"; ...
    "Least Squares Regression Kernel"; ...
    "Boosted Trees"; ...
    "Bagged Trees"; ...
    "Coarse Tree"; ...
    "Fine Tree"; ...
    "Medium Tree"; ...
    "Linear"; ...
    "Robust Linear"; ...
    "Efficient Linear Least Squares"; ...
    "Efficient Linear SVM"];
assert(numel(modelNames)==23);
J = numel(modelNames);

Status = repmat("NOT_RUN",F,J);
Errors = strings(F,J);
FitSeconds = nan(F,J);
PredictSeconds = nan(F,J);
NTrainUsed = nan(F,J);
Converged = strings(F,J);
NewCategoryCounts = zeros(F,numel(catVars));
OOF = nan(numel(yDev),J);

% Run computationally manageable families first; output row order remains
% IDENTICAL to the old Table 9. The end places most costly native models.
runOrder = [17 18 19 20 21 22 23 15 16 1 2 3 4 5 13 14 11 10 12 8 9 6 7];
assert(isequal(sort(runOrder),1:23));

outDir = fullfile(folderPath,'Stage27_Full23_5Fold');
if ~exist(outDir,'dir'), mkdir(outDir); end
ckFile = fullfile(outDir,'Stage27_Full23_Checkpoint.mat');
logFile = fullfile(outDir,'Stage27_ExecutionLog.txt');
diary(logFile); diary on;

if isfile(ckFile)
    S = load(ckFile);
    assert(strcmp(string(S.fingerprint),string(fingerprint)), ...
        'Source Excel path/size/timestamp changed: cannot resume checkpoint.');
    assert(strcmp(string(S.PROJECT_TAG),string(PROJECT_TAG)), ...
        'Checkpoint belongs to a different script version.');
    assert(isequal(string(S.modelNames),modelNames), ...
        'Checkpoint has different presets.');
    assert(isequal(double(S.yDev(:)),yDev), ...
        'Checkpoint target values/order are different.');
    Status=S.Status; Errors=S.Errors; OOF=S.OOF;
    FitSeconds=S.FitSeconds; PredictSeconds=S.PredictSeconds;
    NTrainUsed=S.NTrainUsed; Converged=S.Converged;
    NewCategoryCounts=S.NewCategoryCounts;
    fprintf('Existing checkpoint loaded. Completed tasks will be skipped.\n');
    % Recover from an interrupted fit, if a RUNNING flag was persisted.
    interrupted = Status=="RUNNING";
    if any(interrupted,'all')
        fprintf('Resetting %d interrupted RUNNING entries to NOT_RUN.\n',sum(interrupted,'all'));
        Status(interrupted)="NOT_RUN";
        Errors(interrupted)="Interrupted before completion; not counted.";
    end
end

fprintf('\n=== ALL 23 FULL-DATA PRESETS / 5 TEMPORAL FOLDS ===\n');
fprintf('Development flights: %d; unique validation flights: %d\n', ...
    sum(allDay<=24),numel(yDev));
fprintf('Exact GPR: resource guard budget %.1f GiB / safety x %.1f\n', ...
    GPR_SINGLE_MATRIX_BUDGET_GIB,GPR_EXTRA_SAFETY_MULTIPLIER);
fprintf('SVM training uses ALL fold rows; no fixed-N subsampling.\n');

%% MAIN 5 x 23 LOOP
for f=FOLDS_TO_RUN
    tr = allDay>=1 & allDay<=trainEnd(f);
    va = allDay>=valStart(f) & allDay<=valEnd(f);
    vDev = devDays>=valStart(f) & devDays<=valEnd(f);
    nTr = sum(tr); nVa = sum(va);
    assert(sum(vDev)==nVa);
    fprintf('\n===================================================\n');
    fprintf('FOLD %d/5 | train Jan 1-%d (%d) | val Jan %d-%d (%d)\n', ...
        f,trainEnd(f),nTr,valStart(f),valEnd(f),nVa);

    % Category vocabularies are constructed EXCLUSIVELY using fold training.
    [XTr,XVa,unseen] = prepareFoldCategories( ...
        Xraw(tr,:),Xraw(va,:),catVars);
    NewCategoryCounts(f,:) = unseen;
    fprintf('Unseen validation categories (mapped to training mode): %s\n', ...
        mat2str(unseen));
    yTr = yAll(tr);
    yVa = yAll(va);

    for z=1:J
        j=runOrder(z);

        % If isolating a preset, leave all other checkpoint entries untouched.
        if ~ismember(j,ACTIVE_PRESETS)
            continue;
        end

        % Defer a computational bottleneck WITHOUT attempting to fit it.
        % This is not a successful result and never enters pooled metrics.
        if ismember(j,DEFER_PRESETS)
            if Status(f,j)~="OK" && Status(f,j)~="DEFERRED"
                Status(f,j)="DEFERRED";
                Errors(f,j)="Explicitly deferred; no fit, no approximation, no data subsampling.";
                stage27Persist(ckFile,fingerprint,PROJECT_TAG, ...
                    modelNames,yDev,Status,Errors,OOF,FitSeconds, ...
                    PredictSeconds,NTrainUsed,Converged,NewCategoryCounts);
            end
            fprintf('DEFER f=%d, j=%02d: %s (NOT fitted)\n', ...
                f,j,modelNames(j));
            continue;
        end

        % A previously deferred preset CAN be resumed once DEFER_PRESETS
        % is removed and ACTIVE_PRESETS includes that model.
        if Status(f,j)=="DEFERRED"
            Status(f,j)="NOT_RUN";
            Errors(f,j)="";
        end

        if Status(f,j)=="OK" || ...
          (~RETRY_FAILED && Status(f,j)~="NOT_RUN")
            fprintf('SKIP f=%d, j=%02d: %s [%s]\n', ...
                f,j,modelNames(j),Status(f,j));
            continue;
        end
        fprintf('\n[%d/5 | %02d/23] %s\n',f,j,modelNames(j));
        rng(970000+1000*f+j,'twister');
        Status(f,j)="RUNNING";
        Errors(f,j)="";
        NTrainUsed(f,j)=NaN; % Assign only AFTER successful full-N fit
        tStart=tic;
        try
            % Prevent a known-impossible GPR allocation. This never
            % substitutes sparse/active-set/small-N GPR.
            if j==6 || j==7
                oneMatrixGiB = 8 * double(nTr)^2 / 2^30;
                estWorkingGiB = oneMatrixGiB * GPR_EXTRA_SAFETY_MULTIPLIER;
                if oneMatrixGiB > GPR_SINGLE_MATRIX_BUDGET_GIB
                    Status(f,j)="RESOURCE_BLOCKED";
                    Errors(f,j)=sprintf([ ...
                        'Full exact GPR requires >= %.1f GiB for ONE ' ...
                        'kernel matrix; >= %.1f GiB conservative working ' ...
                        'memory estimate. No fit/substitute was run. ' ...
                        'Use HPC and raise guard after verifying RAM.'], ...
                        oneMatrixGiB,estWorkingGiB);
                    fprintf(2,'%s\n',Errors(f,j));
                    FitSeconds(f,j)=toc(tStart);
                    stage27Persist(ckFile,fingerprint,PROJECT_TAG, ...
                        modelNames,yDev,Status,Errors,OOF,FitSeconds, ...
                        PredictSeconds,NTrainUsed,Converged,NewCategoryCounts);
                    continue;
                end
            end

            [mdl,impl] = fitOriginalPresetFull(j,XTr,yTr,catVars, ...
                SVM_CACHE_MB);
            FitSeconds(f,j)=toc(tStart);
            NTrainUsed(f,j)=nTr; % Every observed fold-training record was used
            if any(~isfinite(yTr))
                error('Nonfinite training responses unexpectedly found.');
            end
            fprintf('FITTED all %d rows using %s\n',nTr,impl);
            Converged(f,j) = convergenceFlag(mdl);
            fprintf('Convergence check: %s\n',Converged(f,j));

            tP=tic;
            yp = predictBatches(mdl,XVa,PRED_BATCH);
            PredictSeconds(f,j)=toc(tP);
            clear mdl;
            yp=double(yp(:));
            assert(numel(yp)==nVa && all(isfinite(yp)), ...
                'Invalid or nonfinite validation predictions.');

            OOF(vDev,j)=yp;
            Status(f,j)="OK";
            Errors(f,j)="";
            e=yVa-yp;
            rmse=sqrt(mean(e.^2));
            mae=mean(abs(e));
            r2=1-sum(e.^2)/sum((yVa-mean(yVa)).^2);
            fprintf('OK RMSE=%.4f MAE=%.4f R2=%.5f fit=%.1f predict=%.1f s\n', ...
                rmse,mae,r2,FitSeconds(f,j),PredictSeconds(f,j));

        catch ME
            Status(f,j)="FAILED";
            Errors(f,j)=string(getReport(ME,'basic','hyperlinks','off'));
            FitSeconds(f,j)=toc(tStart);
            OOF(vDev,j)=NaN;
            fprintf(2,'FAILED without substitution: %s\n',ME.message);
            clear mdl;
        end
        if CHECKPOINT_EVERY_MODEL
            stage27Persist(ckFile,fingerprint,PROJECT_TAG, ...
                modelNames,yDev,Status,Errors,OOF,FitSeconds, ...
                PredictSeconds,NTrainUsed,Converged,NewCategoryCounts);
        end
    end
    clear XTr XVa yTr yVa;
end

%% COMPUTE RESULTS - NEVER ASSIGN METRICS TO FAILED PRESETS
[FoldTable,PooledTable] = buildResults( ...
    Status,OOF,modelNames,trainEnd,valStart,valEnd, ...
    devDays,yDev,FitSeconds,PredictSeconds,NTrainUsed, ...
    Converged,Errors);

fprintf('\n=========== FOLD SUMMARY ===========\n');
disp(FoldTable(:,{'Fold','Preset','Status','NTrain', ...
    'RMSE_min','MAE_min','R2'}));
fprintf('\n=========== POOLED 5-FOLD OOF - ORIGINAL 23 ORDER ===========\n');
disp(PooledTable(:,{'Preset','CompletedFolds','Status', ...
    'RMSE_min','MAE_min','R2','TotalFit_sec'}));

complete = PooledTable.Status=="COMPLETE_5_OF_5";
if any(complete)
    ranked = sortrows(PooledTable(complete,:), 'RMSE_min','ascend');
    fprintf('\n=========== RANKING: ONLY FULLY COMPLETED PRESETS ===========\n');
    disp(ranked(:,{'Preset','RMSE_min','MAE_min','R2'}));
else
    fprintf('No model has all five folds completed.\n');
end

fprintf('\nDONE: %d of 115 fold/preset runs completed.\n',sum(Status=="OK",'all'));
fprintf('GPR resource blocks: %d. Training failures: %d.\n', ...
    sum(Status=="RESOURCE_BLOCKED",'all'),sum(Status=="FAILED",'all'));
fprintf('Output folder: %s\n',outDir);
writetable(FoldTable,fullfile(outDir,'Stage27_All23_FoldResults.csv'));
writetable(PooledTable,fullfile(outDir,'Stage27_All23_PooledOOF.csv'));
try
    writetable(FoldTable,fullfile(outDir,'Stage27_Results.xlsx'), ...
        'Sheet','FoldResults');
    writetable(PooledTable,fullfile(outDir,'Stage27_Results.xlsx'), ...
        'Sheet','PooledOOF');
catch ME
    warning('Optional XLSX export failed; CSV files remain: %s',ME.message);
end
diary off;

%% LOCAL FUNCTIONS (R2024b)
function [XTr,XVa,unseen] = prepareFoldCategories(XTr,XVa,catVars)
    unseen=zeros(1,numel(catVars));
    for q=1:numel(catVars)
        v=catVars{q};
        trVal=string(XTr.(v));
        vaVal=string(XVa.(v));
        labs=sort(unique(trVal));
        catTr=categorical(trVal,cellstr(labs));
        counts=countcats(catTr);
        [~,imax]=max(counts);
        fallback=labs(imax);  % TRAIN-only most frequent category
        bad=~ismember(vaVal,labs);
        unseen(q)=sum(bad);
        vaVal(bad)=fallback;
        XTr.(v)=categorical(trVal,cellstr(labs));
        XVa.(v)=categorical(vaVal,cellstr(labs));
        assert(~any(isundefined(XVa.(v))), ...
            'Unseen categorical levels remain after mapping.');
    end
end

function [mdl,impl] = fitOriginalPresetFull(j,XTr,yTr,catVars,cacheMB)
    % All arguments supplied here are original Table 10 settings, except
    % CacheSize which is only a memory cache of fitrsvm (default 1000 MB).
    switch j
        case {1,2,3,4,5}
            layers={[10 10 10],[10 10],100,25,10};
            mdl=fitrnet(XTr,yTr,'CategoricalPredictors',catVars, ...
                'LayerSizes',layers{j},'Activations','relu', ...
                'IterationLimit',1000,'Lambda',0,'Standardize',true, ...
                'Verbose',0);
            impl=sprintf('native fitrnet, original layers %s, N=full', ...
                mat2str(layers{j}));
        case {6,7}
            if j==6, ker='rationalquadratic';
            else, ker='exponential'; end
            mdl=fitrgp(XTr,yTr,'CategoricalPredictors',catVars, ...
                'BasisFunction','constant','KernelFunction',ker, ...
                'Standardize',true, ...
                'FitMethod','exact','PredictMethod','exact', ...
                'Optimizer','quasinewton');
            impl=sprintf('fitrgp EXACT/EXACT, %s, N=full',ker);
        case {8,9,10,11,12}
            switch j
                case 8
                    kval={'KernelFunction','gaussian','KernelScale',3.6};
                case 9
                    kval={'KernelFunction','gaussian','KernelScale',14};
                case 10
                    kval={'KernelFunction','polynomial', ...
                        'PolynomialOrder',2,'KernelScale','auto'};
                case 11
                    kval={'KernelFunction','linear','KernelScale','auto'};
                case 12
                    kval={'KernelFunction','polynomial', ...
                        'PolynomialOrder',3,'KernelScale','auto'};
            end
            mdl=fitrsvm(XTr,yTr,'CategoricalPredictors',catVars, ...
                'Standardize',true,'CacheSize',cacheMB,kval{:});
            impl='native FULL-N fitrsvm; NO training subset or low-rank method';
        case {13,14}
            if j==13, learner='svm';
            else, learner='leastsquares'; end
            mdl=fitrkernel(XTr,yTr, ...
                'CategoricalPredictors',catVars,'Learner',learner, ...
                'NumExpansionDimensions','auto', ...
                'Lambda','auto','KernelScale','auto', ...
                'Standardize',true,'IterationLimit',1000);
            impl=['native FULL-N fitrkernel, original AUTO expansion; ' ...
                'intrinsic kernel approximation preset'];
        case {15,16}
            learner=templateTree('MinLeafSize',8, ...
                'NumVariablesToSample','all');
            if j==15
                mdl=fitrensemble(XTr,yTr,'Method','LSBoost', ...
                    'NumLearningCycles',30,'LearnRate',0.1, ...
                    'Learners',learner,'CategoricalPredictors',catVars);
                impl='native full-N Boosted Trees, 30 learners/leaf 8/LR .1';
            else
                mdl=fitrensemble(XTr,yTr,'Method','Bag', ...
                    'NumLearningCycles',30,'Learners',learner, ...
                    'CategoricalPredictors',catVars);
                impl='native full-N Bagged Trees, 30 learners/leaf 8';
            end
        case {17,18,19}
            leafs=[36 4 12];
            leaf=leafs(j-16);
            mdl=fitrtree(XTr,yTr,'MinLeafSize',leaf, ...
                'Surrogate','off','CategoricalPredictors',catVars);
            impl=sprintf('native full-N regression tree; leaf=%d',leaf);
        case {20,21}
            if j==20
                mdl=fitlm(XTr,yTr,'linear','RobustOpts','off');
                impl='native fitlm, original Linear preset, full-N';
            else
                mdl=fitlm(XTr,yTr,'linear','RobustOpts','on');
                impl='native fitlm, RobustOpts=on bisquare, full-N';
            end
        case {22,23}
            if j==22, learner='leastsquares';
            else, learner='svm'; end
            mdl=fitrlinear(XTr,yTr,'CategoricalPredictors',catVars, ...
                'Learner',learner,'Lambda','auto','BetaTolerance',1e-4);
            impl=['native full-N fitrlinear, learner=' learner ...
                ', original AUTO regularization/solver/lambda'];
        otherwise
            error('Unknown original preset index: %d',j);
    end
end

function yp=predictBatches(mdl,XVa,batch)
    n=height(XVa);
    yp=nan(n,1);
    for a=1:batch:n
        b=min(n,a+batch-1);
        yp(a:b)=double(predict(mdl,XVa(a:b,:)));
    end
end

function txt=convergenceFlag(mdl)
    txt="NOT_EXPOSED";
    try
        if isprop(mdl,'ConvergenceInfo')
            c=mdl.ConvergenceInfo;
            if isstruct(c) && isfield(c,'Converged')
                txt=string(c.Converged);
            end
        end
    catch
        txt="NOT_EXPOSED";
    end
end

function [foldTable,poolTable]=buildResults( ...
    status,preds,names,trEnds,vaStarts,vaEnds,dd,yd, ...
    fitSecs,predSecs,ntr,conv,errs)
    rowCount=numel(status);
    Fold=zeros(rowCount,1); OriginalOrder=zeros(rowCount,1);
    Preset=strings(rowCount,1); Status=strings(rowCount,1);
    NTrain=nan(rowCount,1); NValidation=zeros(rowCount,1);
    RMSE=nan(rowCount,1); MAE=nan(rowCount,1);
    MSE=nan(rowCount,1); R2=nan(rowCount,1);
    Bias=nan(rowCount,1); Within15=nan(rowCount,1);
    FitSec=nan(rowCount,1); PredSec=nan(rowCount,1);
    Converged=strings(rowCount,1); ErrorDetails=strings(rowCount,1);
    r=0;
    for f=1:5
        v=dd>=vaStarts(f) & dd<=vaEnds(f);
        y=yd(v);
        for j=1:numel(names)
            r=r+1;
            Fold(r)=f; OriginalOrder(r)=j; Preset(r)=names(j);
            Status(r)=status(f,j); NTrain(r)=ntr(f,j);
            NValidation(r)=numel(y);
            FitSec(r)=fitSecs(f,j); PredSec(r)=predSecs(f,j);
            Converged(r)=conv(f,j); ErrorDetails(r)=errs(f,j);
            yp=preds(v,j);
            if status(f,j)=="OK" && all(isfinite(yp))
                e=y-yp;
                MSE(r)=mean(e.^2);
                RMSE(r)=sqrt(MSE(r));
                MAE(r)=mean(abs(e));
                R2(r)=1-sum(e.^2)/sum((y-mean(y)).^2);
                Bias(r)=mean(yp-y);
                Within15(r)=100*mean(abs(e)<=15);
            end
        end
    end
    foldTable=table(Fold,OriginalOrder,Preset,Status,NTrain, ...
        NValidation,RMSE,MAE,MSE,R2,Bias,Within15,FitSec,PredSec, ...
        Converged,ErrorDetails, ...
        'VariableNames',{'Fold','OriginalOrder','Preset','Status', ...
        'NTrain','NValidation','RMSE_min','MAE_min','MSE_min2', ...
        'R2','Bias_min','Within15_pct','Fit_sec','Predict_sec', ...
        'Converged','ErrorDetails'});

    J=numel(names);
    OriginalOrder=(1:J)'; Preset=names(:);
    CompletedFolds=zeros(J,1);
    Status=strings(J,1);
    NValidation=zeros(J,1);
    RMSE=nan(J,1); MAE=nan(J,1); R2=nan(J,1);
    MSE=nan(J,1); Bias=nan(J,1); Within15=nan(J,1);
    TotalFit=nan(J,1); TotalPredict=nan(J,1);
    for j=1:J
        CompletedFolds(j)=sum(status(:,j)=="OK");
        if CompletedFolds(j)==5
            p=preds(:,j);
            e=yd-p;
            NValidation(j)=numel(yd);
            MSE(j)=mean(e.^2); RMSE(j)=sqrt(MSE(j));
            MAE(j)=mean(abs(e));
            R2(j)=1-sum(e.^2)/sum((yd-mean(yd)).^2);
            Bias(j)=mean(p-yd);
            Within15(j)=100*mean(abs(e)<=15);
            Status(j)="COMPLETE_5_OF_5";
        else
            Status(j)="INCOMPLETE";
        end
        TotalFit(j)=sum(fitSecs(:,j),'omitnan');
        TotalPredict(j)=sum(predSecs(:,j),'omitnan');
    end
    poolTable=table(OriginalOrder,Preset,CompletedFolds,Status, ...
        NValidation,RMSE,MAE,MSE,R2,Bias,Within15, ...
        TotalFit,TotalPredict, ...
        'VariableNames',{'OriginalOrder','Preset','CompletedFolds', ...
        'Status','NValidation','RMSE_min','MAE_min','MSE_min2', ...
        'R2','Bias_min','Within15_pct','TotalFit_sec', ...
        'TotalPredict_sec'});
end

function stage27Persist(ckFile,fingerprint,PROJECT_TAG,modelNames,yDev, ...
    Status,Errors,OOF,FitSeconds,PredictSeconds,NTrainUsed, ...
    Converged,NewCategoryCounts)
    state=struct();
    state.fingerprint=fingerprint;
    state.PROJECT_TAG=PROJECT_TAG;
    state.modelNames=modelNames;
    state.yDev=yDev;
    state.Status=Status;
    state.Errors=Errors;
    state.OOF=OOF;
    state.FitSeconds=FitSeconds;
    state.PredictSeconds=PredictSeconds;
    state.NTrainUsed=NTrainUsed;
    state.Converged=Converged;
    state.NewCategoryCounts=NewCategoryCounts;
    tmpFile=[ckFile,'.partial.mat'];
    save(tmpFile,'-struct','state','-v7.3');
    [ok,message]=movefile(tmpFile,ckFile,'f');
    assert(ok,'Could not replace checkpoint: %s',message);
end
