%% STAGE 24 - ROBUST LINEAR: 5-FOLD TEMPORAL CV AND NUMERICAL AUDIT
% Flight delay manuscript revision / January 2025
% PURPOSE: evaluate ORIGINAL Table 9 'Robust Linear' family with a
% numerically stable sparse-QR implementation of Tukey bisquare IRLS.
% TRAIN/VAL ONLY: Jan 1-24. JAN 25-31 IS NEVER READ FOR SCORING.
%
% We reproduce Stage 23's robust-loss definition (Tukey bisquare,
% tuning constant 4.685, MAD scale, weight floor 1e-5) but replace
% its nonconverged weighted LSQR subproblems with sparse QR A\y.
% SIX IRLS passes are the primary, prespecified Stage 23 setting.
% 12 passes are calculated ONLY as a numerical convergence diagnostic;
% NOT a second candidate tuned on performance.
%
% NOTE: this is a numerically corrected *approximation* to MATLAB's
% Regression Learner Robust Linear preset, not an assertion of exact
% fitlm(...,'RobustOpts','on') equivalence.
%
% Checkpoints after each fold, rerunnable safely. Requirements:
% Statistics and Machine Learning Toolbox, Excel source as Stage 20-23.

clear; clc; format compact;
rng(42,'twister');

%% 1. Select file, load previous temporal fold definitions
[fileName,folderPath]=uigetfile({'*.xlsx;*.xls','Excel files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fileName,0),error('No file selected');end
sourcePath=fullfile(folderPath,fileName);

stage21Path=fullfile(folderPath,'Stage21_OriginalTopPresets_TimeCV.mat');
stage22Path=fullfile(folderPath,'Stage22_Linear_Numerical_Audit.mat');
assert(isfile(stage21Path) && isfile(stage22Path), ...
    'Stage21 and Stage22 MAT files must be in the Excel folder.');
S21=load(stage21Path,'trainEnd','valStart','valEnd','OOF','modelNames');
S22=load(stage22Path,'predQR');
trainEnd=double(S21.trainEnd(:));
valStart=double(S21.valStart(:));
valEnd=double(S21.valEnd(:));
assert(isequal(trainEnd,[9;12;15;18;21]) && ...
       isequal(valStart,[10;13;16;19;22]) && ...
       isequal(valEnd,[12;15;18;21;24]), ...
    'Stage21 temporal CV folds differ from expected definitions.');

fprintf('\nReading data: %s\n',sourcePath);
T=readtable(sourcePath,'Sheet','TranstatsData_Ayse');
keep=T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
cols={'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
      'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_DEP_TIME', ...
      'CRS_ARR_TIME','CRS_ELAPSED_TIME','DISTANCE', ...
      'DEP_DELAY','ARR_DELAY'};
M=rmmissing(T(keep,cols));
clear T;
assert(height(M)==578957,'Eligible sample does not match previous stages.');
day=double(M.DAY_OF_MONTH);
y=double(M.ARR_DELAY);

%% 2. Exactly the same development predictor construction
X=M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
    'DISTANCE','DEP_DELAY'});
d=double(M.CRS_DEP_TIME);
a=double(M.CRS_ARR_TIME);
dm=floor(d/100)*60+mod(d,100);
am=floor(a/100)*60+mod(a,100);
assert(all(dm>=0 & dm<1440 & am>=0 & am<1440));
X.DepTimeSin=sin(2*pi*dm/1440);
X.DepTimeCos=cos(2*pi*dm/1440);
X.ArrTimeSin=sin(2*pi*am/1440);
X.ArrTimeCos=cos(2*pi*am/1440);

catVars={'DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT','DEST_AIRPORT'};
for j=1:numel(catVars)
    v=catVars{j};
    raw=string(X.(v));
    labels=unique(raw(day<=24));
    X.(v)=categorical(raw,labels);
end
clear M d a dm am;

K=5; N=height(X);
foldFinished=false(K,1);
predRobust6=nan(N,1);
predRobust12=nan(N,1);
FoldAudit=table();
checkpoint=fullfile(folderPath,'Stage24_RobustLinear_Checkpoint.mat');

if isfile(checkpoint)
    C=load(checkpoint);
    assert(strcmp(string(C.sourcePath),string(sourcePath)), ...
        'Checkpoint belongs to a different Excel path.');
    assert(isequal(C.trainEnd,trainEnd) && ...
           isequal(C.valStart,valStart) && ...
           isequal(C.valEnd,valEnd),'Checkpoint fold mismatch.');
    foldFinished=C.foldFinished;
    predRobust6=C.predRobust6;
    predRobust12=C.predRobust12;
    FoldAudit=C.FoldAudit;
    fprintf('Checkpoint loaded: completed folds will be skipped.\n');
end

%% 3. Five expanding-window training fits
for f=1:K
    if foldFinished(f)
        fprintf('\nFold %d already completed.\n',f);
        continue;
    end
    tr=day>=1 & day<=trainEnd(f);
    va=day>=valStart(f) & day<=valEnd(f);
    fprintf('\n========== FOLD %d/5 ==========',f);
    fprintf('\nTraining days 1-%d (N=%d), validation days %d-%d (N=%d)\n', ...
        trainEnd(f),sum(tr),valStart(f),valEnd(f),sum(va));

    [A,AV]=stage24SparseDesign(X(tr,:),X(va,:),catVars);
    yt=y(tr); yv=y(va);
    fprintf('Design: %d rows x %d columns, nnz=%d\n', ...
        size(A,1),size(A,2),nnz(A));
    tic;
    beta=A\yt; % exact initial ordinary least squares using sparse QR
    assert(all(isfinite(beta)),'Nonfinite starting coefficients.');
    lastPred=double(AV*beta);
    tolW=1e-5;
    saved6=nan(numel(yv),1);
    pass6Delta=NaN;
    pass12Delta=NaN;
    minW6=NaN;
    relWGrad6=NaN;

    for it=1:12
        r=yt-A*beta;
        center=median(r);
        scale=1.4826*median(abs(r-center));
        scale=max(scale,0.001);
        u=(r-center)/(4.685*scale);
        w=zeros(size(u));
        good=abs(u)<1;
        w(good)=(1-u(good).^2).^2;
        w=max(w,tolW);
        sw=sqrt(w);
        WA=spdiags(sw,0,numel(sw),numel(sw))*A;
        wy=sw.*yt;
        % Unlike Stage23 LSQR, use sparse QR to solve each weighted LS.
        betaNew=WA\wy;
        assert(all(isfinite(betaNew)),'Nonfinite weighted QR coefficients.');
        currentPred=double(AV*betaNew);
        passDelta=max(abs(currentPred-lastPred));
        beta=betaNew;
        lastPred=currentPred;
        if it==6
            saved6=currentPred;
            pass6Delta=passDelta;
            minW6=mean(w<=tolW*1.001)*100;
            % Weighted normal equation gradient for the solved subproblem.
            weightedResidual=wy-WA*beta;
            relWGrad6=norm(WA'*weightedResidual)/ ...
                (norm(WA'*wy)+eps);
        end
        if it==12
            pass12Delta=passDelta;
        end
        fprintf('IRLS pass %02d | scale=%8.4f | max step pred=%7.4f min | w floor=%5.2f%%\n', ...
            it,scale,passDelta,100*mean(w<=tolW*1.001));
    end
    fitSeconds=toc;
    yp6=saved6;
    yp12=lastPred;
    assert(all(isfinite(yp6)) && all(isfinite(yp12)));
    predRobust6(va)=yp6;
    predRobust12(va)=yp12;
    % Save diagnostics for stability, do not choose based on validation score.
    e6=yv-yp6; e12=yv-yp12;
    linearPred=double(S22.predQR(va));
    assert(all(isfinite(linearPred)), ...
        'Stage22 QR predictions do not cover this validation fold.');
    el=yv-linearPred;
    row=table(f,sum(tr),sum(va),fitSeconds, ...
        sqrt(mean(e6.^2)),mean(abs(e6)), ...
        sqrt(mean(e12.^2)),mean(abs(e12)), ...
        sqrt(mean(el.^2)),mean(abs(el)), ...
        max(abs(yp12-yp6)),median(abs(yp12-yp6)), ...
        pass6Delta,pass12Delta,minW6,relWGrad6, ...
        'VariableNames',{'Fold','NTrain','NValidation','FitSeconds', ...
        'Robust6_RMSE','Robust6_MAE','Robust12_RMSE','Robust12_MAE', ...
        'LinearQR_RMSE','LinearQR_MAE', ...
        'MaxPredictionDiff_6to12','MedianPredictionDiff_6to12', ...
        'LastStepPredChange_Pass6','LastStepPredChange_Pass12', ...
        'WeightFloorPercent_Pass6','WeightedRelativeGradient_Pass6'});
    FoldAudit=[FoldAudit;row]; %#ok<AGROW>
    foldFinished(f)=true;
    fprintf('Fold %d | Robust6 RMSE=%.4f MAE=%.4f | LinearQR RMSE=%.4f\n', ...
        f,row.Robust6_RMSE,row.Robust6_MAE,row.LinearQR_RMSE);
    fprintf('Robust6-to-Robust12 max prediction change: %.4f min\n', ...
        row.MaxPredictionDiff_6to12);
    clear A AV WA wy sw w beta betaNew yt;
    save(checkpoint,'sourcePath','trainEnd','valStart','valEnd', ...
        'foldFinished','predRobust6','predRobust12','FoldAudit','-v7.3');
    fprintf('Checkpoint saved.\n');
end

%% 4. Pooled OOF comparison on exactly the same 274691 flights
mask=day>=10 & day<=24;
yv=y(mask);
assert(sum(mask)==274691);
assert(all(isfinite(predRobust6(mask))) && ...
       all(isfinite(predRobust12(mask))));

methods=["Robust Linear (6 QR passes)"; ...
         "Robust Linear (12 QR passes; diagnostic)"; ...
         "Linear (sparse QR; Stage22)"];
preds=[predRobust6(mask),predRobust12(mask), ...
       double(S22.predQR(mask))];

stage11Path=fullfile(folderPath,'FlightDelay_Stage11_ResidualTimeCV.mat');
if isfile(stage11Path)
    J=load(stage11Path,'predOOF');
    assert(size(J.predOOF,1)==N && size(J.predOOF,2)>=4);
    preds=[preds,J.predOOF(mask,4)]; %#ok<AGROW>
    methods=[methods;"Residual Bagged Trees (Stage11)"]; %#ok<AGROW>
end
stage13Path=fullfile(folderPath,'FlightDelay_Stage13_LSBoostTimeCV.mat');
if isfile(stage13Path)
    J=load(stage13Path,'predBoostOOF');
    assert(numel(J.predBoostOOF)==N);
    preds=[preds,double(J.predBoostOOF(mask))]; %#ok<AGROW>
    methods=[methods;"Residual LSBoost (Stage13)"]; %#ok<AGROW>
end
assert(all(isfinite(preds(:))),'Some OOF comparator predictions are incomplete.');

NModels=numel(methods);
RMSE=zeros(NModels,1); MAE=RMSE; R2=RMSE; Bias=RMSE;
for j=1:NModels
    err=yv-preds(:,j);
    RMSE(j)=sqrt(mean(err.^2));
    MAE(j)=mean(abs(err));
    R2(j)=1-sum(err.^2)/sum((yv-mean(yv)).^2);
    Bias(j)=mean(preds(:,j)-yv);
end
Pooled=table(methods,RMSE,MAE,R2,Bias, ...
    'VariableNames',{'Model','RMSE_min','MAE_min','R2','Bias_min'});
Pooled=sortrows(Pooled,'RMSE_min','ascend');

fprintf('\n=========== STAGE 24 FOLD AUDIT ===========\n');
disp(FoldAudit);
fprintf('\n=========== STAGE 24 POOLED OOF ===========\n');
disp(Pooled);
fprintf('\nCAUTION: Stage23 Robust Linear used weighted LSQR that did not converge.\n');
fprintf('Stage24 uses sparse QR for all IRLS passes; it is a numerical audit\n');
fprintf('on development folds, NOT a new or blind evaluation of the holdout.\n');

out=fullfile(folderPath,'Stage24_RobustLinear_TimeCV_NumericalAudit.mat');
save(out,'FoldAudit','Pooled','predRobust6','predRobust12', ...
    'trainEnd','valStart','valEnd','-v7.3');
writetable(FoldAudit,fullfile(folderPath,'Stage24_RobustLinear_FoldAudit.csv'));
writetable(Pooled,fullfile(folderPath,'Stage24_RobustLinear_PooledOOF.csv'));
fprintf('\nSTAGE 24 COMPLETED: %s\n',out);

%% Helper: same sparse design as Stage22 / Stage23
function [A,AV]=stage24SparseDesign(XT,XV,categoriesVars)
    vn=XT.Properties.VariableNames;
    nT=height(XT); nV=height(XV);
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
            a=double(XT.(v)); b=double(XV.(v));
            mu=mean(a); sd=std(a);
            if sd==0,sd=1;end
            A=[A sparse((a-mu)/sd)]; %#ok<AGROW>
            AV=[AV sparse((b-mu)/sd)]; %#ok<AGROW>
        end
    end
end
