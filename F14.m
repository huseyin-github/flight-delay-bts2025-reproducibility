%% STAGE 22 - CHECK NUMERICAL ACCURACY OF ORIGINAL 'LINEAR' PRESET
% Revision / frontiers flight-delay study
% Recomputes ONLY the original Linear preset for the 5 development folds.
% All other Stage21 predictions remain unchanged.
% Old Stage21 LSQR returned flag 1 in all folds: inspect numerical accuracy.
% Primary solve: sparse QR least squares via A\y (no new ML tuning).
% If sparse QR cannot complete, record FAILED instead of mislabeling a
% regularized/approximate fit as exact least squares.
% NO rows from Jan 25-31 are used anywhere in this script.

clear; clc; format compact;
[fileName,folderPath] = uigetfile({'*.xlsx;*.xls','Excel Files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fileName,0), error('Input Excel not selected.'); end
src=fullfile(folderPath,fileName);
s21Path=fullfile(folderPath,'Stage21_OriginalTopPresets_TimeCV.mat');
assert(isfile(s21Path),'Stage21 MAT not found in the Excel folder.');
S21=load(s21Path,'OOF','modelNames','trainEnd','valStart','valEnd');
assert(any(string(S21.modelNames)=="Linear"),'Original Linear not in Stage21 MAT.');
oldIdx=find(string(S21.modelNames)=="Linear",1);

fprintf('\nReading data: %s\n',src);
T=readtable(src,'Sheet','TranstatsData_Ayse');
keep=T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
vars={'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
      'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_DEP_TIME', ...
      'CRS_ARR_TIME','CRS_ELAPSED_TIME','DISTANCE', ...
      'DEP_DELAY','ARR_DELAY'};
M=rmmissing(T(keep,vars)); clear T;
assert(height(M)==578957,'Unexpected eligible sample size.');
day=double(M.DAY_OF_MONTH);
y=double(M.ARR_DELAY);
X=M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
       'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
       'DISTANCE','DEP_DELAY'});
d=double(M.CRS_DEP_TIME); a=double(M.CRS_ARR_TIME);
dm=floor(d/100)*60+mod(d,100);
am=floor(a/100)*60+mod(a,100);
assert(all(dm>=0 & dm<1440 & am>=0 & am<1440),'Invalid clock time.');
X.DepTimeSin=sin(2*pi*dm/1440);
X.DepTimeCos=cos(2*pi*dm/1440);
X.ArrTimeSin=sin(2*pi*am/1440);
X.ArrTimeCos=cos(2*pi*am/1440);
catVars={'DAY_OF_WEEK','MKT_CARRIER','ORIGIN_AIRPORT','DEST_AIRPORT'};
for k=1:numel(catVars)
    v=catVars{k};
    raw=string(X.(v));
    % EXACTLY the same category coding as Stage21
    labels=unique(raw(day<=24));
    X.(v)=categorical(raw,labels);
end

trainEnd=double(S21.trainEnd(:));
valStart=double(S21.valStart(:));
valEnd=double(S21.valEnd(:));
K=numel(trainEnd);
assert(K==5,'Expected five folds.');
N=numel(y);
predQR=nan(N,1);
resultRows=table();
ck=fullfile(folderPath,'Stage22_LinearQR_Checkpoint.mat');
completed=false(K,1);
if isfile(ck)
    C=load(ck,'src','predQR','completed','resultRows', ...
        'trainEnd','valStart','valEnd');
    assert(strcmp(string(src),string(C.src)),'Checkpoint input mismatch.');
    assert(isequal(C.trainEnd,trainEnd) && ...
        isequal(C.valStart,valStart) && isequal(C.valEnd,valEnd), ...
        'Checkpoint time folds mismatch.');
    predQR=C.predQR; completed=C.completed;
    resultRows=C.resultRows;
    fprintf('Resume: completed folds will be skipped.\n');
end

for f=1:K
    if completed(f), fprintf('Fold %d already complete.\n',f); continue; end
    tr=day<=trainEnd(f) & day>=1;
    va=day>=valStart(f) & day<=valEnd(f);
    fprintf('\nFold %d: train Jan 1-%d N=%d, validate Jan %d-%d N=%d\n', ...
        f,trainEnd(f),sum(tr),valStart(f),valEnd(f),sum(va));
    [A,AV]=stage22SparseDesign(X(tr,:),X(va,:),catVars);
    ytr=y(tr); yva=y(va);
    oldPred=double(S21.OOF(va,oldIdx));
    assert(all(isfinite(oldPred)),'Old OOF predictions not complete.');
    fprintf('Design matrix: %d x %d, nnz=%d\n', ...
        size(A,1),size(A,2),nnz(A));
    start=tic;
    lastwarn('');
    try
        beta=A\ytr; % sparse QR least-squares solve
        fitTime=toc(start);
        [warnMsg,warnId]=lastwarn;
        if ~isempty(warnMsg)
            fprintf('MATLAB solver warning (%s): %s\n',warnId,warnMsg);
        end
        assert(all(isfinite(beta)),'QR coefficients are nonfinite.');
        yp=double(AV*beta);
        assert(all(isfinite(yp)),'QR validation predictions nonfinite.');
        trResidual=ytr-A*beta;
        oldE=yva-oldPred; newE=yva-yp;
        qrRMSE=sqrt(mean(newE.^2));
        qrMAE=mean(abs(newE));
        oldRMSE=sqrt(mean(oldE.^2));
        oldMAE=mean(abs(oldE));
        deltaRMSE=qrRMSE-oldRMSE;
        deltaMAE=qrMAE-oldMAE;
        maxPredDiff=max(abs(yp-oldPred));
        medPredDiff=median(abs(yp-oldPred));
        relNormalGrad=norm(A'*trResidual)/(norm(A'*ytr)+eps);
        relTrainingResidual=norm(trResidual)/norm(ytr);
        row=table(f,sum(tr),sum(va),size(A,2),fitTime, ...
            oldRMSE,qrRMSE,deltaRMSE,oldMAE,qrMAE,deltaMAE, ...
            maxPredDiff,medPredDiff,relNormalGrad,relTrainingResidual, ...
            string(warnMsg), ...
            'VariableNames',{'Fold','NTrain','NVal','NDesignColumns', ...
            'FitSeconds','OldLinear_RMSE','QRLinear_RMSE', ...
            'QRminusOld_RMSE','OldLinear_MAE','QRLinear_MAE', ...
            'QRminusOld_MAE','MaxPredictionDifference', ...
            'MedianPredictionDifference','RelativeNormalGradient', ...
            'RelativeTrainingResidual','SolverWarning'});
        resultRows=[resultRows;row]; %#ok<AGROW>
        predQR(va)=yp;
        completed(f)=true;
        fprintf('Old RMSE %.4f | Sparse-QR RMSE %.4f | Delta %.4f\n', ...
            oldRMSE,qrRMSE,deltaRMSE);
        fprintf('Old MAE  %.4f | Sparse-QR MAE  %.4f | Delta %.4f\n', ...
            oldMAE,qrMAE,deltaMAE);
        fprintf('Normal-equation relative gradient: %.3g\n',relNormalGrad);
        fprintf('Max abs prediction change: %.4f minutes\n',maxPredDiff);
    catch ME
        fprintf(2,'Fold %d sparse QR FAILED: %s\n',f,ME.message);
        fprintf(2,'Fold was NOT marked completed. Inspect RAM/warnings.\n');
        rethrow(ME);
    end
    save(ck,'src','predQR','completed','resultRows', ...
        'trainEnd','valStart','valEnd','-v7.3');
end

%% SAME OOF ROWS / FAIR POOLED COMPARISON
m=day>=10 & day<=24;
yv=y(m);
assert(all(isfinite(predQR(m))),'Some QR folds incomplete.');
q=predQR(m);
old=S21.OOF(m,oldIdx);
Pooled=table(["Stage21 Linear (LSQR)";"Stage22 Linear (sparse QR)"], ...
    [sqrt(mean((yv-old).^2));sqrt(mean((yv-q).^2))], ...
    [mean(abs(yv-old));mean(abs(yv-q))], ...
    [1-sum((yv-old).^2)/sum((yv-mean(yv)).^2); ...
     1-sum((yv-q).^2)/sum((yv-mean(yv)).^2)], ...
    [mean(old-yv);mean(q-yv)], ...
    'VariableNames',{'Model','RMSE_min','MAE_min','R2','Bias_min'});

fprintf('\n===== STAGE 22 FOLD AUDIT =====\n');
disp(resultRows);
fprintf('\n===== STAGE 22 POOLED COMPARISON =====\n');
disp(Pooled);
out=fullfile(folderPath,'Stage22_Linear_Numerical_Audit.mat');
save(out,'resultRows','Pooled','predQR', ...
    'trainEnd','valStart','valEnd','-v7.3');
writetable(resultRows,fullfile(folderPath,'Stage22_Linear_FoldAudit.csv'));
writetable(Pooled,fullfile(folderPath,'Stage22_Linear_PooledAudit.csv'));
fprintf('\nSTAGE 22 COMPLETED: %s\n',out);

function [A,AV]=stage22SparseDesign(XT,XV,categoriesVars)
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
