%% STAGE 25 - ROBUST LINEAR NUMERICAL AUDIT AND PROPOSED TABLE 9 CORRECTION
% PURPOSE: Fix previously NONCONVERGED weighted LSQR in Stage20/Stage23
% by solving the SAME six robust IRLS passes with sparse QR (A\y).
% This is a numerical correction, NOT a new hyperparameter search.
% 12 IRLS passes are computed only as a convergence diagnostic.
%
% Two fixed, already examined windows:
%   DEVELOPMENT: Jan 1-19 -> Jan 20-24  (same as Stage20)
%   HOLDOUT:     Jan 1-24 -> Jan 25-31  (same as Stage23)
% Do not claim the holdout is wholly blind. Do not tune on holdout.
%
% Keeps all other 22 rows/presets unchanged; generates a PROPOSED
% Table 9 with the Robust Linear row corrected in both windows.
% Original Stage20/23 CSV, XLSX and MAT files are NEVER overwritten.
%
% Requirements: MATLAB with Statistics and Machine Learning Toolbox.

clear; clc; format compact;
rng(42,'twister');

%% 1. Select original Excel and previous checkpoint files
[fName,folderPath]=uigetfile({'*.xlsx;*.xls','Excel files'}, ...
    'Select TranstatsData_Ayse4.xlsx');
if isequal(fName,0), error('No Excel file selected.'); end
sourcePath=fullfile(folderPath,fName);

p20=fullfile(folderPath,'Stage20_23Presets_Checkpoint.mat');
p23=fullfile(folderPath,'Stage23_23Presets_Checkpoint.mat');
origTable=fullfile(folderPath,'Stage23_Table9_23Preset_Combined.csv');
assert(isfile(p20) && isfile(p23) && isfile(origTable), ...
    'Need Stage20/Stage23 checkpoints and their combined Table9 CSV.');

S20=load(p20,'P','yVal','modelNames');
S23=load(p23,'P','yVal','modelNames');
assert(isequal(string(S20.modelNames(:)),string(S23.modelNames(:))), ...
    'Preset order mismatch.');
assert(numel(S20.modelNames)==23 && ...
    string(S20.modelNames(21))=="Robust Linear", ...
    'Model index 21 must be Robust Linear.');

fprintf('\nReading data: %s\n',sourcePath);
T=readtable(sourcePath,'Sheet','TranstatsData_Ayse');
keep=T.CANCELLED==0 & T.DIVERTED==0 & ...
    ~ismissing(T.ARR_DELAY) & ~ismissing(T.DEP_DELAY);
rawVars={'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_DEP_TIME', ...
    'CRS_ARR_TIME','CRS_ELAPSED_TIME','DISTANCE', ...
    'DEP_DELAY','ARR_DELAY'};
M=rmmissing(T(keep,rawVars));
clear T;
assert(height(M)==578957,'Eligible population does not match.');

day=double(M.DAY_OF_MONTH);
y=double(M.ARR_DELAY);
X=M(:,{'DAY_OF_MONTH','DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT','CRS_ELAPSED_TIME', ...
    'DISTANCE','DEP_DELAY'});

depHHMM=double(M.CRS_DEP_TIME);
arrHHMM=double(M.CRS_ARR_TIME);
D=floor(depHHMM/100)*60+mod(depHHMM,100);
A=floor(arrHHMM/100)*60+mod(arrHHMM,100);
assert(all(D>=0 & D<1440 & A>=0 & A<1440), ...
    'Scheduled times are not valid HHMM values.');
X.DepTimeSin=sin(2*pi*D/1440);
X.DepTimeCos=cos(2*pi*D/1440);
X.ArrTimeSin=sin(2*pi*A/1440);
X.ArrTimeCos=cos(2*pi*A/1440);
clear M depHHMM arrHHMM D A;

catVars={'DAY_OF_WEEK','MKT_CARRIER', ...
    'ORIGIN_AIRPORT','DEST_AIRPORT'};

%% 2. Two prespecified original split definitions
trainEnd=[19;24];
evalStart=[20;25];
evalEnd=[24;31];
periodName=["Stage20 Development";"Stage23 Holdout"];

results=table();
Daily=table();
P6=cell(2,1);
P12=cell(2,1);
passDetail=cell(2,1);

for t=1:2
    tr=day<=trainEnd(t);
    va=day>=evalStart(t) & day<=evalEnd(t);
    yt=y(tr);
    yv=y(va);
    if t==1
        prev=S20;
        assert(numel(yv)==92075 && sum(tr)==354991);
    else
        prev=S23;
        assert(numel(yv)==131891 && sum(tr)==447066);
    end
    assert(isequal(double(prev.yVal(:)),yv), ...
        'Earlier checkpoint target ordering does not match.');

    % Train-only categorical level discovery: identical to Stage20/23.
    XT=X(tr,:);
    XV=X(va,:);
    for j=1:numel(catVars)
        v=catVars{j};
        trString=string(XT.(v));
        vaString=string(XV.(v));
        levels=unique(trString);
        XT.(v)=categorical(trString,levels);
        XV.(v)=categorical(vaString,levels);
        assert(~any(isundefined(XV.(v))), ...
            'New category in evaluation data: %s',v);
    end

    fprintf('\n===== %s =====\n',char(periodName(t)));
    fprintf('Train Jan 1-%d N=%d | Evaluate Jan %d-%d N=%d\n', ...
        trainEnd(t),sum(tr),evalStart(t),evalEnd(t),sum(va));
    [MTrain,MEval]=stage25SparseDesign(XT,XV,catVars);
    fprintf('Design: %d rows x %d columns, nnz=%d\n', ...
        size(MTrain,1),size(MTrain,2),nnz(MTrain));
    clear XT XV;

    tic;
    beta=MTrain\yt;
    assert(all(isfinite(beta)),'Initial OLS QR failed.');
    predPrev=double(MEval*beta);
    stepHistory=nan(12,1);
    scaleHistory=nan(12,1);
    floorHistory=nan(12,1);
    gradHistory=nan(12,1);

    for it=1:12
        r=yt-MTrain*beta;
        rmed=median(r);
        scale=max(1.4826*median(abs(r-rmed)),0.001);
        u=(r-rmed)/(4.685*scale);
        w=zeros(size(u));
        mask=abs(u)<1;
        w(mask)=(1-u(mask).^2).^2;
        w=max(w,1e-5);
        sw=sqrt(w);
        WTrain=spdiags(sw,0,numel(sw),numel(sw))*MTrain;
        yw=sw.*yt;
        betaNew=WTrain\yw;
        assert(all(isfinite(betaNew)), ...
            'Weighted sparse QR returned nonfinite coefficients.');
        predNew=double(MEval*betaNew);
        stepHistory(it)=max(abs(predNew-predPrev));
        scaleHistory(it)=scale;
        floorHistory(it)=100*mean(w<=1.0001e-5);
        gradHistory(it)=norm(WTrain'*(yw-WTrain*betaNew))/ ...
                        (norm(WTrain'*yw)+eps);
        beta=betaNew;
        predPrev=predNew;
        if it==6
            p6=predNew;
            beta6=betaNew; %#ok<NASGU>
        end
        fprintf('Pass %02d | scale %.4f | max step %.4f min | relative weighted gradient %.2g\n', ...
            it,scale,stepHistory(it),gradHistory(it));
    end
    elapsed=toc;
    p12=predPrev;
    P6{t}=p6;
    P12{t}=p12;
    passDetail{t}=table((1:12)',scaleHistory,stepHistory, ...
        floorHistory,gradHistory, ...
        'VariableNames',{'Pass','Scale','MaxPredictionStep_min', ...
        'FloorWeight_pct','RelativeWeightedGradient'});
    
    % Earlier models are frozen, not retrained.
    oldRobust=double(prev.P(:,21));
    oldLinear=double(prev.P(:,20));
    labels=["Robust Linear - previous LSQR"; ...
            "Robust Linear - 6 QR passes"; ...
            "Robust Linear - 12 QR passes diagnostic"; ...
            "Linear - old preset QR"];
    predictions=[oldRobust,p6,p12,oldLinear];
    for k=1:numel(labels)
        yp=predictions(:,k);
        err=yv-yp;
        nz=abs(yv)>eps;
        row=table(periodName(t),labels(k),numel(yv), ...
            sqrt(mean(err.^2)),mean(abs(err)),mean(err.^2), ...
            1-sum(err.^2)/sum((yv-mean(yv)).^2), ...
            mean(yp-yv),100*mean(abs(err)<=15), ...
            100*mean(abs(err(nz)./yv(nz))), ...
            'VariableNames',{'Period','Model','N', ...
            'RMSE_min','MAE_min','MSE_min2','R2', ...
            'Bias_min','Within15_pct','MAPE_nonzero_pct'});
        results=[results;row]; %#ok<AGROW>
    end

    dt=day(va);
    for d=evalStart(t):evalEnd(t)
        m=dt==d;
        er=yv(m)-p6(m);
        el=yv(m)-oldLinear(m);
        oneDay=table(periodName(t),d,sum(m), ...
            sqrt(mean(er.^2)),mean(abs(er)), ...
            sqrt(mean(el.^2)),mean(abs(el)), ...
            'VariableNames',{'Period','Day','N','Robust6_RMSE', ...
                'Robust6_MAE','Linear_RMSE','Linear_MAE'});
        Daily=[Daily;oneDay]; %#ok<AGROW>
    end
    fprintf('6-pass to 12-pass: median |prediction change| %.5f min, max %.4f min\n', ...
        median(abs(p12-p6)),max(abs(p12-p6)));
    fprintf('6-pass QR time, INCLUDING 12-pass diagnostic: %.2f s\n',elapsed);
    clear MTrain MEval WTrain beta betaNew w sw r u;
end

%% 3. Print corrected model metrics and daily performance
fprintf('\n============ STAGE 25 NUMERICAL AUDIT ============\n');
disp(results);
fprintf('\n============ DAILY HOLDOUT COMPARISON ============\n');
disp(Daily(Daily.Period=="Stage23 Holdout",:));

%% 4. Create a NEW PROPOSED Table 9 (do not overwrite previous version)
Table9=readtable(origTable,'TextType','string');
ix=find(string(Table9.Preset)=="Robust Linear");
assert(numel(ix)==1 && height(Table9)==23, ...
    'Original Table9 must have 23 models and one Robust Linear row.');
for t=1:2
    row=results.Period==periodName(t) & ...
        results.Model=="Robust Linear - 6 QR passes";
    assert(sum(row)==1);
    if t==1
        prefix='Dev_';
        nCol='Dev_NTrain';
        implementationCol='DevImplementation';
    else
        prefix='Test_';
        nCol='Test_NTrain';
        implementationCol='TestImplementation';
    end
    % Update only fields that actually exist; preserve all other 22 rows.
    replacements={ ...
        [prefix 'MAE_min'],results.MAE_min(row); ...
        [prefix 'MSE_min2'],results.MSE_min2(row); ...
        [prefix 'RMSE_min'],results.RMSE_min(row); ...
        [prefix 'R2'],results.R2(row); ...
        [prefix 'MAPE_nonzero_pct'],results.MAPE_nonzero_pct(row)};
    for j=1:size(replacements,1)
        v=replacements{j,1};
        if ismember(v,Table9.Properties.VariableNames)
            currentCol=Table9.(v);
            currentCol(ix)=replacements{j,2};
            Table9.(v)=currentCol;
        end
    end
    if ismember(nCol,Table9.Properties.VariableNames)
        currentCol=Table9.(nCol);
        currentCol(ix)=sum(day<=trainEnd(t));
        Table9.(nCol)=currentCol;
    end
    if ismember(implementationCol,Table9.Properties.VariableNames)
        currentCol=string(Table9.(implementationCol));
        currentCol(ix)= ...
            "Tukey bisquare robust sparse-QR IRLS; 6 passes (numerical correction)";
        Table9.(implementationCol)=currentCol;
    end
end

proposedCsv=fullfile(folderPath, ...
    'Stage25_Table9_23Presets_QRCorrected_PROPOSED.csv');
proposedXlsx=fullfile(folderPath, ...
    'Stage25_Table9_23Presets_QRCorrected_PROPOSED.xlsx');
writetable(Table9,proposedCsv);
try
    writetable(Table9,proposedXlsx);
catch ME
    warning('Excel output could not be saved: %s',ME.message);
end

%% 5. Export audit files
outMat=fullfile(folderPath,'Stage25_RobustLinear_QR_Audit.mat');
save(outMat,'results','Daily','P6','P12','passDetail', ...
    'trainEnd','evalStart','evalEnd','periodName', ...
    'sourcePath','-v7.3');
writetable(results,fullfile(folderPath, ...
    'Stage25_RobustLinear_Audit_Results.csv'));
writetable(Daily,fullfile(folderPath, ...
    'Stage25_RobustLinear_Daily.csv'));

fprintf('\n===== STAGE 25 COMPLETED =====\n');
fprintf('MAT: %s\n',outMat);
fprintf('Proposed revised Table9: %s\n',proposedCsv);
fprintf('IMPORTANT: Original Stage20/23 outputs are unchanged.\n');
fprintf('IMPORTANT: This is numerical correction, NOT new/blind holdout validation.\n');
fprintf('NOTE: MAPE remains unsuitable for zero/near-zero/negative delays.\n');

%% Local helper copied from Stage24/Stage23 feature construction.
function [A,AV]=stage25SparseDesign(XT,XV,categoriesVars)
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
            a=double(XT.(v));
            b=double(XV.(v));
            mu=mean(a); sd=std(a);
            if sd==0,sd=1;end
            A=[A sparse((a-mu)/sd)]; %#ok<AGROW>
            AV=[AV sparse((b-mu)/sd)]; %#ok<AGROW>
        end
    end
end
