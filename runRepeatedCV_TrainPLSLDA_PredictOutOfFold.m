function [cvCombi, cvReps] = runRepeatedCV_TrainPLSLDA_PredictOutOfFold(XFeats, YLabel, subjID, cvOpts)
% Repeated K-fold cross-validation (CV) wrapper for training and evaluating a model:
%
% This function is a high-level wrapper that controls the repeated cross-validation workflow and is responsible for:
%   - generating repeated K-fold partitions,
%   - collecting strictly out-of-fold predictions across folds and repetitions,
%   - computing and storing performance metrics and summary statistics.
%
% The actual model fitting and prediction logic is fully delegated to the core function:
%   trainModel_PredictTestSet(Fold_Train_XFeats, Fold_Train_YLabel, Fold_Test_XFeats, priorType)
% The core function is invoked once per fold within each repetition:
%   - the model is trained using only the fold-specific training data,
%   - predictions are generated exclusively on the corresponding held-out test set.
%
% This function supports binary classification (two classes) and produces key performance metrics.
% For multi-class problems, the metrics are computed using one-vs-all strategies.
%
% Naming convention: coordinate system first (Rep/Fold/Train/Test/OutOfFold).
%   Rep        = repetition index (outer repeated CV loop)
%   Fold       = fold index within a repetition
%   Train/Test = data partition within a fold
%   OutOfFold  = samples never seen during training for that prediction
%
% ---------------------- INPUTS ----------------------
% XFeats        [N×P]  predictors/features (rows = samples/recordings; cols = features)
% YLabel        [N×1]  binary labels (must match cvOpts.classTypes; e.g., 0/1)
% subjID        [N×1]  subject identifier used for C2 leakage control
% cvOpts        struct (optional):
%   priorType     char/string  'uniform' or 'empirical' (used for classifier configuration)
%   .numCvReps            scalar int, default 10   - number of cross-validation repetitions
%   .numCvFolds           scalar int, default 10   - number of folds per CV repetition
%   .rngSeed              scalar int, default 1    - seed for random number generator
%   .avoidTrainOnTestSubj scalar {0,1}, default 1  - controls subject leakage prevention (train/test subject separation)
%   .zscoreFromTrain      scalar {0,1}, default 1  - z-score normalization based on training data
%   .classTypes           1×K vector, default sort(unique(YLabel)) - class labels (binary or multi-class)
%   .posClass             scalar, default 1        - positive class label for AUC/F1 computation
%
% ---------------------- OUTPUTS ----------------------
% cvReps: repetition-level outputs (reps on rows; includes Rep dimension)
%   cvReps.Sample_TrueYLabel                [1×N]   ground truth labels for all samples
%   cvReps.Sample_SubjID                    [1×N]   subject identifiers for all samples
%   cvReps.RepSample_OutOfFold_Score        [Rep×N] continuous model scores for each sample (NaN if not tested)
%   cvReps.RepSample_OutOfFold_PredYLabel   [Rep×N] predicted labels for each sample (NaN if not tested)
%   cvReps.RepSample_OutOfFold_TestMask     [Rep×N] logical mask indicating if sample was tested in each rep
%   cvReps.Rep_ConfMat_TruePred             [Rep×K×K] confusion matrix per rep (True×Pred, K=number of classes)
%   cvReps.Rep_AROC                         [Rep×1] AROC score per repetition
%   cvReps.Rep_Acc                          [Rep×1] accuracy score per repetition
%   cvReps.Rep_F1_posClass                  [Rep×1] F1 score for positive class (NaN for non-binary cases)
%   cvReps.Rep_F1_negClass                  [Rep×1] F1 score for negative class (NaN for non-binary cases)
%   cvReps.RepFold_VipScores                [Rep×Fold×P] VIP scores per fold (derived from training data)
%   cvReps.RepFold_OptRocPoint              [Rep×Fold×2] Optimal ROC operating point per fold
%   cvReps.Rep_Vip_MeanAcrossFolds          [Rep×P] Mean VIP scores across all folds within a repetition
%---------   ROC points at each run. Grid approach:
%   cvReps.RepRoc_FPRgrid                   [1×G]   fixed FPR grid used to interpolate ROC curves
%   cvReps.RepRoc_TPRgrid                   [Rep×G] TPR values interpolated onto fixed FPR grid (safe for averaging)
%---------   ROC points at each run. Unique FPR point approach:
%   cvReps.RepRoc_nRocPts                   [Rep×1] number of unique ROC points returned by perfcurve per repetition
%   cvReps.RepRoc_FPRu                      [Rep×Lmax] unique FPR coordinates per rep (NaN-padded)
%   cvReps.RepRoc_TPRu                      [Rep×Lmax] TPR values at unique FPR coordinates per rep (NaN-padded)
%   cvReps.RepRoc_Thru                      [Rep×Lmax] decision thresholds corresponding to unique ROC points (NaN-padded)

% cvCombi: aggregated outputs ONLY (no Rep dimension)
%   cvCombi.OverReps_Med_OutOfFold_Score   [N×1]  median score over reps for each sample (computed via prctile)
%   cvCombi.OverReps_HasAnyRep_OutOfFold   [N×1]  logical vector: true if sample was tested at least once across reps
%   cvCombi.OverReps_MedIQR_AROC                   [1×3]   [50th, 25th, 75th] percentiles of AROC across reps
%   cvCombi.OverReps_MedIQR_Acc                    [1×3]   [50th, 25th, 75th] percentiles of accuracy across reps
%   cvCombi.OverReps_MedIQR_F1                     [1×3]   [50th, 25th, 75th] percentiles of F1 score for positive class across reps
%   cvCombi.OverReps_MedIQR_F1_negClass            [1×3]   [50th, 25th, 75th] percentiles of F1 score for negative class across reps
%   cvCombi.OverReps_MedIQR_Vip                    [3×P]   VIP score percentiles over reps (mean, 25th, 75th)
%   cvCombi.OverReps_Mean_ConfMat_PredTrue         [K×K]   mean confusion matrix (Pred×True), transposed for plotting consistency
%   cvCombi.OverReps_MeanPct_ConfMat_PredTrue      [K×K]   column-normalized confusion matrix (% of TRUE class)

%% ---------------------- Crucial Logic----------------------
% 1) OutOfFold_*: These variables hold predictions in global sample coordinates. 
%    They are populated only for the test fold in each repetition.
%    This guarantees no leakage between training and testing samples.
% 2) Subject leakage control (optional) ensures that samples from the same subject are not used in both training and testing sets for the same fold.
% 3) The confusion matrix is stored as True×Pred. Metrics like Accuracy and F1 are computed from this True×Pred matrix.
%    For non-binary cases, F1 scores remain NaN, and only Accuracy is calculated.
% 4) The AROC (Area under the Receiver Operating Characteristic curve) is computed if both classes are present in the test set for each repetition.
%    This is applicable to binary classification only.

% For multi-class classification, metrics are computed using one-vs-all strategies, including AROC for each class against the rest.

%% ---------------------- ROC storage (stored both as unique points and with fixed-grid) ----------------------
% function *perfcurve* returns a stepwise ROC obtained by sweeping a threshold over the SCORE ranking.
%  Conceptually: each time the threshold passes a Negative-class sample, FPR increases; each time it passes a Positive sample, TPR increases.
%  Given a set of FPR points, the corresponding TPR (ROC y-values) depend on the model quality: they reflect where positives fall in the SCORE ranking (earlier positives => higher TPR at low FPR).
%  To compare/average ROC curves across Repetitions (aka Runs) we need a common x-axis (FPR points) with fixed length.

% 1) UNIQUE-POINT ROC (RepRoc_FPRu / RepRoc_TPRu / RepRoc_Thru)
%    - The set of FPR x-values is mainly determined by:
%        (a) the number of negatives in the evaluated set (FPR step = 1/Nneg),
%        (b) the presence of SCORE ties (discrete/quantized scores collapse steps).
%       Hence, as typical, if ROC is computed on pooled OOF (out-of-folds) predictions 
%       that include the same samples at each rep (eg all the dataset is used in each rep), and if scores are mostly continuous,
%       the resulting unique FPR values (RepRoc_FPRu) are identical across reps (often with L ? Nneg+1), allowing to average ROC points across reps.

% 2) FIXED FPR GRID ROC (RepRoc_TPRgrid on RepRoc_FPRgrid)
%    - Tobe able to always compare/average ROC curves across reps, we also ensure a common x-axis with fixed length.
%      We therefore interpolate TPR onto a predefined FPR grid (RepRoc_FPRgrid) constant across reps.
%    - Note that, in the approach 1) , unique-point ROC curves can differ across reps (making direct averaging unsafe) when:
%        • the evaluated set or class counts change (dropouts, per-fold ROC, missing class), or
%        • SCORE ties differ across reps (discrete posteriors, rounding, hard labels).

%% -------------------- basic checks --------------------
if nargin < 5 || isempty(cvOpts), cvOpts = struct(); end
YLabel = YLabel(:);
subjID = subjID(:);

N = size(XFeats,1);
P = size(XFeats,2);

if numel(YLabel) ~= N || numel(subjID) ~= N
    error('XFeats, YLabel, subjID must have compatible lengths. Got N=%d, numel(YLabel)=%d, numel(subjID)=%d.', ...
        N, numel(YLabel), numel(subjID));
end

%% -------------------- defaults --------------------
if ~isfield(cvOpts,'numCvReps'),            cvOpts.numCvReps = 10; end
if ~isfield(cvOpts,'numCvFolds'),           cvOpts.numCvFolds = 10; end
if ~isfield(cvOpts,'rngSeed'),              cvOpts.rngSeed = 1; end
if ~isfield(cvOpts,'avoidTrainOnTestSubj'), cvOpts.avoidTrainOnTestSubj = 1; end
if ~isfield(cvOpts,'zscoreFromTrain'),      cvOpts.zscoreFromTrain = 1; end
if ~isfield(cvOpts,'classTypes'),           cvOpts.classTypes = sort(unique(YLabel)); end
if ~isfield(cvOpts,'posClass'),             cvOpts.posClass = 1; end
if ~isfield(cvOpts,'priorType') || isempty(cvOpts.priorType)
    cvOpts.priorType = 'uniform'; %% alt: empirical
end

numCvReps  = cvOpts.numCvReps;
numCvFolds = cvOpts.numCvFolds;
classTypes = cvOpts.classTypes(:).';
posClass   = cvOpts.posClass;

nClas = numel(classTypes);

%% -------------------- initialize cvReps (reps on rows) --------------------
cvReps = struct();
cvReps.cvOpts = cvOpts;

cvReps.Sample_N         = N;
cvReps.Sample_P         = P;
cvReps.Sample_TrueYLabel = YLabel.'; % 1×N
cvReps.Sample_SubjID     = subjID.'; % 1×N

% OutOfFold stored as Rep × Sample
cvReps.RepSample_OutOfFold_Score      = nan(numCvReps, N);
cvReps.RepSample_OutOfFold_PredYLabel = nan(numCvReps, N);
cvReps.RepSample_OutOfFold_TestMask   = false(numCvReps, N);

% Rep-level performance (pooled OutOfFold samples within rep)
cvReps.Rep_ConfMat_TruePred = nan(numCvReps, nClas, nClas); % True×Pred
cvReps.Rep_AROC              = nan(numCvReps, 1);
cvReps.Rep_Acc              = nan(numCvReps, 1);
cvReps.Rep_F1_posClass       = nan(numCvReps, 1); % binary only
cvReps.Rep_F1_negClass       = nan(numCvReps, 1); % binary only

% Fold-level TRAIN-derived quantities
cvReps.RepFold_VipScores    = nan(numCvReps, numCvFolds, P);
cvReps.RepFold_OptRocPoint  = nan(numCvReps, numCvFolds, 2);

% Within-rep VIP summary
cvReps.Rep_Vip_MeanAcrossFolds = nan(numCvReps, P);

% roc curve at fixed intervals
cvReps.RepRoc_FPRgrid = linspace(0,1,101);                 % 1×G % eval the TPR (i.e. ROC curve y points) at these FPR points
cvReps.RepRoc_TPRgrid = nan(numCvReps, numel(cvReps.RepRoc_FPRgrid)); % TPR corresponding at these FPR points

% roc curve at unique intervals (could potentially change length acrossruns) NaN-pad to N+1 worst case
% Lmax = N + 1;                                                  % large, extra safe upper bound
Nneg = sum(YLabel ~= posClass);
Lmax = Nneg + 1;
cvReps.RepRoc_FPRu = nan(numCvReps, Lmax); % Rep_FPRu contains only unique FPR values.
cvReps.RepRoc_TPRu = nan(numCvReps, Lmax); % TPRu values are stored at those unique values taken from FPR 
cvReps.RepRoc_Thru = nan(numCvReps, Lmax);   % same for thresholds
cvReps.RepRoc_nRocPts  = zeros(numCvReps, 1);                  % how many valid pts in each row
%% -------------------- repeated CV --------------------
for RepIdx = 1:numCvReps
    
    rng(cvOpts.rngSeed + RepIdx - 1); % seed for reapeatibility. comment out if doing shuffled-permutation or bootstraps!
    
    % Stratified K-fold at SAMPLE level
    cv = cvpartition(YLabel, 'KFold', numCvFolds);
    
    % Per-rep OutOfFold containers in GLOBAL sample space (N×1)
    OutOfFold_Score      = nan(N,1);
    OutOfFold_PredYLabel = nan(N,1);
    
    % Per-rep fold containers
    RepFold_VipScores   = nan(numCvFolds, P);
    RepFold_OptRocPoint = nan(numCvFolds, 2);
    
    for FoldIdx = 1:numCvFolds
        
        Fold_isTrain = training(cv, FoldIdx);
        Fold_isTest  = test(cv, FoldIdx);
        
        %=============================================================
        % C2) Subject-wise leakage control:
        % remove from TRAIN any subject that appears in TEST
        %=============================================================
        if cvOpts.avoidTrainOnTestSubj == 1
            Fold_Test_Subj = unique(subjID(Fold_isTest));
            Fold_isTrain   = Fold_isTrain & ~ismember(subjID, Fold_Test_Subj);
        end
        
        % Fold datasets
        Fold_Train_XFeats = XFeats(Fold_isTrain, :);
        Fold_Train_YLabel = double(YLabel(Fold_isTrain));
        
        Fold_Test_XFeats  = XFeats(Fold_isTest, :);
        Fold_Test_Idx     = find(Fold_isTest);
        if isempty(Fold_Test_Idx)
            continue;
        end
        
        % Z-score from TRAIN only
        if cvOpts.zscoreFromTrain == 1
            Fold_Train_mu = mean(Fold_Train_XFeats, 1);
            Fold_Train_sd = std(Fold_Train_XFeats, [], 1);
            Fold_Train_sd(Fold_Train_sd==0) = 1;
            
            Fold_Train_XFeats = bsxfun(@rdivide, bsxfun(@minus, Fold_Train_XFeats, Fold_Train_mu), Fold_Train_sd);
            Fold_Test_XFeats  = bsxfun(@rdivide, bsxfun(@minus, Fold_Test_XFeats,  Fold_Train_mu), Fold_Train_sd);
        end
        
        % Guard against degenerate training folds after C2 purge
        if isempty(Fold_Train_YLabel) || numel(unique(Fold_Train_YLabel)) < 2
             disp('empty fold or unique class'); continue;
        end
        
        %=============================================================
        % Core model call (TRAIN only; PREDICT on TEST only)
        %=============================================================
        [Fold_Test_PlsScore, Fold_Test_PredYLabel, Fold_VipScores, Fold_OptRocPoint] = ...
            trainModel_PredictTestSet(Fold_Train_XFeats, Fold_Train_YLabel, Fold_Test_XFeats, cvOpts.priorType);
        % For each repetition and each fold, the model is trained on the fold-specific training set only
         % and then applied to the corresponding held-out test set.
        % Predictions are collected out-of-fold, so every test sample is evaluated without ever being seen during its own training.
        %=============================================================
        
        if numel(Fold_Test_PlsScore) ~= numel(Fold_Test_Idx)
            error('Size mismatch: numel(Fold_Test_PlsScore)=%d but numel(Fold_Test_Idx)=%d', ...
                numel(Fold_Test_PlsScore), numel(Fold_Test_Idx));
        end
        
        % Safety: each sample filled at most once per rep
        if any(~isnan(OutOfFold_Score(Fold_Test_Idx)))
            error('OutOfFold fill collision in RepIdx=%d: some Fold_Test_Idx already filled.', RepIdx);
        end
        
        % Write into global index space (pools across folds within rep)
        OutOfFold_Score(Fold_Test_Idx)      = Fold_Test_PlsScore(:);
        OutOfFold_PredYLabel(Fold_Test_Idx) = Fold_Test_PredYLabel(:);
        
        % Store TRAIN-derived fold quantities
        RepFold_VipScores(FoldIdx,:)   = Fold_VipScores(:).';
        RepFold_OptRocPoint(FoldIdx,:) = Fold_OptRocPoint(:).';
    end
    
    % -------------------- store OutOfFold vectors: Rep × Sample --------------------
    OutOfFold_TestMask = ~isnan(OutOfFold_Score);
    
    cvReps.RepSample_OutOfFold_Score(RepIdx,:)      = OutOfFold_Score(:).';
    cvReps.RepSample_OutOfFold_PredYLabel(RepIdx,:) = OutOfFold_PredYLabel(:).';
    cvReps.RepSample_OutOfFold_TestMask(RepIdx,:)   = OutOfFold_TestMask(:).';
    
    % -------------------- per-rep performance on pooled OutOfFold samples --------------------
    if any(OutOfFold_TestMask)
        
        FoldsPool_True      = double(YLabel(OutOfFold_TestMask));
        FoldsPool_PredYLabel = double(OutOfFold_PredYLabel(OutOfFold_TestMask));
        FoldsPool_Score     = double(OutOfFold_Score(OutOfFold_TestMask));
        
        % Confusion matrix stored as True×Pred (rows=True, cols=Pred)
        Rep_ConfMat_TruePred = confusionmat(FoldsPool_True, FoldsPool_PredYLabel, 'order', classTypes);
        cvReps.Rep_ConfMat_TruePred(RepIdx,:,:) = Rep_ConfMat_TruePred;
        
        % Binary-smart dense metrics (Acc + F1s). If not binary -> F1s remain NaN.
        [Rep_Acc, Rep_F1_posClass, Rep_F1_negClass] = ...
            metricsFromConfMatBinary_TruePred(Rep_ConfMat_TruePred, classTypes, posClass);
        
        % Always store accuracy (if matrix has counts). For non-binary, helper still returns Acc properly.
        cvReps.Rep_Acc(RepIdx)           = Rep_Acc;
        cvReps.Rep_F1_posClass(RepIdx)    = Rep_F1_posClass;
        cvReps.Rep_F1_negClass(RepIdx) = Rep_F1_negClass;
        
        % AROC (defined only if both classes present in tested subset)
        if numel(unique(FoldsPool_True)) > 1
        [Rep_FPR, Rep_TPR, Rep_Thr, Rep_AROC] = perfcurve(FoldsPool_True, FoldsPool_Score, posClass);
        %         [~, Rep_TPRgrid, ~, Rep_AROC] = perfcurve(FoldsPool_True, FoldsPool_Score, posClass,  'XVals', cvReps.RepRoc_FPRgrid); % this seems easier but does not guarantees fixed length
        
        % We first make FPR unique (and keep stable order, monotone) to ensure interp1 stability
        [Rep_FPRu, ia] = unique(Rep_FPR(:), 'stable');
        Rep_TPRu = Rep_TPR(ia);
        Rep_Thru = Rep_Thr(ia);
        % Interpolate onto common grid (guaranteed fixed length)
        
        Rep_TPRgrid = interp1(Rep_FPRu, Rep_TPRu, cvReps.RepRoc_FPRgrid(:), 'linear', 'extrap');
        else
            Rep_AROC = nan;     Rep_TPRgrid = nan(size(cvReps.RepRoc_FPRgrid));
        end
        cvReps.Rep_AROC(RepIdx) = Rep_AROC;
        cvReps.RepRoc_TPRgrid(RepIdx,:) = Rep_TPRgrid(:).';

        % store unique FPR and TPR (NaN-padded)
        L = numel(Rep_FPRu);
        cvReps.RepRoc_nRocPts(RepIdx) = L;
        cvReps.RepRoc_FPRu(RepIdx,1:L)  = Rep_FPRu(:).';
        cvReps.RepRoc_TPRu(RepIdx,1:L)  = Rep_TPRu(:).';
        cvReps.RepRoc_Thru(RepIdx,1:L) = Rep_Thru(:).';
end
    
    % -------------------- fold-level outputs --------------------
    cvReps.RepFold_VipScores(RepIdx,:,:)   = RepFold_VipScores;
    cvReps.RepFold_OptRocPoint(RepIdx,:,:) = RepFold_OptRocPoint;
    
    % VIP mean across folds within Rep (NaN-robust)
    cvReps.Rep_Vip_MeanAcrossFolds(RepIdx,:) = nanAverage(RepFold_VipScores, 1);
end

%% ======================================================================
% cvCombi: aggregated outputs ONLY (no Rep dimension)
%% ======================================================================
cvCombi = struct();
cvCombi.cvOpts = cvOpts;
cvCombi.Sample_TrueYLabel = YLabel; % N×1
cvCombi.Sample_SubjID     = subjID; % N×1

% Median over reps per sample (via prctile; your display vector for boxplots/T-test)
cvCombi.OverReps_Med_OutOfFold_Score = prctile(cvReps.RepSample_OutOfFold_Score, 50, 1).'; % N×1

% Tested at least once over reps
cvCombi.OverReps_HasAnyRep_OutOfFold = any(~isnan(cvReps.RepSample_OutOfFold_Score), 1).'; % N×1

% Performance summaries reported as median + IQR over reps
cvCombi.OverReps_MedIQR_AROC            = prctile(cvReps.Rep_AROC,           [50 25 75]);
cvCombi.OverReps_MedIQR_Acc            = prctile(cvReps.Rep_Acc,           [50 25 75]);
cvCombi.OverReps_MedIQR_F1_posClass             = prctile(cvReps.Rep_F1_posClass,   [50 25 75]);
cvCombi.OverReps_MedIQR_F1_negClass    = prctile(cvReps.Rep_F1_negClass,   [50 25 75]);

% VIP summary over reps (rep-level VIP already averaged across folds)
cvCombi.OverReps_MedIQR_Vip = prctile(cvReps.Rep_Vip_MeanAcrossFolds, [50 25 75], 1); % 3×P

% Confusion matrix averaged over reps (legacy plotting convention Pred×True)
OverReps_Mean_ConfMat_TruePred = squeeze(nanAverage(cvReps.Rep_ConfMat_TruePred, 1)); % True×Pred
cvCombi.OverReps_Mean_ConfMat_PredTrue = OverReps_Mean_ConfMat_TruePred.';            % Pred×True

% Percent of REAL class (REAL=True). With Pred×True, normalize each column.
colSum = sum(cvCombi.OverReps_Mean_ConfMat_PredTrue, 1);
colSum(colSum==0) = NaN;
cvCombi.OverReps_MeanPct_ConfMat_PredTrue = 100 * bsxfun(@rdivide, cvCombi.OverReps_Mean_ConfMat_PredTrue, colSum);

% ROC curves: median TPR across runs, given fixed FPR points (grid intervals)
cvCombi.OverReps_Roc_FPRgrid = cvReps.RepRoc_FPRgrid(:);
cvCombi.OverReps_Roc_TPRgrid = prctile(cvReps.RepRoc_TPRgrid, 50, 1).';

%%---- t test statistic on the predicted pls scores (from their median across runs) contrast between classes
scores= cvCombi.OverReps_Med_OutOfFold_Score; 
c0= cvCombi.Sample_TrueYLabel== min(cvCombi.Sample_TrueYLabel);
c1= cvCombi.Sample_TrueYLabel== max(cvCombi.Sample_TrueYLabel);
[H,P,CI,STATS] = ttest2( scores(c0), scores(c1));
STATS.Pval = P; % NOTE: do not interpret this P !
STATS.Hypothesis = H;
cvCombi.ttest_Score_OverReps_Med_OutOfFold= STATS;
% “median OOF scores per sample across reps” are not i.i.d. observations in a classical sense:
% each sample’s score is produced by many overlapping training sets across reps,
% so the induced dependence structure prevents from using this P value.
end


%=============================================================
function m = nanAverage(x, dim)
% nanAverage(x,dim) - mean ignoring NaNs, toolbox-independent, works on N-D arrays
if nargin < 2, dim = 1; end
x = double(x);
n = sum(~isnan(x), dim);
x(isnan(x)) = 0;
m = sum(x, dim) ./ n;
m(n==0) = NaN;
end

%=============================================================
function [Acc, F1_PosClass, F1_OtherClass] = metricsFromConfMatBinary_TruePred(C_TruePred, classTypes, posClass)
% metricsFromConfMatBinary_TruePred
% Dense, binary-smart metrics from confusion matrix with rows=True, cols=Pred.

Acc = NaN;
F1_PosClass = NaN;
F1_OtherClass = NaN;

% ---- Accuracy (works whenever matrix has counts) ----
if ~ismatrix(C_TruePred) || size(C_TruePred,1) ~= size(C_TruePred,2)
    return;
end
tot = sum(C_TruePred(:));
if tot > 0
    Acc = sum(diag(C_TruePred)) / tot;
end

% ---- Binary F1s only ----
if numel(classTypes) ~= 2 || ~isequal(size(C_TruePred), [2 2])
    return;
end

posIdx = find(ismember(classTypes, posClass), 1);
if isempty(posIdx)
    error('posClass=%g not found in classTypes.', posClass);
end
negIdx = 3 - posIdx;

% Positive class F1
TP = C_TruePred(posIdx, posIdx);
FP = C_TruePred(negIdx, posIdx);
FN = C_TruePred(posIdx, negIdx);
den = 2*TP + FP + FN;
if den > 0
    F1_PosClass = 2*TP / den;
end

% Other class treated as positive
TPn = C_TruePred(negIdx, negIdx);
FPn = C_TruePred(posIdx, negIdx);
FNn = C_TruePred(negIdx, posIdx);
denn = 2*TPn + FPn + FNn;
if denn > 0
    F1_OtherClass = 2*TPn / denn;
end

end

%%


%%
% % Explanation of Key Variables:
% % Inputs:
% % 
% % XFeats: The matrix of features (predictors) for all samples, with each row corresponding to a sample and each column corresponding to a feature.
% % 
% % YLabel: The labels (binary or multi-class) for each sample. Must match cvOpts.classTypes.
% % 
% % subjID: A vector of subject IDs used for controlling leakage in cross-validation, ensuring that no subject appears in both training and testing sets in the same fold.
% % 
% % priorType: A string that specifies the prior distribution type for training. Can be 'uniform' or 'empirical'.
% % 
% % cvOpts: A struct containing the cross-validation options:
% % 
% % numCvReps: Number of cross-validation repetitions.
% % 
% % numCvFolds: Number of folds for each repetition.
% % 
% % rngSeed: Seed for the random number generator.
% % 
% % avoidTrainOnTestSubj: Flag for controlling subject leakage.
% % 
% % zscoreFromTrain: Flag for z-score normalization using training data.
% % 
% % classTypes: A vector of class labels.
% % 
% % posClass: The label of the positive class for AROC/F1 calculations.
% % 
% % Outputs:
% % 
% % cvReps: A struct containing repetition-level results:
% % 
% % RepSample_OutOfFold_Score: The out-of-fold predictions (continuous scores) for each sample across repetitions.
% % 
% % RepSample_OutOfFold_PredYLabel: The predicted labels for each sample across repetitions.
% % 
% % RepSample_OutOfFold_TestMask: A mask indicating whether a sample was tested in each repetition.
% % 
% % Rep_ConfMat_TruePred: The confusion matrix for each repetition.
% % 
% % Rep_AROC: The AROC score for each repetition.
% % 
% % Rep_Acc: The accuracy for each repetition.
% % 
% % Rep_F1_posClass: The F1 score for the positive class.
% % 
% % Rep_F1_negClass: The F1 score for the negative class.
% % 
% % RepFold_VipScores: The VIP scores for each fold in each repetition.
% % 
% % RepFold_OptRocPoint: The optimal ROC operating point for each fold in each repetition.
% % 
% % Rep_Vip_MeanAcrossFolds: The mean VIP scores across all folds within a repetition.
% % 
% % cvCombi: A struct containing aggregated outputs (no repetition dimension):
% % 
% % OverReps_Med_OutOfFold_Score: The median out-of-fold scores for each sample across repetitions.
% % 
% % OverReps_HasAnyRep_OutOfFold: A logical vector indicating whether a sample was tested at least once across repetitions.
% % 
% % OverReps_MedIQR_AROC: The median and IQR (25th, 75th percentiles) for AROC scores across repetitions.
% % 
% % OverReps_MedIQR_Acc: The median and IQR for accuracy scores across repetitions.
% % 
% % OverReps_MedIQR_F1: The median and IQR for F1 scores for the positive class across repetitions.
% % 
% % OverReps_MedIQR_F1_negClass: The median and IQR for F1 scores for the negative class across repetitions.
% % 
% % OverReps_MedIQR_Vip: The median and IQR for VIP scores across repetitions.
% % 
% % OverReps_Mean_ConfMat_PredTrue: The average confusion matrix across repetitions (predicted vs true labels).
% % 
% % OverReps_MeanPct_ConfMat_PredTrue: The column-normalized confusion matrix (percent of TRUE class).