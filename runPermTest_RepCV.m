function permTest = runPermTest_RepCV( feats, classY, subjID,  cvOpts, cvCombi_obs, permOpts)
%
% Permutation test for runRepeatedCV_TrainPLSLDA_PredictOutOfFold.
% Observed and null metrics are extracted via the same helper (extractStats),
% enforcing identical estimators/fields for both.
%
% Null: permute labels only; keep feats/subjID/cvOpts fixed.
% Note: wrapper reseeds rng() internally; we reseed just before randperm()
% at each permutation to avoid repeated permutations.

% ---- defaults ----
if nargin < 6 || isempty(permOpts), permOpts = struct(); end
if ~isfield(permOpts,'nPerm')     || isempty(permOpts.nPerm),     permOpts.nPerm = 1000;  end
if ~isfield(permOpts,'permSeed0') || isempty(permOpts.permSeed0), permOpts.permSeed0 = 12345; end
if ~isfield(permOpts,'dispEvery') || isempty(permOpts.dispEvery), permOpts.dispEvery = 50; end

% ---- observed statistic s----
permTest = struct();
permTest.obs = extractStats(cvCombi_obs);

% ---- null allocations ----
null = struct();
null.nPerm     = permOpts.nPerm;
null.permSeed0 = permOpts.permSeed0;

null.nClass = numel(cvCombi_obs.cvOpts.classTypes);
null.nConf  = null.nClass.^2;   

null.med_AROC        = nan(null.nPerm,1);
null.med_Acc         = nan(null.nPerm,1);
null.med_F1_posClass = nan(null.nPerm,1);
null.med_F1_negClass = nan(null.nPerm,1);
null.tstat           = nan(null.nPerm,1);

null.nValidReps      = nan(null.nPerm,1);
null.nTested         = nan(null.nPerm,1);

% store mean confusion matrix per permutation (Perm x Pred × True)
null.ConfMat_PredTrue = nan(null.nPerm, null.nClass, null.nClass);

% ---- permutations ----
for pp = 1:null.nPerm
    if pp==1 || mod(pp, permOpts.dispEvery)==0, disp(pp); end

    rng(null.permSeed0 + pp - 1);
    permClass = classY(randperm(numel(classY)));

    [cvCombi_perm, cvReps_perm] = runRepeatedCV_TrainPLSLDA_PredictOutOfFold( ...
        feats, permClass, subjID,  cvOpts);

    s = extractStats(cvCombi_perm);

    null.med_AROC(pp)        = s.med_AROC;
    null.med_Acc(pp)         = s.med_Acc;
    null.med_F1_posClass(pp) = s.med_F1_posClass;
    null.med_F1_negClass(pp) = s.med_F1_negClass;
    null.tstat(pp)           = s.tstat;

    null.ConfMat_PredTrue(pp,:,:) = cvCombi_perm.OverReps_Mean_ConfMat_PredTrue;

    null.nValidReps(pp) = sum(~isnan(cvReps_perm.Rep_AROC));
    null.nTested(pp)    = sum(cvCombi_perm.OverReps_HasAnyRep_OutOfFold);
end

permTest.null = null;

% ---- p-values (+1 correction; two-sided for tstat) ----
permTest.p = struct();
permTest.p.med_AROC        = (sum(null.med_AROC        > permTest.obs.med_AROC)        + 1) / (null.nPerm + 1);
permTest.p.med_Acc         = (sum(null.med_Acc         > permTest.obs.med_Acc)         + 1) / (null.nPerm + 1);
permTest.p.med_F1_posClass = (sum(null.med_F1_posClass > permTest.obs.med_F1_posClass) + 1) / (null.nPerm + 1);
permTest.p.med_F1_negClass = (sum(null.med_F1_negClass > permTest.obs.med_F1_negClass) + 1) / (null.nPerm + 1);
permTest.p.tstat           = (sum(abs(null.tstat) >= abs(permTest.obs.tstat)) + 1) / (null.nPerm + 1);

end

% ===== local helper: retrieve the median across runs of performancs metrics (aroc , acc, f1 for pos and neg class)
% ===== and the T value obtained from the (single) contrast of median values across runs of the out-of-fold predicted scores 
function s = extractStats(cvC)
s = struct();
s.med_AROC        = cvC.OverReps_MedIQR_AROC(1);
s.med_Acc         = cvC.OverReps_MedIQR_Acc(1);
s.med_F1_posClass = cvC.OverReps_MedIQR_F1_posClass(1);
s.med_F1_negClass = cvC.OverReps_MedIQR_F1_negClass(1);
s.tstat           = cvC.ttest_Score_OverReps_Med_OutOfFold.tstat;
end
