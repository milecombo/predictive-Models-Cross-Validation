%% Correlated synthetic data for classifier testing (nFeat-general, effect-safe)

% This script demonstrates how to run and interpret the repeated-CV PLS–LDA classifier using synthetic data.
% The goal is twofold: (i) provide a usage example of the full pipeline (control of runs of CV, aggregating performance, permutation tests, plots),
% and (ii) verify internal consistency under known, controlled generative conditions, including the strict null case.

% Specifically, we generate correlated Gaussian features (Toeplitz / AR(1)) with optional between-class mean separation,
% visualize feature distributions, compute strictly out-of-fold predictions via repeated stratified K-fold CV,
% and quantify significance by comparing observed performance to an empirical null obtained by label permutations.
% The pipeline works for variable number of features (nFeat-general), compatible with older MATLAB versions, 
% and intended to expose (the absence of) spurious effects, data leakage, or miscalibration in both prediction and permutation testing.

%% CLEAR
% Clears workspace and command window to ensure a clean, reproducible run
clear; clc;

%% GENERATE SYNTHETIC DATA
% Controlled synthetic data generator:
% - balanced binary labels
% - optional class effect implemented as a pure mean shift (controlled by effect)
% - correlated Gaussian features with AR(1)/Toeplitz covariance (controlled by rho)
%
% Features are generated with an AR(1)/Toeplitz covariance (Sigma_ij = rho^|i-j|),
% implying strong correlation between adjacent features and exponentially decaying correlation with increasing feature distance.
%  As a consequence, neighboring features carry partially redundant information and feature importance measures (e.g., VIP)
% may be distributed across correlated predictors rather than concentrated on a single feature.
%
% NOTE: With effect = 0, features and labels are independent by construction; therefore, as sample size N tends to ?
%  any consistent classifier will converge to chance-level performance,  while finite-N runs may fluctuate.
%  Permutation test should show that under no effect the behaviour is within chance-level
% ---------------- PARAMETERS FOR SYNTHETIC DATA --------------
synth=[];
synth.randSeed   =   0; %10000 100 randi(1e9);  %    6.7109
synth.nPerClass  = 500;     % samples per class
synth.nFeat      = 50;      % any N > 2 works
synth.sigmaNoise = 1.0;     % overall noise scale
synth.rho        = 0.5;     %   feature correlation (AR(1) Toeplitz),  
% synth.effect     = 1.5;     % class mean separation in units of sigmaNoise (0 => no effect)
synth.effect     = 0;     % class mean separation in units of sigmaNoise (0 => no effect)
% ----------------------------------------------------------------------------
dataSynth = generateSynthCorrelatedData(synth);


%% PLOT FEATURES and CLASS
% Optional diagnostic: inspect class overlap and feature correlations before running the classifier
 plotFeaturesClass_univariate(dataSynth.feats, dataSynth.class, dataSynth.featName)
 
%%  PREDICTION WITH RESAMPLINGS + PLOT
% % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CORE: repeated, stratified K-fold CV producing strictly out-of-fold predictions.
% In each fold, the model is fit on TRAIN only, with a PLS +LDA strategy:
% a predictive PLS1 model (Partial Least Square with 1 component, PLS) leads to continuous PLS scores 
% and a 1D diag-linear Linear Discriminant Analysis classifier binarizes scores (trained on TRAIN data only),
% with optional subject-wise leakage control (exclude from TRAIN any subject appearing in TEST)
% and z-scoring computed from TRAIN statistics only.
% Out-of-fold scores/predictions are stored in the original sample coordinates and then aggregated across repetitions
% (medians/IQR, median ROC on a fixed FPR grid), yielding a robust observed-performance summary.

% ---------------- PARAMETERS FOR REPEATED CROSS-VALIDATION of prediction + ----------------
cvOpts =[];
cvOpts.numCvReps = 10;
cvOpts.numCvFolds= 10;
cvOpts.avoidTrainOnTestSubj = 1; % subject-wise leakage control (in this example is irrelevant, each subject appears once)
cvOpts.zscoreFromTrain     =1;      % normalize using TRAIN stats only
cvOpts.rngSeed  =    1;   %  seed for CV splits (set fixed for exact reproducibility) or random to change splitting. randi(1e9)

cvOpts.classTypes           = [0 1];    % enforce order
cvOpts.posClass             = 1;
cvOpts.priorType= 'uniform';

% Repeated K-fold cross-validation (CV) for training and evaluating PLS-LDA model:
[cvCombined, cvReps] = runRepeatedCV_TrainPLSLDA_PredictOutOfFold( ...
     dataSynth.feats, dataSynth.class, dataSynth.subjID, cvOpts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot summary of PLS prediction and LDA classification results across CV repetitions
plsPlot = plotPLS_CV_Summary(cvCombined, cvReps , 'featName', plsPlot.featName);

%% SIGNIFICANCE VIA PERMUTATIONS
% assess significance  of observed (real) performance, against empirical null distribution
% i.e. get p values from random permutations
permOpts.nPerm= 1000;
permT = runPermTest_RepCV( dataSynth.feats, dataSynth.class, dataSynth.subjID,  cvOpts, cvCombined, permOpts);

%% PLOT REAL vs NULL (PERMUTATION) RESULTS
% plot comparison of observed (real) performance vs distribution of null results from permutations
plt = plotPermTest_RepCV(permT, cvCombined);





%%
% cvOpts0=cvOpts;
% cvOpts0.numCvReps= 2;
% permT = runPermTest_RepCV( dataSynth.feats, dataSynth.class, dataSynth.subjID,  cvOpts0, cvCombined, permOpts);
