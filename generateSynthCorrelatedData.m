function dataSynth = generateSynthCorrelatedData(synth)
%
% Correlated synthetic data generator for binary classifier testing.
% This generator creates a binary classification dataset with:
% Gaussian features
% Controlled inter-feature correlation
% Controlled class effect size (mean shift only)
% It is designed to test classifiers under known ground truth, including null cases,
% without accidentally leaking label information through covariance, variance, or sample ordering.

% ---------------------- INPUT ----------------------
% synth : struct with fields (required unless stated otherwise)
%   .randSeed     scalar int  RNG seed for reproducibility. Same seed (and other params)=> identical dataset.
%   .nPerClass    scalar int     Number of samples per class (total N = 2 * nPerClass).
%   .nFeat        scalar int     Number of features (dimensions). Any nFeat >= 2 is supported.
%   .sigmaNoise   scalar > 0    Global noise scale. All effects are expressed in these units.
%   .rho          scalar in (-1, 1)
%       Inter-feature correlation coefficient for AR(1)/Toeplitz covariance.
%       rho = 0    -> independent features
%       rho > 0    -> smoothly correlated features
%
%   .effect       scalar >= 0      Class mean separation expressed in units of sigmaNoise.
%    If synth.effect == 0, features X are independent of labels Y  by construction (strict null).

% ---------------------- OUTPUT ----------------------
% dataSynth : struct
%   .feats     [N x nFeat] feature matrix
%   .class     [N x 1]     binary labels (0/1)
%   .subjID    [N x 1]     unique subject/sample identifiers
%   .featName  {1 x nFeat} cell array of feature names
%   .N         scalar     total number of samples
%
%% ---------------- DEFAULTS (minimal, null-safe) ----------------
if ~isfield(synth,'randSeed'),     synth.randSeed   = 0;   end
if ~isfield(synth,'sigmaNoise'),   synth.sigmaNoise = 1;  end
if ~isfield(synth,'rho'),          synth.rho        = 0;  end
if ~isfield(synth,'effect'),       synth.effect     = 0;  end
disp(synth)

req = {'nPerClass','nFeat'};
for k = 1:numel(req)
    if ~isfield(synth,req{k})
        error('Missing synth.%s', req{k});
    end
end

rng(synth.randSeed);

%% ---------------- LABELS ----------------
dataSynth = struct();
dataSynth.N = 2 * synth.nPerClass;
dataSynth.class  = [zeros(synth.nPerClass,1); ones(synth.nPerClass,1)];
dataSynth.subjID = (1:dataSynth.N)';

%% ---------------- CORRELATED NOISE ----------------
% Features are drawn from a multivariate normal distribution
% Feature covariance follows an AR(1) / Toeplitz structure
% AR(1) / Toeplitz covariance: Sigma(i,j) = rho^|i-j|
% Noise scale is controlled by sigmaNoise
% Critically: at this stage, features are independent of class labels by construction

idx   = 0:(synth.nFeat-1);
Sigma = synth.rho .^ abs(bsxfun(@minus, idx(:), idx(:)'));
R = chol(Sigma);

Z = randn(dataSynth.N, synth.nFeat);
X = (Z * R) * synth.sigmaNoise;   % label-independent noise

%% ---------------- OPTIONAL MEAN SHIFT ----------------
% (B) Optional class effect (pure mean shift)
% A class effect is introduced only as a mean displacement
% The shift occurs along a fixed unit direction in feature space
% The magnitude is expressed in units of the noise scale
% This ensures:
% -Effect size is dimensionally interpretable
% -Increasing nFeat does not inflate effect by accident
% No variance or covariance differences are introduced between classes

if synth.effect ~= 0
    v = ones(1, synth.nFeat);
    v = v / norm(v);              % unit direction (nFeat-invariant)

    delta = synth.effect * synth.sigmaNoise;
    mu0 = -0.5 * delta * v;
    mu1 =  0.5 * delta * v;

    X(dataSynth.class==0,:) = bsxfun(@plus, X(dataSynth.class==0,:), mu0);
    X(dataSynth.class==1,:) = bsxfun(@plus, X(dataSynth.class==1,:), mu1);
end

dataSynth.feats = X;

%% ---------------- FEATURE NAMES ----------------
for f = 1:synth.nFeat
    dataSynth.featName{f} = sprintf('Feat. %d',f);
end

%% ---------------- JOINT SHUFFLING ----------------
% Prevent ordering artefacts before CV
% (C) Post-generation shuffling:
% Rows are jointly permuted (features, labels, subject IDs)
% This removes ordering artifacts that can interact with CV fold assignment
% Specifically, cvpartition(...,'KFold') assigns folds using label order and RNG state; if samples
% are ordered (all class 0 then all class 1), some RNG seeds can generate fold layouts
% that interact with finite-sample correlation structure and inflate or deflate apparent performance under the null.
% statistical structure is preserved exactly as set before.
% This step is crucial for Stable cross-validation , Correct null calibration, Fair permutation testing

permIdx = randperm(dataSynth.N);
dataSynth.feats  = dataSynth.feats(permIdx,:);
dataSynth.class  = dataSynth.class(permIdx);
dataSynth.subjID = dataSynth.subjID(permIdx);

%% ---------------- SANITY ----------------
assert(size(dataSynth.feats,2) == synth.nFeat, 'Feature dimension mismatch.');

fprintf('Synthetic data: N=%d, nFeat=%d, rho=%.2f, effect=%.2f\n', ...
    dataSynth.N, synth.nFeat, synth.rho, synth.effect);

end
