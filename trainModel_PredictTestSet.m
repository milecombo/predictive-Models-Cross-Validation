function [Test_plsScore, Test_predYLabel, vipScores, optRocPoint] = ...
         trainModel_PredictTestSet(Train_XFeats, Train_YLabel, Test_XFeats, priorType)
% TRAINMODEL_PREDICTTESTSET
%
% Fits a one-component PLS1 model on TRAINING data only, derives the LDA
% decision threshold from TRAINING data only, and applies it to TEST data.
% The function NEVER sees test labels.
%
% INPUTS:
%   Train_XFeats     [Ntrain × P] predictors (training)
%   Train_YLabel     [Ntrain × 1] binary labels (training) expected 0/1 (logical ok)
%   Test_XFeats      [Ntest × P]  predictors (test), no labels provided
%   priorType  LDA prior ('uniform' or 'empirical')
%
% OUTPUTS:
%   Test_plsScore   continuous PLS1 scores for Test_XFeats
%   Test_predYLabel  LDA-derived binary predicted labels for Test_XFeats
%   vipScores       VIP values computed from training data
%   optRocPoint     [FPR, TPR] corresponding to LDA threshold

    % Ensure column vectors
    Train_YLabel = Train_YLabel(:);

    %=============================================================
    % 1. Fit one-component PLS model on TRAINING data
    %=============================================================
    [XL, YL, XS, ~, BETA, ~, ~, stats] = plsregress(Train_XFeats, Train_YLabel, 1);

    % Continuous PLS scores (latent projection)
    Train_plsScore = [ones(size(Train_XFeats,1),1), Train_XFeats] * BETA;
    Test_plsScore  = [ones(size(Test_XFeats,1),1),  Test_XFeats]  * BETA;

    %=============================================================
    % 2. Train 1D diag-linear LDA on TRAINING scores only
    %    (compat: works with and without fitcdiscr)
    %=============================================================
    ldaParams = fitDiagLinearLDA_1D_compat(Train_plsScore, Train_YLabel, priorType);

    % Predict binary labels for TEST data (using robust analytic boundary)
    Test_predYLabel = predictDiagLinearLDA_1D(Test_plsScore, ldaParams);

    %=============================================================
    % 3. Determine LDA decision threshold in score space
    %=============================================================
    % classify as 1 if w*x + b >= 0
    ldaThresh = -ldaParams.b / ldaParams.w;

    %=============================================================
    % 4. Compute the ROC curve on TRAINING data only
    %=============================================================
    [rocFpr, rocTpr, rocThresh] = perfcurve(Train_YLabel, Train_plsScore, 1);

    % Find ROC point corresponding to LDA threshold
    [~, idx] = min(abs(rocThresh - ldaThresh));
    optRocPoint = [rocFpr(idx), rocTpr(idx)];

    %=============================================================
    % 5. Compute VIP scores (training only)
    %=============================================================
    % Normalize PLS weights
    plsWeightsNorm = stats.W ./ sqrt(sum(stats.W.^2));

    % For one component:
    plsVarFactor = sum(XS.^2) * sum(YL.^2);

    vipScores = sqrt(size(XL,1) * (plsVarFactor * (plsWeightsNorm.^2)) / plsVarFactor);

end



%%
%======================================================================
% Local helpers: version-robust 1D diaglinear LDA (0/1 classes)
%======================================================================
function ldaParams = fitDiagLinearLDA_1D_compat(Train_XFeats, Train_YLabel, priorType)
% Returns ldaParams with fields:
%   w, b : boundary for delta(x)=w*x+b, predict 1 if delta>=0
%   prior0, prior1
%   mu0, mu1, sigma2 (pooled variance)
%
% Works whether or not fitcdiscr exists. We *optionally* try fitcdiscr,
% but we compute the boundary analytically to avoid Coeffs/version issues.

    Train_XFeats = Train_XFeats(:);
    Train_YLabel = Train_YLabel(:);

    % Force binary {0,1}
    u = unique(Train_YLabel);
    if numel(u) ~= 2
        error('Train_YLabel must have exactly two classes (binary).');
    end

    % Map to 0/1 if needed (keeps semantics: larger unique -> 1)
    if ~isequal(u(:).', [0 1])
        y01 = zeros(size(Train_YLabel));
        y01(Train_YLabel == u(2)) = 1;
    else
        y01 = Train_YLabel;
    end

    x0 = Train_XFeats(y01 == 0);
    x1 = Train_XFeats(y01 == 1);

    n0 = numel(x0);
    n1 = numel(x1);
    if n0 < 2 || n1 < 2
        error('Each class needs at least 2 samples for LDA variance estimation.');
    end

    mu0 = mean(x0);
    mu1 = mean(x1);

    % pooled within-class variance (unbiased variances combined)
    v0 = var(x0, 1); % use population form to keep stable; pooled below
    v1 = var(x1, 1);
    sigma2 = ((n0 * v0) + (n1 * v1)) / (n0 + n1);

    if sigma2 <= 0
        error('Degenerate variance in training scores; cannot fit LDA.');
    end

    % Priors
    switch lower(priorType)
        case {'uniform','equal'}
            prior0 = 0.5;
            prior1 = 0.5;
        case {'empirical','observed'}
            prior0 = n0 / (n0 + n1);
            prior1 = n1 / (n0 + n1);
        otherwise
            error('priorType must be ''uniform'' or ''empirical''.');
    end

    % Optional: try fitcdiscr if available (not required for the boundary),
    % useful if you later want posteriors/scores in newer MATLAB.
    if exist('fitcdiscr','file') == 2 %#ok<EXIST>
        try
%             fitcdiscr(Train_XFeats, y01, 'DiscrimType','diaglinear', 'Prior',[prior0 prior1], 'ClassNames',[0 1]); %#ok<NASGU>
        catch
            % ignore: we still have a complete analytic model
        end
    end

    % Analytic delta(x) = log p(1|x) - log p(0|x) for 1D LDA with pooled var:
    % delta(x) = ((mu1-mu0)/sigma2)*x + [ -0.5*(mu1^2-mu0^2)/sigma2 + log(prior1/prior0) ]
    w = (mu1 - mu0) / sigma2;
    b = (-0.5 * (mu1^2 - mu0^2) / sigma2) + log(prior1 / prior0);

    ldaParams.mu0    = mu0;
    ldaParams.mu1    = mu1;
    ldaParams.sigma2 = sigma2;
    ldaParams.prior0 = prior0;
    ldaParams.prior1 = prior1;
    ldaParams.w      = w;
    ldaParams.b      = b;
end

function yPred = predictDiagLinearLDA_1D(x, ldaParams)
% Predicts labels in {0,1} using analytic boundary delta(x)=w*x+b.

    x = x(:);
    delta = ldaParams.w .* x + ldaParams.b;
    yPred = double(delta >= 0);
end



%% OPTIONAL USE OF BUILT IN FUNCTIONS FOR LDA
% useFitc = (exist('fitcdiscr','file') == 2);
% 
% % Ensure y01 is 0/1 numeric
% y01 = double(y01);
% 
% if useFitc
%     ldaModel = fitcdiscr(Train_XFeats, y01, ...
%         'DiscrimType','diaglinear', ...
%         'Prior',[prior0 prior1], ...
%         'ClassNames',[0 1]);
%     Test_predYLabel = predict(ldaModel, Test_XFeats);
% else
%     % Older MATLAB (R2013b) - ClassificationDiscriminant exists in Stats TB
%     ldaModel = ClassificationDiscriminant.fit(Train_XFeats, y01, ...
%         'DiscrimType','diaglinear', ...
%         'Prior',[prior0 prior1], ...
%         'ClassNames',[0 1]);
%     Test_predYLabel = predict(ldaModel, Test_XFeats);
% end
