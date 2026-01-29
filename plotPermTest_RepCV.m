function plt = plotPermTest_RepCV(permTest, cvCombinedObs, plotOpz)
%
% Plot permutation-test results for PLS–LDA CV wrapper outputs.
%
% Inputs
%   permTest        struct produced by the permutation function (runPermTest_RepCV), expected fields:
%       .obs.med_AROC, .obs.med_Acc, .obs.med_F1_posClass, .obs.med_F1_negClass, .obs.tstat
%       .null.nPerm
%       .null.med_AROC, .null.med_Acc, .null.med_F1_posClass, .null.med_F1_negClass, .null.tstat
%       .null.ConfMat_PredTrue   [nClass × nClass × nPerm]  (Pred × True × Perm)
%       .p.med_AROC, .p.med_Acc, .p.med_F1_posClass, .p.med_F1_negClass, .p.tstat
%
%   cvCombinedObs   observed wrapper output (cvCombined from real data), used for:
%       .OverReps_Mean_ConfMat_PredTrue        (Pred × True counts)
%       .OverReps_MeanPct_ConfMat_PredTrue     (Pred × True %, optional)
%       .cvOpts.classTypes                    (class ordering)
%
%   plotOpz         struct (optional):
%       .fontsz      fontsize for metric panels (default 16)
%       .classNames  cellstr class labels for confusion matrices (default from classTypes)
%
% Output
%   plt struct with a few handles/derived matrices:
%       .figMetrics, .figConf
%       .Cnull_count, .Cnull_pct, .Cobs_count, .Cobs_pct
%
% IMPORTANT CONVENTION
% Confusion matrices are treated as Predicted × True (Pred on rows, True on columns).
% Therefore "percent within each true class" is column-normalization:
%   C_pct = 100 * C ./ sum(C,1)
% (Columns sum to 100.)

% ---------------- defaults ----------------
if nargin < 3 || isempty(plotOpz), plotOpz = struct(); end
if ~isfield(plotOpz,'fontsz') || isempty(plotOpz.fontsz), plotOpz.fontsz = 16; end

plt = struct();

% ---------------- sanity: required fields ----------------
req = {'obs','null','p'};
for ii=1:numel(req)
    if ~isfield(permTest,req{ii}), error('plotPermTest_PLSLDA: missing permTest.%s', req{ii}); end
end

% ---------------- METRICS FIGURE ----------------
plt.figMetrics = figure('color','w','name','Performance: Random Permutations (gray) VS Observed (red)');
jit = 1 + rand(permTest.null.nPerm,1)./6 - 1/12;

mylim = [permTest.null.med_AROC; permTest.null.med_Acc; ...
         permTest.null.med_F1_posClass; permTest.null.med_F1_negClass; ...
         permTest.obs.med_AROC; permTest.obs.med_Acc; ...
         permTest.obs.med_F1_posClass; permTest.obs.med_F1_negClass];
mylim = [min(mylim)-.05  max(mylim)+.05];

subplot(1,6,1); hold on; set(gca,'fontsize',plotOpz.fontsz);
plot(jit, permTest.null.med_AROC, '.', 'color',[.5 .5 .5]);
boxplot(permTest.null.med_AROC, 'Symbol','');
plot(1, permTest.obs.med_AROC, 'r.', 'MarkerSize',25);
ylim(mylim); xlim([.75 1.25]);
title(sprintf('AROC = %.3f \n p = %.4g', permTest.obs.med_AROC, permTest.p.med_AROC));
ylabel(['Performance Metrics' char(10) 'permutations (gray) vs observed (red)']);

subplot(1,6,2); hold on; set(gca,'fontsize',plotOpz.fontsz);
plot(jit, permTest.null.med_Acc, '.', 'color',[.5 .5 .5]);
boxplot(permTest.null.med_Acc, 'Symbol','');
plot(1, permTest.obs.med_Acc, 'r.', 'MarkerSize',25);
ylim(mylim); xlim([.75 1.25]);
title(sprintf('Acc = %.3f \n p = %.4g', permTest.obs.med_Acc, permTest.p.med_Acc));

subplot(1,6,3); hold on; set(gca,'fontsize',plotOpz.fontsz);
plot(jit, permTest.null.med_F1_posClass, '.', 'color',[.5 .5 .5]);
boxplot(permTest.null.med_F1_posClass, 'Symbol','');
plot(1, permTest.obs.med_F1_posClass, 'r.', 'MarkerSize',25);
ylim(mylim); xlim([.75 1.25]);
title(sprintf('F1 (+) = %.3f \n p = %.4g', permTest.obs.med_F1_posClass, permTest.p.med_F1_posClass));

subplot(1,6,4); hold on; set(gca,'fontsize',plotOpz.fontsz);
plot(jit, permTest.null.med_F1_negClass, '.', 'color',[.5 .5 .5]);
boxplot(permTest.null.med_F1_negClass, 'Symbol','');
plot(1, permTest.obs.med_F1_negClass, 'r.', 'MarkerSize',25);
ylim(mylim); xlim([.75 1.25]);
title(sprintf('F1 (-) = %.3f \n p = %.4g', permTest.obs.med_F1_negClass, permTest.p.med_F1_negClass));

subplot(1,6,5); hold on; set(gca,'fontsize',plotOpz.fontsz);
plot(jit, abs(permTest.null.tstat), '.', 'color',[.5 .5 .5]);
boxplot(abs(permTest.null.tstat), 'Symbol','');
plot(1, abs(permTest.obs.tstat), 'r.', 'MarkerSize',25);
mylimT = [abs(permTest.null.tstat(:)); abs(permTest.obs.tstat)];
mylimT = [min(mylimT)-.05  max(mylimT)+.1];
ylim(mylimT); xlim([.75 1.25]);
title(sprintf('|T| = %.3f \n p = %.4g', abs(permTest.obs.tstat), permTest.p.tstat));

%% ---------------- CONFUSION MATRICES FIGURE ----------------
% Requires: tightSubplot + plotConfMat in path.
% plt.figConf = figure('color','w','name','Confusion matrices: Null mean vs Observed');

% classNames default from classTypes
if ~isfield(plotOpz,'classNames') || isempty(plotOpz.classNames)
    ct = cvCombinedObs.cvOpts.classTypes(:)';
    plotOpz.classNames = cell(1,numel(ct));
    for ii=1:numel(ct), plotOpz.classNames{ii} = sprintf('%g', ct(ii)); end
end

plt.hPad = 0.015;
plt.vPad = 0.020;

% ---- NULL ----
tightSubplot(2,6,6,plt.hPad,plt.vPad); hold on; set(gca,'fontsize',20);

plt.Cnull_count = squeeze(mean(permTest.null.ConfMat_PredTrue, 1));            % Pred × True
plt.Cnull_pct   = confPct_PredTrue(plt.Cnull_count);                  % Pred × True (% within true class)

v = prctile(permTest.null.med_Acc,[50 25 75]);
title({sprintf('Null Acc: %.3f (%.3f–%.3f)', v(1), v(2), v(3))});

plotConfMat(plt.Cnull_count, plt.Cnull_pct, plotOpz.classNames);

%% ---- OBSERVED ----
tightSubplot(2,6,12,plt.hPad,plt.vPad); hold on; set(gca,'fontsize',20);

plt.Cobs_count = cvCombinedObs.OverReps_Mean_ConfMat_PredTrue;         % Pred × True

if isfield(cvCombinedObs,'OverReps_MeanPct_ConfMat_PredTrue') && ~isempty(cvCombinedObs.OverReps_MeanPct_ConfMat_PredTrue)
    plt.Cobs_pct = cvCombinedObs.OverReps_MeanPct_ConfMat_PredTrue;    % Pred × True
else
    plt.Cobs_pct = confPct_PredTrue(plt.Cobs_count);                   % compute if not stored
end

title(sprintf('Observed Acc: %.3f', permTest.obs.med_Acc));
plotConfMat(plt.Cobs_count, plt.Cobs_pct, plotOpz.classNames);

end

% ===== local helper: percent ConfMat for Pred × True =====
function C_pct = confPct_PredTrue(C_count)
% confPct_PredTrue
% Convert confusion counts (Pred × True) into column-normalized percentages,
% i.e. within each TRUE class (columns sum to 100).
den = sum(C_count,1);
den(den==0) = NaN;  % avoid divide-by-zero if a column is empty
C_pct = 100 * bsxfun(@rdivide, C_count, den);
end
