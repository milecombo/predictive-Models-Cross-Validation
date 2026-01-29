function plt = plotPLS_CV_Summary(cvCombi, cvRuns,  varargin)
% plotPLS_CV_Summary
% Summary visualization of repeated-CV PLS-LDA results
%
% OPTIONAL name-value overrides (varargin):
%   'fontsz'     (default 12)
%   'classNames' (default inferred from classTypes)
%   'featName'   (default {'feat1','feat2', ...'featN'})
%   'hPad'       (default 0.015)
%   'vPad'       (default 0.020)

%% defaults
plt=[];
plt.fontsz=12;
plt.classNames= {'0' '1'};
plt.hPad = 0.015;
plt.vPad = 0.020;
plt.nFeats = size(cvRuns.Rep_Vip_MeanAcrossFolds,2);
plt.featNameN = arrayfun(@(i) sprintf('feat%d', i),  1:plt.nFeats, 'UniformOutput', false);
plt.featName= plt.featNameN;
%% varargin override (robust to old-MATLAB versions)
for ii=1:2:numel(varargin)
    plt.(varargin{ii}) = varargin{ii+1};
end

%% FIGURE 
plt.classY     =  cvCombi.Sample_TrueYLabel(:); % myClass(:);

% ---- class definitions (MUST come early: used throughout the plot) ----
plt.classTypes = cvCombi.cvOpts.classTypes(:);
plt.nClasses   = numel(plt.classTypes);
plt.classCounts = arrayfun(@(c) sum(plt.classY == c), plt.classTypes);
plt.classNames = cellstr(num2str(plt.classTypes)); % safe tick labels for old/new MATLAB
plt.vipReps= cvRuns.Rep_Vip_MeanAcrossFolds;


txt = arrayfun(@(c,n) sprintf('Class %g (N=%d) ', c, n), ...
    plt.classTypes, plt.classCounts, 'UniformOutput', false);
plt.classCountName= (strjoin(txt', ' vs '));

% title (compact feature list)
if isfield(plt,'featName') && ~isempty(plt.featName)
    plt.titleFig= sprintf('PLS1 %s', strjoin(plt.featName,' + '));
else plt.titleFig= 'PLS1';
end

figure('Color','w', 'name', plt.titleFig);

%% ===== VIP over reps =====
subplot(151),
plt.ax1 = gca; set(plt.ax1, 'fontsize', plt.fontsz)
hold on, boxplot( plt.vipReps),
for fz = 1: plt.nFeats
  plot(fz, plt.vipReps(:,fz),'.k')
end
ylabel('Variable Importance in Prediction (VIP)'), xlabel('Features/Predictors')
v = cvCombi.OverReps_MedIQR_Vip';
[~, si]= sort(v(:,1),'descend');  
vs= v(si,:);
 plt.featNameSort=  plt.featName(si);

v = arrayfun(@(i) sprintf('%s %.3f (%.3f - %.3f)', ...
    plt.featNameSort{i}, vs(i,1), vs(i,2), vs(i,3)), ... %     plt.featName{i}, v(i,1), v(i,2), v(i,3)), ...
    (1:size(v,1))', 'UniformOutput', false);
set(plt.ax1, 'Xtick',1: plt.nFeats, 'XtickLabel', plt.featNameN); %  plt.featName, 'XTickLabelRotation',45)

plt.titleVIP= ['VIP, median (IQR)'; v]; title(plt.titleVIP)
if numel(v)>3;
    v=[v(1:3)];  title( ['VIP, top 3 features, median (IQR)'; v]);
end

box off

%% ===== AROC over reps =====
subplot(1,5,2)
plt.ax2 = gca;   set(plt.ax2, 'fontsize', plt.fontsz)
hold on, boxplot(cvRuns.Rep_AROC),
plot(1,  cvRuns.Rep_AROC,'.k')
ylabel('Area under the ROC (AROC)'); 
ylim([min([cvRuns.Rep_AROC(:); .5]) 1]); xlim([.8 1.2])
v = cvCombi.OverReps_MedIQR_AROC';
v=  {sprintf('%.3f (%.3f - %.3f)', v(1), v(2), v(3)) };
plt.titleAROC= ['AROC, median (IQR)'; v];  title(plt.titleAROC)
box off

%%%%%%%%%%%%%%%%%%%%
%% ===== Median OutOfFold scores by class =====
plt.scoreMed   = cvCombi.OverReps_Med_OutOfFold_Score(:);

[plt.isValid, plt.group] = ismember(plt.classY, plt.classTypes);
plt.scoreMed = plt.scoreMed(plt.isValid);
plt.group    = plt.group(plt.isValid);
plt.positions = 1:plt.nClasses;

subplot(1,5,3:4) , plt.ax3 = gca;
hold(plt.ax3,'on'); set(plt.ax3, 'fontsize', plt.fontsz)

boxplot(plt.scoreMed, plt.group, 'positions', plt.positions, ...
    'labels', cellstr(num2str(plt.classTypes)),  'symbol', '');

plt.jitterWidth = 0.18;
for c = 1 :plt.nClasses
    plt.jitt=  (rand(sum(plt.group==c),1)-0.5)*plt.jitterWidth;
    plot(plt.positions(c) + plt.jitt,  plt.scoreMed(plt.group==c), '.k');
end

set(plt.ax3, 'XTick', plt.positions, ...
    'XTickLabel', cellstr(num2str(plt.classTypes)), ...
    'TickDir', 'out', 'Box', 'off');
xlim(plt.ax3, [0.5, plt.nClasses + 0.5]);
xlabel(plt.ax3,'True Class'); ylabel(plt.ax3,'Median out-of-fold PLS score');

plt.ttest = cvCombi.ttest_Score_OverReps_Med_OutOfFold;
v= {sprintf('T(%g)=%.3f', plt.ttest.df, plt.ttest.tstat)};
v=[ [ plt.classCountName] v];
plt.titleScores=['t-test on median (across runs) out-of-fold PLS scores  ' v];
title(plt.ax3, plt.titleScores);

%%%%%%%%%%%%%%%%%%%%
%% ===== Confusion matrix =====
plt.ax4 = tightSubplot(2,5,5,plt.hPad,plt.vPad);
hold(plt.ax4,'on'); set(plt.ax4, 'fontsize', plt.fontsz)
v = cvCombi.OverReps_MedIQR_Acc';
v=  {sprintf('%.3f (%.3f - %.3f)', v(1), v(2), v(3)) };
plt.titleAcc= ['Accuracy, median (IQR)'; v];  title(plt.titleAcc)

plt.C_count = cvCombi.OverReps_Mean_ConfMat_PredTrue;
plt.C_pct   = cvCombi.OverReps_MeanPct_ConfMat_PredTrue;
plt.confMat= plotConfMat (plt.C_count, plt.C_pct, plt.classNames);
%%%%%%%%%%%%%%%%%%%%
%% ===== ROC curves =====
plt.ax5 = tightSubplot(2,5,10,plt.hPad,plt.vPad);
hold(plt.ax5,'on'); set(plt.ax5, 'fontsize', plt.fontsz)

plt.FPRu = cvRuns.RepRoc_FPRu;   plt.TPRu = cvRuns.RepRoc_TPRu;
plt.FPRgrid = cvRuns.RepRoc_FPRgrid(:);
plt.TPRgrid = cvRuns.RepRoc_TPRgrid;

tol = 1e-12;
plt.FPRref = plt.FPRu(1,:);
plt.valid  = ~isnan(plt.FPRref);
plt.sameFPR = all( all( abs(plt.FPRu(:,plt.valid) - ...
    repmat(plt.FPRref(plt.valid), size(plt.FPRu,1), 1)) < tol, 2 ) );
plt.sameFPR = plt.sameFPR && all(all(isnan(plt.FPRu(:,~plt.valid)),2));

if plt.sameFPR
    %%% if all CV runs yield same unique thresholds and FPR values, plot these unique steps
    stairs(plt.FPRu', plt.TPRu', '-', 'color',[.5 .5 .5])
    plt.TPRmed = prctile(plt.TPRu, 50, 1);
    plt.FPRmed = plt.FPRref(plt.valid);
    plt.TPRmed = plt.TPRmed(plt.valid);
    stairs(plt.FPRmed, plt.TPRmed, 'k', 'linewidth', 2)
    title('Median ROC from unique FPR points');
else
    %%% if some CV runs have different thresholds and FPR values across runs,
    % plot the ROC curve as a grid/sweep across FPR values (default to 100 steps)
    stairs(plt.FPRgrid(:), plt.TPRgrid.', '-', 'color',[.5 .5 .5])
    plt.TPRmed = prctile(plt.TPRgrid, 50, 1);
    stairs(plt.FPRgrid(:), plt.TPRmed(:), 'k', 'linewidth', 2)
    title('Median ROC from fixed FPR grid');
end

plot([0 1],[0 1],'k:'); axis square; xlim([0 1]); ylim([0 1]);
xlabel('FPR'); ylabel('TPR');

%%%%%%%%%%
v = cvCombi.OverReps_MedIQR_F1_posClass';
plt.titleF1_posClass= ['F1 of positive class, median (IQR)'; ...
    {sprintf('%.3f (%.3f - %.3f)', v(1), v(2), v(3))}];

v = cvCombi.OverReps_MedIQR_F1_negClass';
plt.titleF1_negClass= ['F1 of negative class, median (IQR)'; ...
    {sprintf('%.3f (%.3f - %.3f)', v(1), v(2), v(3))}];

 v = cvCombi.OverReps_MedIQR_Acc';
 v=  {sprintf('%.3f (%.3f - %.3f)', v(1), v(2), v(3)) };
 plt.titleAcc= ['Accuracy, median (IQR)'; v];  title(plt.titleAcc)

disp('Feature importance in PLS, based on VIP scores'); disp(plt.titleVIP);
disp('Performance of continuous PLS scores'); disp(plt.titleAROC); disp(plt.titleScores);
disp('Performance of LDA classification based on PLS scores');
disp(plt.titleAcc); disp(plt.titleF1_posClass); disp(plt.titleF1_negClass);

end
