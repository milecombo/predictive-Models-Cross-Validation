function plt = plotConfMat (C_count, C_pct, classNames)
% cvCombi.OverReps_Mean_ConfMat_PredTrue;
% cvCombi.OverReps_MeanPct_ConfMat_PredTrue;
%  cvCombi.OverReps_MedIQR_Acc';

plt.C_count = C_count; % cvCombi.OverReps_Mean_ConfMat_PredTrue;
plt.C_pct   = C_pct; %cvCombi.OverReps_MeanPct_ConfMat_PredTrue;

plt.nClasses = size(plt.C_pct,1);

imagesc(plt.C_pct,[0 100]); axis square tight; colormap(parula(50));

set(gca, 'TickDir','out','Box','off', ...
    'XTick',1:plt.nClasses,'XTickLabel',classNames, ...
    'YTick',1:plt.nClasses,'YTickLabel',classNames);
xlabel('TRUE class');  ylabel('PREDICTED class');

[plt.x, plt.y] = meshgrid(1:plt.nClasses);
text(plt.x(:), plt.y(:), ...
    arrayfun(@(v) sprintf('%.0f',v), plt.C_count(:), 'UniformOutput', false), ...
    'HorizontalAlignment','center','FontSize', 20);


set(gca,'Units','normalized'); plt.axPos = get(gca,'Position');
set(gca,'Position', plt.axPos, 'ydir', 'reverse' )
plt.cb = colorbar; set(plt.cb,'YTick',[0 50 100]);

% pos = get(plt.cb,'Position');   % [x y width height]

% set(plt.cb,'Position',pos);


drawnow
pos=plt.axPos;
 pos(4) = 0.6 * pos(4);       % shrink height to 80%
 pos(2) = pos(2) + 0.3*pos(4); % re-center vertically (optional)

set(plt.cb,'Units','normalized','Position',[pos(1)+pos(3)+0.01    pos(2)    0.012    pos(4)]);
ylabel(plt.cb,'% of true class');