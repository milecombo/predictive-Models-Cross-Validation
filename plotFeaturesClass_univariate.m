function plotFeaturesClass_univariate(feats, classY, featName)

N= numel(classY);
Nfeats= size(feats,2);


if Nfeats>=2
    %     ax = tightSubplot(1,2,1,plotOpz.hPad,plotOpz.vPad,plotOpz.rect);
    %     set(ax,'fontsize', plotOpz.fontsz)
    %     gscatter( feats(:,1),  feats(:,2),  classY,[0 0 1; 1 0 0]);  % axis tight; %equal
    %     xlabel('feature 1'); ylabel('feature 2');
    pairs = nchoosek(1:Nfeats,2);
    K = size(pairs,1);
    
    nc = min(5, K);
    nr = ceil(K/nc);
    if Nfeats <=12
        figure('Color','w', 'Name','Bivariate feature pairs');
        hPad = 0.02;  vPad = 0.04;
        rect = [0.04 0.06 0.96 0.92];
        
        for k = 1:K
            
            ax = tightSubplot(nr,nc,k,hPad,vPad,rect);
            gscatter( feats(:,pairs(k,2)), feats(:,pairs(k,1)),  classY, ...
                [0 0 1; 1 0 0], '.', 8);
            ylabel(sprintf('f%d',pairs(k,1)));
            xlabel(sprintf('f%d',pairs(k,2)));
            axis tight;
            if k>1; legend off
            end
        end
    end
    
end
%%
figure('Color','w','name','feature correlations'),
plotOpz.hPad = 0.02;          % horizontal spacing between panels
plotOpz.vPad = 0.05;          % vertical spacing (more room for labels)
plotOpz.rect = [0.025 0.025 0.95 0.95];   % centered container
plotOpz.fontsz= 16;

ax = tightSubplot(1,1,1,plotOpz.hPad+.05,plotOpz.vPad,plotOpz.rect);
set(ax,'fontsize', plotOpz.fontsz)
plotCorr=[];
plotCorr.ticks=1:Nfeats+1;
plotCorr.ticksName=[ featName, 'class'];
plotCorr.r=corr([ feats,  classY]);
imagesc(plotCorr.ticks ,plotCorr.ticks, abs( plotCorr.r ),[0 1]);
set(ax, 'xtick',plotCorr.ticks, 'xticklabel', plotCorr.ticksName , 'ytick',plotCorr.ticks, 'yticklabel',plotCorr.ticksName )
colormap(parula); axis square

for i = 2:size(plotCorr.r,1)
    for j = 1:i-1
        text(j,i,sprintf('%.2f', plotCorr.r(i,j)), ...
            'HorizontalAlignment','center','FontSize',8);
    end
end

colorbar; title('Abs. Correlation Coeff. R')
%% %
try
    figure('Color','w','name','feature display for each class'),
    for f = 1:Nfeats
        ax = tightSubplot(1,Nfeats,f,plotOpz.hPad,plotOpz.vPad+.01, plotOpz.rect);
        hold(ax,'on'); plotOpz.pos = get(ax,'Position');
        set(ax,'fontsize', plotOpz.fontsz)
        boxplot( feats(:,f),  classY, 'symbol','', 'position', unique( classY) );
        plot( classY +rand( N,1)./2 -.25,  feats(:,f) , '.')
        
        c0= classY==min(classY);    c1= classY==max(classY);
        [H,P,CI,STATS] = ttest2(  feats(c0,f), feats(c1,f) );
        xlabel('Class'); % ylabel(sprintf('Feature %d',f))
        txt= sprintf('T(%d) = %.3f ', STATS.df, STATS.tstat);
        txt1= sprintf('p= %.3g ', P); % note: p is uncorrected for multiple comparisons
        
        [uni_FPR, uni_TPR, uni_Thr, uni_AROC] = perfcurve( double(c1), feats(:,f), 1);
        txt2=  sprintf('AROC= %.2f ', uni_AROC);
        plotCorr.contrast.t(f)= STATS.tstat;
        plotCorr.contrast.df(f)= STATS.df;
        plotCorr.contrast.p_uncorrected(f)= P;
        plotCorr.contrast.AROC(f)= uni_AROC;
        plotCorr.contrast.txt{f}= [ featName{f} ' ' txt2  ' '  txt  ' ' txt1];
        title([ featName{f} ' ' txt2    char(10)  txt  char(10) txt1]);     box off
        set(ax, 'Position', plotOpz.pos)   % <-- restore geometry
    end
catch me
    disp(me) % usually too many features:   -->'tightSubplot: pads too large for the chosen grid/rect.'
end
