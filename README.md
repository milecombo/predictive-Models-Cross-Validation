# Predictive-Models-Cross-Validation
**Interpretable prediction for correlated variables: regression + classification with resampling**.

## Overall: Achieving Good, Interpretable Predictions
**The main goal is to perform an interpretable regression and classification from few (up to many) correlated variables**.

A flexible linear framework for combining similar, correlated predictors (e.g. questionnaires or physiological features) is offered by *Partial Least Squares (PLS)*. 
PLS allows you to predict one (or potentially more) ordinal or continuous outcome variable (e.g. a clinical scale or the presence/absence of a diagnosis). Although PLS is formally a regression method, using a binary-coded outcome produces a latent dimension that naturally aligns with class separation. We then turn these continuous scores into discrete class labels by learning an optimal linear decision boundary between groups with *Linear Discriminant Analysis (LDA)*.

The goodness of our prediction (and all performance metrics) is computed strictly on unseen data, never on training data. 
To obtain reliable estimates of performance and variable importance, we use a common resampling strategy:a  *Cross-Validation scheme*, repeated N times for robustness.
Finally, we verify whether predictions are better than those that can arise by chance, using a common *Label Permutation framework*.

## Why a Partial Least Squares (PLS) Model ?
PLS is a supervised dimensionality-reduction method that projects correlated predictors onto a small set of latent components that best explain their covariance with the target variable. It is robust to multicollinearity, works well with few samples or many predictors, and yields flexible, interpretable linear projections suitable for classification or regression.
PLS also provides Variable Importance in Projection (VIP), which quantifies each feature’s contribution to the latent component and supports transparent model interpretation.
*This is extremely valuable in scientific contexts, where understanding feature relevance matters as much as predictive performance—unlike many engineering-oriented machine-learning workflows that prioritize accuracy alone.*

## Why Linear Discriminant Analysis (LDA) after PLS?
When the prediction target is categorical, PLS first maps predictors onto a continuous latent score that summarizes how “case-like” or “control-like” each sample is. 
LDA then takes these scores and learns an optimal linear decision boundary, converting them into discrete class labels in a principled, statistically grounded way instead of using an arbitrary threshold.

## Why Resampling Data With Cross-Validation ?
Cross-validation tests how well a model generalizes by repeatedly training it on one subset of the data and testing it on another. Instead of relying on a single split, the data are divided into several folds so every sample is used for both training and evaluation across iterations. This reduces overfitting and yields a more reliable estimate of real-world performance. Stratified K-fold cross-validation preserves class balance, while repeated runs smooth variability and improve robustness.

## Why Signifiance Testing Via Permutation ?
Permutation testing checks whether model performance is genuinely meaningful or could arise by chance. By shuffling class labels and repeating the full cross-validation workflow, we generate a “chance-level” distribution to compare against the real model.
Each permutation randomizes labels while preserving data structure, and the entire CV pipeline is rerun to obtain null metrics. Significance is then evaluated using the unbiased (k+1)/(N+1) rule. This provides a nonparametric, rigorous baseline for determining whether the observed predictive signal exceeds chance

## Our Specific Scenario: EEG to Predict Confusional State
We aim to use a small set of EEG spectral features to estimate whether a person who suffered head trauma is currently in a confusional state (**Post-Traumatic Confusional State, PTCS**) or not (**Traumatic Brain Injury Control, TBI-Control**). Because PLS can handle strongly correlated predictors and performs well even with relatively small sample sizes, it is particularly suitable for neurophysiological datasets such as EEG (where many correlated predictors can be easily obtained).
In our (relatively simple) case, we extract a single latent component from the EEG features. This component captures both the shared variance among predictors and the variance associated with the outcome.
We choose a single component to maximize parsimony and interpretability; however, in more complex settings, multiple components could be estimated and selected via nested cross-validation.

The task is a binary prediction problem (PTCS vs TBI-Control), encoded numerically as 1 and 0. We employ a linear modeling framework to estimate a continuous discriminative score, which is later thresholded for classification.
A crucial detail in our dataset is that PTCS patients contribute two EEG recordings at distinct time points. To prevent subject-wise data leakage—and thereby avoid artificially inflated performance—recordings from the same participant are never split across training and testing folds. This ensures that the model is always evaluated on truly unseen subjects rather than on repeated measurements of individuals it has already “seen” during training.

*Possible extensions: We use only one component to enforce parsimony and interpretability, and because the small number of predictors makes additional components unnecessary; however, in more complex settings one could estimate multiple components and let nested cross-validation determine the optimal number.
Although we consider only one outcome variable here, the PLS framework naturally extends to settings with multiple predictors and even multiple response variables.*

## In Brief

* **brief explanation**<br>
  We reduce correlated variables (here, EEG features) to a single meaningful latent dimension/axis via PLS, then learn an optimal threshold separating cases and controls (here, confused vs non-confused patients) using LDA, and evaluate performance on fully held-out data via repeated cross-validation. As a final sanity check, we assess if predictions are better than chance (via permutation).

* **brief procedure**<br>
  Within repeated stratified 10-fold cross-validation: Train Partial Least Square model > Latent scores > Threshold classes with LDA > Predict held-out data > Compute performance + Variable Importance> Aggregate over runs. P-values by permutations.
This can achieve robust and interpretable predictions, with quantified feature importance and explicit control for overfitting and chance-level performance.

