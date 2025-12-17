# Predictive Modelling and Cross-Validation:<br> explanation and pseudocode

**Interpretable prediction for correlated variables: regression + classification with resampling**.

## Overall: Achieving Good, Interpretable Predictions
**The main goal is to perform an interpretable regression and classification from few (up to many) correlated variables**.

A flexible linear framework for combining similar, correlated predictors (e.g. questionnaires or physiological features) is offered by *Partial Least Squares (PLS)*. 
PLS allows you to predict one (or potentially more) ordinal or continuous outcome variable (e.g. a clinical scale or the presence/absence of a diagnosis). Although PLS is formally a regression method, using a binary-coded outcome produces a latent dimension that naturally aligns with class separation. We then turn these continuous scores into discrete class labels by learning an optimal linear decision boundary between groups with *Linear Discriminant Analysis (LDA)*.

The goodness of our prediction (and all performance metrics) is computed strictly on unseen data, never on training data. 
To obtain reliable estimates of performance and variable importance, we use a common resampling strategy:a  *Cross-Validation scheme*, repeated N times for robustness.
Finally, we can verify whether predictions are better than those that can arise by chance, using a common *Label Permutation framework*.

## Why a Partial Least Squares (PLS) Model ?
PLS is a supervised dimensionality-reduction method that projects correlated predictors onto a small set of latent components that best explain their covariance with the target variable. It is robust to multicollinearity, works well with few samples or many predictors, and yields flexible, interpretable linear projections suitable for classification or regression.
PLS also provides Variable Importance in Projection (VIP), which quantifies each feature’s contribution to the latent component and supports transparent model interpretation.
*This is extremely valuable in scientific contexts, where understanding feature relevance matters as much as predictive performance—unlike many engineering-oriented machine-learning workflows that prioritize accuracy alone.*

## Why Linear Discriminant Analysis (LDA) after PLS?
When the prediction target is categorical, PLS first maps predictors onto a continuous latent score that summarizes how “case-like” or “control-like” each sample is. 
LDA then takes these scores and learns an optimal linear decision boundary, converting them into discrete class labels in a principled, statistically grounded way instead of using an arbitrary threshold.

## Why Resampling Data With Cross-Validation ?
Cross-validation tests how well a model generalizes by repeatedly training it on one subset of the data and testing it on another. Instead of relying on a single split, the data are divided into several folds so every sample is used for both training and evaluation across iterations. This reduces overfitting and yields a more reliable estimate of real-world performance. Stratified K-fold cross-validation preserves class balance, while repeated runs smooth variability and improve robustness.

## Why Significance Testing Via Permutation ?
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
  We reduce correlated variables (here, EEG features) to a single meaningful latent dimension/axis via PLS, then learn an optimal threshold separating cases and controls (here, confused vs non-confused patients) using LDA, and evaluate performance on fully held-out data via repeated cross-validation. As a final sanity check, we assess if predictions are better than chance (via permutation). This can achieve robust and interpretable predictions, with quantified feature importance and explicit control for overfitting and chance-level performance.


* **brief procedure**<br>
  Within repeated stratified 10-fold cross-validation: Train Partial Least Square model > Latent scores > Threshold classes with LDA > Predict held-out data > Compute performance + Variable Importance> Aggregate over runs. P-values by permutations.

<br>

# **PSEUDO-CODE**


**Repeated, subject-blocked, stratified Cross-Validation of PLS continuous scores with LDA classification <br> 
using correlated EEG features to predict presence of confusional state, and assess feature importance in prediction**

```text
FOR each run of the repeated stratified K-fold cross-validation (10 runs):

    Generate a stratified partition of the data into K folds (K = 10),
    ensuring that no subject appears in both train and test
    (i.e. avoiding any data-leakage bias from the subject’s repeated measures).

    FOR each fold of one cross-validation run:
        --------------------------------------------------
        TRAINING PHASE (build model from training set only)

        1. Define train and test sets.
           Extract training predictors (e.g. four EEG spectral features)
           and training class labels (e.g. confused vs not confused).

        2. Standardize predictors (train only).
           Z-score predictors within the training set.
           Store the training-set mean and variance for later application
           to the test set.

        3. Train the supervised model (PLS regression).
           Fit a one-component Partial Least Squares (PLS) model on the
           current training set to predict the outcome
           (e.g. confusional state, coded 1/0).
           PLS linearly combines the predictors into a single supervised
           latent dimension that maximally covaries with the class structure,
           producing a continuous discriminative score for each training sample.

        4. Estimate feature importance (VIP).
           Compute Variable Importance in Projection (VIP) values from
           the training fold to quantify each feature’s contribution
           to the PLS projection.

        5. Define a threshold to binarize continuous scores (LDA).
           Train a 1-dimensional Linear Discriminant Analysis (LDA)
           classifier on the training PLS scores, using uniform priors.
           LDA yields a linear decision threshold in the latent space
           to binarize scores into discrete class predictions
           (e.g. confused vs not confused).

        6. Characterize the LDA threshold on the ROC curve (train only).
           Construct the ROC curve based only on training scores
           and identify the ROC point corresponding to the
           LDA decision boundary.

        ---------------------------------------------
        TESTING PHASE (blind prediction on test set)

        7. Apply training standardization parameters to the test predictors.

        8. Compute continuous PLS scores for the test samples
           using the trained PLS model.

        9. Use the trained LDA model to convert test scores into
           binary predicted labels.

        --------------------
        METRICS AND STORAGE

        10. Store:
            - continuous test-set scores
            - predicted test-set labels
            - true test-set labels (from outside the model function)
            - fold-level VIP scores
            - LDA-matched ROC point.

        11. Compute fold-level classification metrics
            (on held-out data only):
            - ROC AUC (from continuous scores)
            - accuracy
            - F1 (confused)
            - F1 (non-confused).
            
-------------------------
AFTER all runs and folds:

- Aggregate performance across all folds and runs by computing the
      median and IQR of all stored metrics.
- Pool held-out test scores across folds and runs and compare their
      distributions between classes to assess separability.
```

### Optional: Permutation testing

Permutation testing is used to check whether the observed performance
could have arisen by chance. The idea is to keep the predictors and
cross-validation scheme identical, but break the true relationship
between predictors and labels by randomly shuffling the labels.

This creates a “null” model that reflects what performance looks like
when there is no real signal, while preserving the structure of the data
and the resampling procedure.

```text
OPTIONAL: PERMUTATION TESTING

1. Run the real model once:
   - Input: original labels y
   - Apply the full repeated, subject-blocked, stratified K-fold
     PLS–LDA pipeline (as above).
   - Store observed metrics:
       AUC_obs, ACC_obs, F1_conf_obs, F1_nonconf_obs.

---------------------------------------------
2. FOR each permutation (perm = 1 to N_perm):

      a. Randomly permute the labels at the sample level:
         y_perm = random permutation of y.

      b. Keep predictors X, subject-blocking and stratification
         exactly the same.

      c. Apply the same repeated PLS–LDA cross-validation pipeline
         as for the real data, but using y_perm.

      d. Store aggregated metrics for this permutation:
         AUC_perm[perm], ACC_perm[perm],
         F1_conf_perm[perm], F1_nonconf_perm[perm].
---------------------------------------------------
 
3. Compute permutation p-values using the unbiased (k + 1) / (N_perm + 1) rule:
   - For each metric, let k be the number of permuted values
     at least as extreme as the observed one, e.g.:
       p_AUC       = ( count(AUC_perm      >= AUC_obs)       + 1 ) / (N_perm + 1)
       p_ACC       = ( count(ACC_perm      >= ACC_obs)       + 1 ) / (N_perm + 1)
       p_F1_conf   = ( count(F1_conf_perm  >= F1_conf_obs)   + 1 ) / (N_perm + 1)
       p_F1_nonconf= ( count(F1_nonconf_perm>=F1_nonconf_obs)+ 1 ) / (N_perm + 1)
```
These permutation-derived p-values tell you how often a performance
as good as the observed one would occur if there were no true
relationship between X and y. Usually, statistical significance is claimed when p < 0.05.

