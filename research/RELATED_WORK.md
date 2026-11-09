# Related work: verified comparison

Prepared 2026-10-11 for the ASD-Predict study. Companion to [REVIEW.md](REVIEW.md).

## How this was verified

Each paper has an **Access** level. Claims in the tables come only from what I could read
at that level.

| Access | Meaning |
|---|---|
| **Full text** | I downloaded and read the paper text myself |
| **Partial** | I read some sections; the unread sections are listed |
| **Abstract only** | I read the PubMed or arXiv abstract, not the paper |
| **Metadata only** | I confirmed title, authors, venue and DOI (Crossref/PubMed), nothing else |
| **Second-hand** | I only know what another paper I read says about it |

Statements marked **(my check)** are my own analysis of the data, not claims by the paper.
Items marked **(inference)** are hypotheses I could not confirm.

Papers behind paywalls (no PMC copy, 403 or login pages) were not read. I did not work
around access controls. Where a preprint was read, the published version may differ.

## What the papers I read do and do not cover

- Neither ML-on-UCI paper I read (Rasul; ASD-Bench) reports that the dataset's label is a
  deterministic function of the AQ-10 total, or tests for it.
- ASD-Bench states that labels are questionnaire-derived and attributes near-perfect
  adult scores to small sample size and limited feature diversity. It does not test
  whether the label is a threshold on the score.
- Bone et al. (2015) make the broader circularity argument for ADOS and ADI-R: predicting
  an instrument's own algorithmic diagnosis from its item codes is a near-trivial task.
  The concept is therefore not new; quantifying it on the UCI AQ-10 data is the narrow
  contribution.
- I read two of the many ML papers that use these datasets. Absence of the finding in
  those two is not evidence that nobody has published it.

## Table 1: Autism screening and autism-ML papers

| Paper | Access | What it studied | What it reports | What this study could add |
|---|---|---|---|---|
| Allison, Auyeung, Baron-Cohen (2012), *JAACAP* 51(2):202-212.e7, doi:10.1016/j.jaac.2011.11.003 | Abstract only | Deriving 10-item short forms of the AQ (adult, adolescent, child) and the Q-CHAT as referral "red flag" tools. Cases: 449 adults, 162 adolescents, 432 children, 126 toddlers with ASC, recruited from the Autism Research Centre's volunteer database. Controls: 3,000 recruited from various sources. | At cut-point 6 on the adult AQ-10: sensitivity 0.88, specificity 0.91, PPV 0.85. Adolescent 0.93/0.95/0.86, child 0.95/0.97/0.94. Q-CHAT-10 at cut-point 3: 0.91/0.89/0.58. Internal consistency above 0.85. | This is a case-control validation against diagnosis. The UCI data has no diagnoses. A contribution here is showing the UCI labels follow `>= 7`, not the published cut-point 6, so UCI agreement cannot be read as validation of the instrument. |
| Booth et al. (2013), *J Autism Dev Disord* 43(12):2997-3000, doi:10.1007/s10803-013-1844-5 | Abstract only | Whether the AQ-10 classifies ASD correctly in adults: 149 people with ASD and 134 controls without. | AQ-50, AQ-S and AQ-10 all performed well; sensitivity, specificity and AUC were very similar at suggested cut-offs; little difference between AQ-10 and AQ-50. | Independent validation of the instrument exists. This study does not validate the AQ-10; it audits a dataset that uses it. |
| Bone et al. (2015), *J Autism Dev Disord* 45(5):1121-1136, doi:10.1007/s10803-014-2268-6 | **Full text** (PMC4390409) | Critical evaluation and attempted replication of two studies (Wall et al. 2012) claiming ML can shorten ADOS Module 1 and ADI-R diagnosis. Replication on larger, more balanced data (AGRE; a combined research/clinical set called BID). | Conceptual problems: the studies predicted the ADOS or ADI-R diagnosis, not best-estimate clinical diagnosis; they excluded the middle severity class, making the task easier ("classifying ADOS diagnosis from ADOS codes is trivial with polarized categories"; the ADOS algorithm is built on summation and thresholding, so "by definition" it scores 100% on that task). Methodological problems: test samples simulated from training data are not independent; no negative cases in validation data; no bootstrapping of feature reliability. Replication: the proposed 8 ADOS codes gave 87% unweighted average recall (UAR) in AGRE against the reported 100% accuracy; 94% (8 codes) vs 99% (all 29) in BID. Adding the middle class dropped UAR (for example 99% to 90% in BID). They recommend UAR over accuracy for imbalanced classes and warn that the same subject in train and test inflates performance. | Same circularity structure, now for a questionnaire dataset with an automatically computed score. Their setting had a gold-standard comparison (best-estimate diagnosis in BID); the UCI data does not. Borrow their metric advice (UAR or balanced accuracy) and their independence warning. **Novelty caution:** cite them for the concept. |
| Rasul et al. (2023), arXiv:2309.11646v2 (preprint "submitted to Elsevier"; peer-review status unknown) | **Full text** through the conclusion (pages 1-17); reference and figure pages skimmed by keyword search | Eight classifiers and five clustering algorithms on UCI child (292 rows), adult (704) and a combined set (996). 70/30 split, 5-fold CV for model selection, chi-square top-10 feature selection, hyperparameter search, a Streamlit GUI. | Children: SVM and LR reach 100% accuracy. Adults: LR best at 97.14%. Combined: ANN 94.28%. Spectral clustering best on NMI and ARI. A9 reported as prominent. A keyword search found no mention of leakage, thresholds, limitations or how labels were derived. No confidence intervals or repeated splits. | (1) Show the label identity for both UCI datasets (adult: in this repo; child: my check below). (2) **(inference)** Their Figure 4 chi-square ranking, read left to right, puts all ten items in the child top-10 and only eight items (A1 and A8 missing) in the adult top-10. That pattern would fit 100% vs 97% if the label is a threshold on all ten items. I could not confirm the axis order is descending rank. (3) Report uncertainty: one 70/30 split of about 211 test rows moves several points (see my check below). |
| Singh et al. (2026), "ASD-Bench", arXiv:2605.11091v1 (11 May 2026; preprint, peer-review status unknown) | **Full text** (20 pages as extracted; the HAP sensitivity appendix it mentions is not in my copy) | Benchmark of 17 ML, deep and foundation models on "Dataset v3": 4,068 AQ-10 records combining UCI data with a Kaggle set, in three age cohorts. Four axes: performance, calibration, interpretability, robustness. New metric HAP (cost-weighted errors plus a fold-variance penalty). | Adults: 10 of 17 models reach F1 = 1.000 and AUC = 1.000. Adolescent F1 ceiling 0.837, child 0.915. AdaBoost: adult F1 1.000 with ECE 0.302. Consensus feature importance flat for adults, A9 top for children, A5 top for adolescents. Authors' own caveats: labels are questionnaire-derived, single split with seed 42, no external validation, adult results "likely reflect the constrained sample size and limited feature diversity rather than true diagnostic task saturation". Pipeline "will be released upon acceptance". | Offer a more direct explanation for the perfect adult scores: if their adult labels are also a threshold on the AQ total (not checked; I have not examined their data), perfect separability is expected at any sample size. The flat adult importance is what equal item weights would produce (**my check** on UCI adult only: fitted weights 10.2-10.9, CV 2%). The paper interprets flat importance as "consistent with developmental social masking". Apply this study's label audit to their public v3 dataset (link in their Data Availability section; I did not open it). Note their fold-variance term in HAP is not the variance of the estimator (see Bengio and Grandvalet below). |
| Thabtah (2017), *Proc. 1st Int. Conf. on Medical and Health Informatics*, doi:10.1145/3107514.3107515, and Thabtah, *Inform Health Soc Care* 44(3):278-297, doi:10.1080/17538157.2017.1399132 | Metadata only | Listed as "relevant papers" in the dataset's own description file. Not read. | Not read. | Read both before writing Methods. Check whether either defines `Class/ASD` and the `>= 7` rule. This could settle the label-provenance question that the UCI documentation leaves open. |

### Details behind specific cells

**ASD-Bench inconsistencies I noticed (not resolved)**

- Table 1 lists 736 final adult records (380 duplicates removed), while the text says "only
  148 unique adult records remain" (Section 6.1) and "n = 148 after deduplication"
  (Conclusion). **(inference)** 148 may be the 20% test split of 736, but the paper does not
  say.
- Table 2's item wording and keying (for example A1 "I prefer to do things the same way
  over and over again"; reverse-scored items A3, A8, A9, A10) do not match the ARC adult
  AQ-10 form I read (items 1, 7, 8, 10 score on agree; 2, 3, 4, 5, 6, 9 on disagree). The
  paper does not explain the difference.
- Rasul et al.'s Table 3 describes A1-A10 with child-style questions for all datasets,
  also unlike the adult AQ-10 items.

Together these show that item wording and keying for this dataset family is documented
inconsistently in the literature. That supports treating item encoding as unverified in
this study's Limitations.

**My checks prompted by these papers (scratch analysis, not saved in the repo)**

- **UCI child dataset (id 419, 292 rows):** the label is exactly `sum >= 7` on the ten
  items: 0 disagreements at threshold 7, 40 at 6, 44 at 8. Scores 0-6 are all `NO`, scores
  7-10 all `YES`. The recorded `result` equals the item sum on every row. The UCI
  adolescent file (id 420) could not be downloaded from the URL pattern I tried, so it was
  not checked.
- **Rasul adult hypothesis:** logistic regression on A2-A7, A9, A10 plus age and ethnicity
  (the eight items in their top-10) gave 5-fold CV accuracy 0.935-0.939 and, over three
  70/30 splits, 0.934, 0.967 and 0.953. Their reported 97.14% sits at the high end of that
  split-to-split range. This is consistent with their result, not proof of how they got
  it; their preprocessing differs from mine.

## Table 2: Leakage and uncertainty methodology

| Paper | Access | What it studied | What it reports | What this study could add |
|---|---|---|---|---|
| Kapoor and Narayanan (2023), *Patterns* 4(9):100804, doi:10.1016/j.patter.2023.100804 | **Partial**: I read the arXiv v1 preprint (2207.07048): abstract, overview, survey table, taxonomy and other-issues sections. Not read: model info sheets (Section 3), the civil-war reproduction details, and the published version beyond its PubMed abstract. | Data leakage as a cause of irreproducible ML-based science: a survey of fields where errors were found, a taxonomy, "model info sheets", and a reproduction study of civil-war prediction. | Published abstract: 17 fields, 294 papers affected (the preprint said 20 papers, 17 fields, 329 papers; cite the published numbers). Taxonomy of eight leakage types: L1.1 no test set, L1.2 pre-processing on train and test, L1.3 feature selection on train and test, L1.4 duplicates, L2 illegitimate features (for example proxies for the outcome), L3.1 temporal, L3.2 non-independence of train and test (same people), L3.3 sampling bias. The survey table includes an autism-diagnostics entry (Bone et al. 2015). In the civil-war study, corrected complex models did not beat logistic regression. | Use the taxonomy to structure Methods. My mapping (the paper does not discuss AQ-10): the label-from-items problem is closest to L2; repeated and identical records to L1.4; unknown participant identity to L3.2; the self-selected app sample to L3.3. Their model info sheets could organize the audit. |
| Bengio and Grandvalet (2004), *J Mach Learn Res* 5:1089-1105 | **Full text** | The variance of the K-fold cross-validation estimator. | Main theorem: no universal unbiased estimator of the variance of K-fold CV exists. Errors are correlated because test and training sets overlap, so naive estimators can badly underestimate variance; in their experiments the bias was of the order of the variance itself. The problem "cannot be addressed by performing multiple K-fold splits". CV estimates the expected error of the algorithm trained on n - n/K examples. A single hold-out set estimates test-set variance but not training-set variance. | Basis for the uncertainty recommendations in REVIEW.md F6: do not present a binomial interval on pooled out-of-fold predictions as a confidence interval for CV performance. |
| Varoquaux (2018), *NeuroImage* 180:68-77, doi:10.1016/j.neuroimage.2017.06.061 | **Full text** of the arXiv v1 preprint (1706.07581), including appendices; the published version may differ | Error bars of cross-validated accuracy at small sample sizes (neuroimaging data, simulations, a binomial model). | About ±10% for 100 samples. The binomial law is a best-case lower bound; actual errors are larger with correlated data or unstable models. Standard error across folds underestimates actual bounds by a factor of 0.73 (leave-one-out) and 0.26 (50 repeated 20% splits) in his simulation. With a data-independent perfect predictor, the error is traced to test-set sampling noise (Appendix E). Permutation testing gives good control. Simulations use balanced classes, 75% accuracy and a linear SVM. | Shows a binomial interval is right for a fixed predictor and only a lower bound for trained models. Supports reporting split-to-split SD as "not a confidence interval" and adding a permutation null for the non-questionnaire AUC. |
| Kaufman et al. (2012), *ACM TKDD* 6(4), doi:10.1145/2382577.2382579 | Metadata only; **second-hand** via Kapoor and Narayanan | Described there as formative work on leakage types and mitigations, including proxy variables. | Not read. | Read before citing for anything beyond "foundational leakage reference". |
| Nadeau and Bengio (2003), *Machine Learning* 52(3):239-281, doi:10.1023/A:1024068626366 | Metadata only (full text gated); **second-hand** via Bengio and Grandvalet | Inference for the generalization error. | Bengio and Grandvalet say it investigated several variance estimators and considers K independent training and test splits. I have not read it. | Candidate source for a corrected paired comparison of models. Read before recommending any specific estimator. |
| Dietterich (1998), *Neural Computation* 10(7):1895-1923, doi:10.1162/089976698300017197 | Metadata only | Approximate statistical tests for comparing classifiers. | Not read. | Read if you add paired model comparisons. |
| Clopper and Pearson (1934), *Biometrika* 26(4):404-413; Brown, Cai and DasGupta (2001), *Stat Sci* 16(2) | Metadata only | Exact and alternative binomial intervals. | Not read. | Cite for the interval formulas; read Brown et al. before describing coverage properties. |

## Reading list still to do

1. Thabtah (2017) and the 2018 review: label definition and the `>= 7` rule.
2. Kapoor and Narayanan, model info sheets (Section 3) and the published version.
3. Nadeau and Bengio (2003), if you want a paired model-comparison method.
4. ASD-Bench appendix and public dataset, to test whether their adult labels follow a
   threshold.
5. Kaufman et al. (2012).

## Not verified, do not cite

- NICE CG142 (403 error). Attribute the "recommended in CG142" statement to the ARC form.
- A Chilean thesis mentioned in a search summary as having removed a leaking variable. I
  could not confirm it.
- The README's references (Zhang 2025; Duke Psychiatry 2023). Not checked in this review.
