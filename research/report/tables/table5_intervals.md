**Intervals and counts with the conditions they depend on**

| Quantity | Correct | n | Estimate | Method | Lower | Upper | Conditional on | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Serving logistic model, original 141-row holdout | 141 | 141 | 1.0000 | Clopper-Pearson | 0.9742 | 1.0000 | the fitted model (or rule) and independent test rows | acceptable if rows are independent |
| Rule total >= 7, original holdout | 141 | 141 | 1.0000 | Clopper-Pearson | 0.9742 | 1.0000 | the fitted model (or rule) and independent test rows | acceptable if rows are independent |
| Rule total >= 6, original holdout | 129 | 141 | 0.9149 | Clopper-Pearson | 0.8561 | 0.9552 | the fitted model (or rule) and independent test rows | acceptable if rows are independent |
| Documented rule total >= 6, all 704 rows | 634 | 704 | 0.9006 | Clopper-Pearson | 0.8760 | 0.9217 | a rule fixed before looking at the data and independent rows | acceptable if rows are independent |
| Documented rule total >= 6, all 704 rows | 634 | 704 | 0.9006 | Wilson | 0.8762 | 0.9205 | a rule fixed before looking at the data and independent rows | acceptable if rows are independent |
| Rule total >= 7 (found after looking at these labels), all 704 rows | 704 | 704 | 1.0000 | count only |  |  | this file | descriptive fact: 0 disagreements; not an estimate of future agreement |
| Upper bound on disagreement rate for rule >= 7 (two-sided 95%, single pre-specified rule) | 704 | 704 | 0.0000 | exact, zero errors | 0.0000 | 0.0052 | independent rows from the same labeling process | conditional; not clinical accuracy |
| Upper bound on disagreement rate for rule >= 7 (one-sided 95%, adjusted for 12 scanned cut-offs) | 704 | 704 | 0.0000 | exact, zero errors | 0.0000 | 0.0078 | independent rows from the same labeling process | conditional; not clinical accuracy |
