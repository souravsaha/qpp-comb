# Combining Query Performance Predictors: A Reproducibility Study. 

This repository contains the codebase of [Combining Query Performance Predictors: A Reproducibility Study](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_9) (published in ECIR 2025)

## Pre-Retrieval Methods


   - As the original pre-retrieval QPP methods were not provided by Hauff et al. (ECIR 2009), we have implemented them here, specifically `AvgIDF`, `MaxIDF`, `SumSCQ`, `AvgSCQ`, `MaxSCQ`, `SumVAR`, `AvgVAR`, `MaxVAR`, `AvP`, `AvNP`. The implementations are avaialble  in the [PreRetQPP](PreRetQPP/) directory.

1. **MaxIDF / AvgIDF**  
   - [Pre-retrieval IDF-based predictors]()

2. **SumSCQ / AvSCQ / MaxSCQ**  
   - [SCQ-based QPP Predictors](https://)

3. **Ambiguity-based (AvP, AvNP)**  
   - [Ambiguity and Similarity-based QPP predictors](https://)

4. **Ranking Sensitivity-based (SumVAR, AvVAR, MaxVAR)**  
   - [Ranking Sensitivity-based Predictors](https://)

## Post-Retrieval Methods

   - For the post-retrieval QPP methods, we utilized code provided by the original authors when available. For methods lacking a readily available codebase, we implemented them independently. Below are the links to all 10 methods used in this project:

1. **WIG (Weighted Information Gain)**  (unofficial implementation)
   - [Link](https://github.com/suchanadatta/qpp-eval)

2. **NQC (Normalized Query Commitment)**  (unofficial implementation)
   - [Link](https://github.com/suchanadatta/qpp-eval)

3. **Clarity**  (unofficial implementation)
   - [Link](https://github.com/suchanadatta/qpp-eval)

4. **UEF (Uncertain Estimation Fusion)**  (unofficial implementation)

   - [Link](https://github.com/suchanadatta/qpp-eval)

5. **NeuralQPP**
    - available in the [NeuralQPP](NeuralQPP/) directory

6. **Deep-QPP**  (from the author's repository)
   - [Link](https://github.com/suchanadatta/DeepQPP)

6. **qppBERT-PL** (from the author's repository)
   - [Link](https://github.com/suchanadatta/qppBERT-PL)

7. **BERT-QPP**  (from the author's repository)
    - [Link](https://github.com/Narabzad/BERTQPP)

These repositories serve as resources for implementations of various QPP methods, some of which may require adaptation to integrate into the project’s specific framework.

## QPP Scores
Pre-computed QPP, AP measures can be found in [data](data/) folder.

## Combining different QPP methods
To run different (penalized) regression methods and sampling strategies on three collections — TREC Robust, TREC DL 2019 & 2020, and ClueWeb09B — use the following instructions. 

### Example command
To run the leave one out based approaches used by Hauff et al. : 
```bash
python3 leave-one-out.py --k 1000 --input data --qpp_type pre --dataset trec678 --ols_type ols
```
To run lars-traps with leave one out :
```bash
python3 lars-traps.py --k 1000 --input data --qpp_type pre --dataset trec678
```
To run bolasso with leave one out:
```bash
python3 bolasso.py --k 1000 --input data --qpp_type pre --dataset trec678
```


To run lars-traps with half-split :
```bash
python3 lars-traps-split-half.py --k 1000 --input data --qpp_type pre --dataset trec678rb
```
To run bolasso with half-split:
```bash
python3 bolasso-split-half.py --k 1000 --input data --qpp_type post --dataset trec678rb
```
To compute smare: 

```bash
python3 smare <path_to_csv_file>
```
Compute correlation metrics:

```bash
python3 compute-correlation.py <path_csv_file> <retrieval_depth (1000 / 100)>
```
Fit linear regression with indv. predictor (half-split):

```bash
python3 indv-predictor-regr-half-split.py --k 1000 --input data --qpp_type pre --dataset trec678rb --ols_type ols
```

To run multiple regression with half-split:
```bash
python3 multiple-regression-half-split.py --k 1000 --input data --qpp_type pre --dataset trec678rb --ols_type ols
```

## Citation
```bibtex
@InProceedings{10.1007/978-3-031-88717-8_9,
   author="Saha, Sourav
   and Datta, Suchana
   and Roy, Dwaipayan
   and Mitra, Mandar
   and Greene, Derek",
   title="Combining Query Performance Predictors: A Reproducibility Study",
   booktitle="Advances in Information Retrieval",
   year="2025",
   publisher="Springer Nature Switzerland",
   address="Cham",
   pages="112--129",
   isbn="978-3-031-88717-8"
}
