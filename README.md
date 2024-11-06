# Combining Query Performance Predictors: A Reproducibility Study - under review at ECIR 2024.

**Note**: This paper is currently under review at ECIR 2025. This repository is shared to provide access to the source code and data solely for review purposes. We kindly request that the repository not be shared externally until the review process is complete.



## Pre-Retrieval Methods

   - Available in the [PreRetQPP](PreRetQPP/) directory.

1. **MaxIDF / AvgIDF**  
   - [Pre-retrieval IDF-based predictors]()

2. **SumSCQ / AvSCQ / MaxSCQ**  
   - [SCQ-based QPP Predictors](https://)

3. **Ambiguity-based (AvP, AvNP)**  
   - [Ambiguity and Similarity-based QPP predictors](https://)

4. **Ranking Sensitivity-based (SumVAR, AvVAR, MaxVAR)**  
   - [Ranking Sensitivity-based Predictors](https://)

## Post-Retrieval Methods
1. **WIG (Weighted Information Gain)**  
   - [Link](https://github.com/suchanadatta/qpp-eval)

2. **NQC (Normalized Query Commitment)**  
   - [Link](https://github.com/suchanadatta/qpp-eval)

3. **Clarity**  
   - [Clarity Implementation]()

4. **UEF (Uncertain Estimation Fusion)**  

   - [UEF Framework]()

5. **NeuralQPP**
    - available in the [NeuralQPP](NeuralQPP/) directory
6. **Deep-QPP**  
   - [Link](https://github.com/suchanadatta/DeepQPP)

6. **qppBERT-PL** 
   - [Link](https://github.com/Narabzad/BERTQPP)

7. **BERT-QPP**  
    - [Link](https://github.com/suchanadatta/qppBERT-PL)

These repositories serve as resources for implementations of various QPP methods, some of which may require adaptation to integrate into the project’s specific framework.

## QPP Scores
Pre-computed QPP, AP measures can be found in [data](data/) folder.

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


