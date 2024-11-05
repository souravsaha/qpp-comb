### QPP Scores
Pre-computed QPP, AP measures can be found in data/ folder.

### Example command
To run the leave one out based approaches: 
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
