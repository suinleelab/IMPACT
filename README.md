# IMPACT
This repository provides the dataset, models and code for paper "Interpretable machine learning prediction of all-cause mortality".
Code for paper "Interpretable machine learning prediction of all-cause mortality". Please read our preprint at the following link: https://www.medrxiv.org/content/10.1101/2021.01.20.21250135v2


## Data

### NHANES
Please find the NHANES data in **_data/NHANES/NHANES.csv_**.
Here are the mortality labels:
- mortstat: mortality status (0=Assumed alive, 1=Assumed deceased
- permth_int: person months from the date of interview to the date of death or the end of the mortality period
- x_year_label (x=1,2,3,4,5): the label for x-year mortality prediction
More description of the features can be found in **_data/NHANES/NHANES_feature_list.csv_**

### UK Biobank
The overlapping features' information between NHANES and UK Biobank can be found in **_data/UKB/overlapped_NHANES_UKB.csv_**

Data cannot be shared publicly by the authors because of information governance restrictions around health data. The data can however be downloaded following a project approval process by the UK Biobank. Researchers wishing to access the data can apply directly to the UK Biobank https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access and the process involves registering on the access management system, submitting a research study protocol and paying a fee directly to the UK Biobank. UK Biobank is an open access research resource for researchers and accepts applications with no restriction.

## Models
All of the mortality prediction models are available in **model/NHANES_xxx/**. Please find the input feature list in **_model/model_features.csv_**. We also share the explicands and SHAP values:
- **_model.pickle.dat_**: the trained xgboost model
- **_fore_data.csv_**: the explicands when calculating the SHAP values
- **_shap_values.npy_**: the SHAP values

The model for IMPACT-20 mortality risk scores are available in **_model/IMPACT-20/_**. Please find the input feature list in **_model/IMPACT-20/IMPACT-20_features.csv_**

The input feature list can be also obtained by
    ```
    import pickle
    model = pickle.load(open(model_path, 'rb')
    print(model.get_booster().feature_names)
    ```
## Code
### Model training
- **_shap_NHANES_classification.py_**: code for mortality prediction model training and SHAP values calculation
- **_mortality_risk_scores_feature_elimination.ipynb_**: code for mortality risk scores training and feature elimination
- **_supervised_distance_feature_elimination.ipynb_**: code for supervised distance calculation and supervised distance-based feature elimination approach
### Visualization
**_supervised_distance_feature_elimination.ipynb_** process results from the IMPACT framework and generate figures presented in the paper:
- SHAP summary plot
- SHAP values plot
- SHAP main effect plot: please calculate the SHAP interaction values using **shap_NHANES_classification.py** before generating the SHAP main effect plot
- SHAP interaction plot: please calculate the SHAP interaction values using **shap_NHANES_classification.py** before generating the SHAP main effect plot
- SHAP individualized plot
- Partial dependence plot for reference interval

## Dependencies 

This software was originally designed with Python 3.6.13. Standard python software packages used: numpy (1.20.3), pandas (1.3.2), scikit-learn (0.22.2 or 0.21.3), shap (0.39.0), matplotlib (3.4.2).
** The _model/IMPACT-20/IMPACT_5_year_top20.pickle.dat_ and _model/IMPACT-20/IMPACT_5_year_Demo_Lab_top20.pickle.dat_ must be loaded using scikit-learn (0.21.3).**

## References

If you find contrastiveVI useful for your work, please consider citing our preprent:

```
@article{qiu2022interpretable,
  title={Interpretable machine learning prediction of all-cause mortality},
  author={Qiu, Wei and Chen, Hugh and Dincer, Ayse Berceste and Lundberg, Scott and Kaeberlein, Matt and Lee, Su-In},
  journal={medRxiv},
  pages={2021--01},
  year={2022},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
