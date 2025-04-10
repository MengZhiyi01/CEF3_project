# Result Folder

## Overview

This folder contains the processed data generated from the notebook `EDA_Part3_fillna.ipynb`. The data has undergone a series of cleaning and preprocessing steps to ensure it is ready for further analysis or modeling tasks.

## Generating Processed Data

To generate the processed data yourself, you can run the Jupyter notebook `EDA_Part3_fillna.ipynb`. This notebook outlines the steps taken to clean and preprocess the raw data, including handling missing values and other necessary transformations.

## Downloading Processed Data

For your convenience, we provide pre-processed datasets that you can download directly:

- **Processed_data_1600.parquet**  
  - [Download Link](https://disk.pku.edu.cn/link/AA1B9780F1200440E2B4E6394BA913396C)
  - Pickup Code: CTNM
  - 1300: Data begins from `date_id = 1300`.

- **Processed_data_1300.parquet.zip**  
  - [Download Link](https://disk.pku.edu.cn/link/AA0225209C528A46B2906702F6972392DE)
  - Pickup Code: M0cl
  - 1600: Data begins from `date_id = 1600`.

- **Processed_data_1695.parquet**  
  - This file is a smaller sample of the processed data, specifically prepared to comply with GitHub's file size limitations. It is intended for demonstration purposes and provides a representative subset of the full dataset.
  - 1695: Data begins from `date_id = 1695`.

Note: All datasets are ended at `date_id = 1699`.

### Notes

- Ensure you have the required pickup code to access the files.
- The `.parquet` format is optimized for efficient storage and fast read/write operations, making it suitable for large datasets.
- The zipped file (`Processed_data_1300.parquet.zip`) should be extracted before use.
- The `Processed_data_1695.parquet` file is a smaller sample provided for demonstration and testing purposes, especially useful for those who want to quickly explore the data without downloading the full dataset.

We recommend reviewing the notebook `EDA_Part3_fillna.ipynb` for detailed information on the preprocessing steps applied to the data. If you have any questions or need further assistance, please feel free to reach out.
