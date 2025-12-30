# This file contains the feature engineering logic 

import pandas as pd 

def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
  X = dataframe.copy()

  X['HasPool'] = X['PoolQC'].notna().astype(int) 
  X['HasMisc'] = X['MiscFeature'].notna().astype(int)
  X['HasAlley'] = X['Alley'].notna().astype(int)
  X['HasFence'] = X['Fence'].notna().astype(int)
  X['HasMasVnr'] = (X['MasVnrArea']>0).astype(int)
  X['HasFireplace'] = X['FireplaceQu'].notna().astype(int)
  X['HasBasement'] = X['BsmtQual'].notna().astype(int)
  X['HasGarage'] = X['GarageType'].notna().astype(int)

  X['TotalInHouseArea'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

  X['HouseSaleAge'] = X['YrSold'] - X['YearBuilt']
  X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']


  X = X.drop(
  columns=[
    'PoolQC','MiscFeature','Alley','Fence','MasVnrArea','FireplaceQu',
    'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
    'GarageType', 'GarageQual', 'GarageCond',
    'TotalBsmtSF','1stFlrSF','2ndFlrSF',
    'YrSold','YearBuilt','YearRemodAdd'
  ]
  )

  return X 
