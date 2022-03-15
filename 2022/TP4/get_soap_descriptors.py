import os
import numpy as np
import copy
import pandas as pd
import time
import sys
from sklearn.preprocessing import StandardScaler

def get_soap_datasets():
  # Read data
  INPUT_DIR = "./"
  df_train = pd.read_csv(INPUT_DIR+"train.csv")
  df_train["dataset"] = "train"
  df_test = pd.read_csv(INPUT_DIR+"test.csv")
  df_test["dataset"] = "test"
  df_crystals = pd.concat([df_train, df_test], ignore_index=True)


  train_data = INPUT_DIR + 'train/'
  test_data = INPUT_DIR + 'test/'
  csv_path = INPUT_DIR
  train_csv = np.loadtxt(csv_path+'/train.csv', skiprows=1, delimiter=',')
  test_csv = np.loadtxt(csv_path+'/test.csv', skiprows=1, delimiter=',')


  test_true_csv = pd.read_csv(INPUT_DIR+"labeled_test.csv")
  bandgap_test_true = test_true_csv["bandgap_energy_ev"]
  formation_test_true = test_true_csv["formation_energy_ev_natom"]
  formation_train_true = df_train["formation_energy_ev_natom"]
  bandgap_train_true = df_train["bandgap_energy_ev"]

  soap_model_saved = np.load('./soap.npz')
  train_mean_atomic_descriptors = soap_model_saved['train_mean_atomic_descriptors']
  test_mean_atomic_descriptors = soap_model_saved['test_mean_atomic_descriptors']
  scaler = StandardScaler(with_mean=True, with_std=True)
  formation_train_true = train_csv[:, -2]
  bandgap_train_true = train_csv[:, -1]
  formation_test_true = np.array(test_true_csv["formation_energy_ev_natom"])
  bandgap_test_true = np.array(test_true_csv["bandgap_energy_ev"])

  combined_mean_atomic_descriptors = scaler.fit_transform(np.append(
          test_mean_atomic_descriptors,
          train_mean_atomic_descriptors,
          axis=0))

  X_test = np.round(combined_mean_atomic_descriptors[:test_csv.shape[0]],
          decimals=5)

  X_train = np.round(combined_mean_atomic_descriptors[test_csv.shape[0]:],
          decimals=5)
  
  train_dataset_Eform = (X_train, formation_train_true)
  train_dataset_Egap = (X_train, bandgap_train_true)
  test_dataset_Eform = (X_test, formation_test_true)
  test_dataset_Egap = (X_test, bandgap_test_true)

  return train_dataset_Eform, train_dataset_Egap, test_dataset_Eform, test_dataset_Egap