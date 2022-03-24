import pandas as pd
import math
import numpy as np

patient_df = pd.read_csv('D:/data/recurrence/complete.csv')

def prrsent_percent(arr, is_ablation):
    nans = 0
    for i in range(len(arr)): 
        if is_ablation[i]:
            try:
                is_nan = np.isnan(arr[i])
                nans=nans+int(is_nan)+int(arr[i]=='-')
            except TypeError:
                continue
    per = nans/sum(is_ablation)
    return '{}({})'.format(sum(is_ablation)-nans,1-per)

def tumor_type_count(arr, is_ablation):
    arr_count = 0
    for i in range(len(arr)):
        if is_ablation[i]:
            arr_count = arr_count+int(arr[i]=='Checked')
    return '{}({})'.format(arr_count,arr_count/sum(is_ablation))


def serum_presence_cal(patient_df):
    AFP = patient_df['AFP']
    PAA = patient_df['PAA']
    AA = patient_df['Aspartate_aminotransferase']
    CEA = patient_df['CEA']
    DCP = patient_df['DCP']
    is_abltion = patient_df['Ablation'] == 'Yes'
    print('afp_present_percnetage: ', prrsent_percent(AFP.values,is_abltion))
    print('PAA_present_percnetage: ', prrsent_percent(PAA.values,is_abltion))
    print('Aspartate_aminotransferase_present_percnetage: ', prrsent_percent(AA.values,is_abltion))
    print('CEA_present_percnetage: ', prrsent_percent(CEA.values,is_abltion))
    print('DCP_percnetage: ', prrsent_percent(DCP.values,is_abltion))

# print(patient_df.columns)
def ablation_patient_overview(patient_df):
    print('total_patient: ',patient_df.shape[0]-1)
    print('ablation_numbers: {}({})'.format(sum(patient_df['Ablation']=='Yes'), sum(patient_df['Ablation']=='Yes')/(patient_df.shape[0]-1)))
    serum_presence_cal(patient_df)

print(tumor_type_count(patient_df['Tumor_type (choice=Metastasis)'].values, patient_df['Ablation'] == 'Yes'))

print(tumor_type_count(patient_df['Tumor_type (choice=HCC)'].values, patient_df['Ablation'] == 'Yes'))

print(tumor_type_count(patient_df['Tumor_type (choice=Other, please explain in textbox below)'].values, patient_df['Ablation'] == 'Yes'))
