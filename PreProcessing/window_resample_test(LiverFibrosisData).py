from window_resample import Preprocessing
import pandas as pd
import os
import SimpleITK as sitk

patient_dictionary = pd.read_excel('D:/dictionary_files/UMCG_PATIENTS/training_date_dictionary_16_6M_update.xls')
anon_ids = patient_dictionary['anonimi_id']
ct_names = patient_dictionary['dicom_name']
path_CTs='D:/data/CT_Fibrosis'
path_livers = 'D:/data/CT_Fibrosis_liver_mask'
scan_savepath = 'D:/data/recurrence/scans/training_scan/CT'
livermask_savepath = 'D:/data/recurrence/scans/training_scan/livermask'

def path_gen(anon_id, dicom_name, path_CT=path_CTs, path_liver = path_livers):
    mask_name = str(anon_id+"_"+dicom_name)
    path_CT = os.path.join(path_CT,anon_id)
    return path_CT, dicom_name, path_liver, mask_name

for i, ct_name in enumerate (ct_names):
    path_ct,name_ct, path_liver, name_liver = path_gen(anon_ids[i],ct_name)
    obj_ct = Preprocessing(path_ct, name_ct)
    processed_scan = obj_ct.resample([obj_ct.img_spacing[0],obj_ct.img_spacing[1],2.0])
    processed_scan = obj_ct.windowing(processed_scan, [-150, 250], [0, 256])

    obj_liver = Preprocessing(path_liver,name_liver)
    processed_mask = obj_liver.resample([obj_liver.img_spacing[0],
                                         obj_liver.img_spacing[1],
                                         2.0])
    if processed_scan.GetSize()==processed_mask.GetSize():
        sitk.WriteImage(processed_scan, os.path.join(scan_savepath, name_liver))
        sitk.WriteImage(processed_mask, os.path.join(livermask_savepath, name_liver))
    else:
        print("{} have a different size scan and mask after resampling".format(name_liver))
        print("scan:{};mask:{}".format(processed_scan.GetSize(),processed_mask.GetSize()))






