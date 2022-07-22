import os
import SimpleITK as sitk
from window_resample import Preprocessing


scan_path = 'D:/data/recurrence/scans/public_data/scan_nifti'
livermask_path = 'D:/data/recurrence/scans/public_data/maskliver_nifti'
nifti_files = os.listdir(scan_path)
scan_savepath = 'D:/data/recurrence/scans/public_data/processed_scan'
liver_savepath = 'D:/data/recurrence/scans/public_data/processed_maskliver'

for nifti_file in nifti_files:
    obj_scan = Preprocessing(scan_path,nifti_file)
    processed_scan = obj_scan.resample([obj_scan.img_spacing[0],
                                         obj_scan.img_spacing[1],
                                         2.0])
    processed_scan = obj_scan.windowing(processed_scan,[-150,250],[0,256])

    obj_liver = Preprocessing(livermask_path,nifti_file)
    processed_liver = obj_liver.resample([obj_liver.img_spacing[0],
                                         obj_liver.img_spacing[1],
                                         2.0])
    if processed_scan.GetSize()==processed_liver.GetSize():
        sitk.WriteImage(processed_scan,os.path.join(scan_savepath,nifti_file))
        sitk.WriteImage(processed_liver,os.path.join(liver_savepath,nifti_file))
