import dezip_GetHeader as dg
import SimpleITK as sitk
import os
import zipfile

file_path = 'D:/data/recurrence/scans/public_data/3Dircadb1'
file_folders = os.listdir('D:/data/recurrence/scans/public_data/3Dircadb1')
scan_savepath = 'D:/data/recurrence/scans/public_data/scan'
label_savepath = 'D:/data/recurrence/scans/public_data/maskliver'
for file_folder in file_folders:
    zip_scanpath = os.path.join(file_path, file_folder, 'PATIENT_DICOM'+'.zip')
    zip_scanfile = zipfile.ZipFile(zip_scanpath)
    if os.path.isdir(os.path.join(scan_savepath,file_folder)):
        pass
    else:
        os.mkdir(os.path.join(scan_savepath,file_folder))
        for names in zip_scanfile.namelist():
            zip_scanfile.extract(names, os.path.join(scan_savepath,file_folder))
    zip_labelpath = os.path.join(file_path,file_folder,'MASKS_DICOM'+'.zip')
    zip_labelfile = zipfile.ZipFile(zip_labelpath)
    if os.path.isdir(os.path.join(scan_savepath,file_folder,'MASKS_DICOM')):
        pass
    else:
        for names in zip_labelfile.namelist():
            if '/liver/' in names:
                zip_labelfile.extract(names,os.path.join(scan_savepath,file_folder))

##################
dicom_path = 'D:/data/recurrence/scans/public_data/scan'
dicom_folders = os.listdir('D:/data/recurrence/scans/public_data/scan')
nifti_savepath = 'D:/data/recurrence/scans/public_data'

for dicom_folder in dicom_folders:
    dicomscan_path = os.path.join(dicom_path,dicom_folder,'PATIENT_DICOM')
    livermask_path = os.path.join(dicom_path,dicom_folder,'MASKS_DICOM','liver')
    scan_itk = dg.dicom_reader(dicomscan_path)
    liver_itk = dg.dicom_reader(livermask_path)
    if scan_itk.GetSize() == liver_itk.GetSize():
        save_name = 'scan_'+dicom_folder.split('.')[1]+'.nii'
        print(save_name)
        sitk.WriteImage(scan_itk,os.path.join(nifti_savepath,'scan_nifti',save_name))
        sitk.WriteImage(liver_itk, os.path.join(nifti_savepath, 'maskliver_nifti', save_name))
