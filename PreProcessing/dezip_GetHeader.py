import os
import zipfile
import pydicom
import csv
import SimpleITK as sitk

def un_zip(path,file_name,save_path):
    '''
    This function is used to unzip the recurrence patients from umcg
    '''
    # file_name = file_name.split(".")[0]
    zip_file = zipfile.ZipFile(os.path.join(path,file_name))
    if os.path.isdir(os.path.join(save_path, str(file_name.split(".")[0]+"_CT"))):
        pass
    else:
        os.mkdir(os.path.join(save_path, str(file_name.split(".")[0] + "_CT")))
        for names in zip_file.namelist():
            zip_file.extract(names, os.path.join(save_path, str(file_name.split(".")[0] + "_CT")))


def get_header(dicom_path, dicom_name, save_path, save_id):
    headfile = pydicom.dcmread(os.path.join(dicom_path,dicom_name))
    with open(os.path.join(save_path,str(save_id+'.csv')),'w') as csvfile:
        writer = csv.writer(csvfile)
        for line in headfile:
            des_str = ''
            if ')' in str(line.description):
                des_str = str(line.description).split(")")[1]
                value = des_str.split(":")[-1]
                name = des_str.split(":")[0]
            writer.writerow([str(line.tag), str(line.VR), name, value])
    return str(headfile.SeriesDescription)
def save_image(img_itk,save_path,save_name):
    # img_itk = windowing(img_itk)
    sitk.WriteImage(img_itk,os.path.join(save_path,save_name+".nii"))

def dicom_reader(path):
    '''
    This func reads a dicom serier, but only one series in the folder,
    multiple series require to get series ID
    '''
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dcm_names)
    img_itk = reader.Execute()
    return img_itk