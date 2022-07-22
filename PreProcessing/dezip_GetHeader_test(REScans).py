import dezip_GetHeader as DG
import os
import pydicom
import SimpleITK as sitk

unzip_path = "C:/Users/yyc13/recurrence/data/CT_scan"
zip_path = "C:/Users/yyc13/recurrence/data/Yin"
zip_files = os.listdir(zip_path)

for zip_file in zip_files[10:]: #this ts the unzip files per patient
    DG.un_zip(zip_path, zip_file, unzip_path)
    file = str(zip_file.split(".")[0]+"_CT")
    pid = file.split("_")[0]
    CT_dates = os.listdir(os.path.join(unzip_path,file,pid))
    for CT_date in CT_dates:
        path_ctdate = os.path.join(unzip_path,file,pid,CT_date)
        reader = sitk.ImageSeriesReader()
        seriesIDs = reader.GetGDCMSeriesIDs(os.path.join(path_ctdate,"CT"))
        for i,seriesID in enumerate(seriesIDs):
            dicom_names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(os.path.join(path_ctdate,"CT"),seriesID)
            headfile = pydicom.dcmread(os.path.join(os.path.join(path_ctdate,"CT"),dicom_names[0]))
            scan_name = "{}_{}_CT{}".format(pid,CT_date,i)
            if '512' in str(headfile[0x0028,0x0010]):
            ###save headfile and nifti
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                if image.GetSize()[2]>5:
                    DG.save_image(image,path_ctdate,scan_name)
                    SeriesDescription = DG.get_header(os.path.join(path_ctdate,"CT"),dicom_names[0],path_ctdate,scan_name)
                    print(scan_name, "  ", SeriesDescription)
            else:
                print("{} is not in the shape of 512,512, SITK cannot read it.".format(scan_name))

