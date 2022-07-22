from window_resample import Preprocessing
import pandas as pd
import os
import SimpleITK as sitk

scan_dic = pd.read_excel('D:/data/recurrence/Selected_CT_for_analysis.xlsx')
scan_readpath = "D:/data/recurrence/scans/original_scan"
scan_savepath = "D:/data/recurrence/scans/processed_scan"
ct_names = scan_dic['CT_scan_name'].values
addinfo_dic = {}
for ct_name in ct_names:
    if not pd.isna(ct_name):
        obj = Preprocessing(scan_readpath,ct_name)
        processed_scan = obj.resample([obj.img_spacing[0],obj.img_spacing[1],2.0])
        processed_scan = obj.windowing(processed_scan,[-150,250],[0,256])
        addinfo_dic[ct_name]= [processed_scan.GetSize(),obj.img_size, obj.img_spacing, obj.img_direction, obj.img_origin]
        sitk.WriteImage(processed_scan,os.path.join(scan_savepath,str(ct_name+'.nii')))

df = pd.DataFrame(data=addinfo_dic)
df.to_excel('D:/data/recurrence/Selected_CT_addinfo.xlsx')
#obj.histogram_plot(processed_scan)