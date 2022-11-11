import registration_class
import os
import pandas as pd


scan_path = r'D:\recurrence_ControlGroup_nifti\processed_scan'
mask_path = r'D:\recurrence_ControlGroup_nifti\postprocess_liver_mask'
scan_savepath = r'D:\recurrence_ControlGroup_nifti\registered_scan'
parameter_path1 = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/RegAffineParams.txt'
parameter_path2 = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/RegBSplineParams.txt'
command_savepath = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/controlGroup_tester.txt'
scan_list = os.listdir(mask_path)


command = registration_class.elastix_command(scan_path, mask_path, scan_savepath, parameter_path1, parameter_path2,command_savepath,
                    scan_list)
command.save_command_line()


#####clean excel and extract contrast phase
def clean_file(dic_path = (r'F:\recurrence_ControlGroup(nifti)\control_group_scan_CTselected.xlsx', save_path = r'F:\recurrence_ControlGroup(nifti)\test.xlsx')
    dic = pd.DataFrame(pd.read_excel(dic_path))
    new_df = []
    phase = []
    arr = dic.values
    print(dic.columns)
    for index, rol in dic.iterrows():
        if rol['manual_info'] != 'ok':
            continue
        else:
            new_df.append(list(rol.values))
            if any(word in rol['series_description'] for word in ['PVP','pvp','Pvp','Abdomen','port', 'ven', 'Ven']):
                phase.append('V')
            elif any(word in rol['series_description'] for word in ['Art','art','ART','thor','Thor']):
                phase.append('A')
            elif any(word in rol['series_description'] for word in ['Laat','laat']):
                phase.append(('L'))
            else:
                phase.append('_')
    new_df = pd.DataFrame(new_df,columns=dic.columns)
    new_df['phase'] = phase
    new_df.to_excel(save_path)
