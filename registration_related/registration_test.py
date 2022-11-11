import registration_class
import os

scan_path = 'C:/Users/yyc13/recurrence/data/processed_CTscan' 
mask_path = 'C:/Users/yyc13/recurrence/data/predict_postprocess_checked' 
scan_savepath = 'C:/Users/yyc13/recurrence/data/classtest'
parameter_path1 = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/RegAffineParams.txt'
parameter_path2 = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/RegBSplineParams.txt'
command_savepath = 'C:/Users/yyc13/recurrence/elastix-5.0.1-win64/elastix-5.0.1-win64/tester.txt'
scan_list = os.listdir(scan_path)[0:5]


command = registration_class.elastix_command(scan_path, mask_path, scan_savepath, parameter_path1, parameter_path2,command_savepath,
                    scan_list)
lines = command.command_line_generator()
print(lines)