import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk

class registration_files():
    def __init__(self, scan_path, mask_path, scan_savepath, parameter_path1, parameter_path2,command_savepath,
                    scan_list):
        '''
        initialize the paths and scan list by init_list and init_path before generating command line
        '''
        self.scan_path = scan_path
        self.mask_path = mask_path
        self.scan_savepath = scan_savepath
        self.parameter_path1 = parameter_path1
        self.parameter_path2 = parameter_path2
        self.command_savepath = command_savepath
        self.scan_list = scan_list

class elastix_command(registration_files):

    def command_line(self, fscan_name, mscan_name, save_path):
        '''
        command line generator
        '''
        line = 'elastix -f {} -m {} -out {} -p {} -p {}'.format(
            os.path.join(self.scan_path, fscan_name),
            os.path.join(self.scan_path, mscan_name),
            save_path,
            self.parameter_path1,
            self.parameter_path2
        )
        return line

    def info_extract(self, name):
        '''
        This function is to extract the patient id, scan date and the scan index from self.scan_list
        It can be changed according to the format given in scan_list
        '''
        pt, date, ind = name.split('.')[0].split('_')
        return pt, date, ind

    def dictionary_gen(self):
        '''
        Generate a dictionary of entire scan list, the structure is:
        {'RE001': {'20150827': ['CT2'], '20160708': ['CT1'], '20161117': ['CT1']},
         'RE002': {...}
         ...}
        '''
        info_dic = {}
        for scan in self.scan_list:
            pt,date,ind = self.info_extract(scan)
            if not (pt in info_dic):
                info_dic[pt] = {date:[ind]}
            elif not (date in info_dic[pt]):
                info_dic[pt][date] = [ind]
            else:
                info_dic[pt][date].append(ind)
        return info_dic

    def command_line_generator(self, if_new = True):
        '''
        enumerate all possible registration pairs and generate the elastix command line
        if_new: True then generate the folder to save results, False remove the old results to empty the folder
        return the list of all elastix command line
        '''
        info_dic = self.dictionary_gen()
        lines = []
        for pt in info_dic:
            if if_new:
                os.mkdir(os.path.join(self.scan_savepath, pt))
            dates = list(info_dic[pt].keys())
            for i in range(0, len(dates)-1):
                for j in range(i+1, len(dates)):
                    date1_cts = info_dic[pt][dates[i]]
                    date2_cts = info_dic[pt][dates[j]]
                    for date1_ct in date1_cts:
                        for date2_ct in date2_cts:
                            ite_savepath = os.path.join(self.scan_savepath, pt,
                                                '{}{}_{}{}'.format(dates[i],date1_ct,dates[j],date2_ct))
                            if if_new:
                                os.mkdir(ite_savepath)
                            else:
                                shutil.rmtree(ite_savepath)
                                os.mkdir(ite_savepath)
                            line = self.command_line(str(pt+'_'+dates[i]+'_'+date1_ct+'.nii'),
                                                    str(pt+'_'+dates[j]+'_'+date2_ct+'.nii'), 
                                                    ite_savepath)
                            lines.append(line)
        return lines
    
    def save_command_line(self):
        '''
        save the elastix command lines as txt file (save later in bat and run in powershell)
        '''
        lines = self.command_line_generator()
        with open(self.command_savepath,"w") as f:
            for Wline in lines:
                f.write(Wline)
                f.write('\n')
    


class registered_img_processor(registration_files):
    '''
    this class generate diff mask between registered images, 
    and checked the generated diff mask of entire registered image dateset by plotting
    '''

    def plot_registered_img(self, title, f_arr, overlay_arr):
        '''
        given the fixed image, moving image(overlay_arr), plt both and a overlayed scan
        '''
        fig,axarr = plt.subplots(3,3)
        fig.suptitle(title)
        fig.set_size_inches(10, 10)
        axarr[0,0].imshow(f_arr[:,:,int(f_arr.shape[-1]/4)], cmap='gray')
        axarr[0,1].imshow(overlay_arr[:,:,int(f_arr.shape[-1]/4)], cmap='gray')
        axarr[0,2].imshow(f_arr[:,:,int(f_arr.shape[-1]/4)], cmap='gray')
        axarr[0,2].imshow(overlay_arr[:,:,int(overlay_arr.shape[-1]/4)], cmap='jet',alpha = 0.5)

        axarr[1,0].imshow(f_arr[:,:,int(f_arr.shape[-1]*2/4)], cmap='gray')
        axarr[1,1].imshow(overlay_arr[:,:,int(overlay_arr.shape[-1]*2/4)], cmap='gray')
        axarr[1,2].imshow(f_arr[:,:,int(f_arr.shape[-1]*2/4)], cmap='gray')
        axarr[1,2].imshow(overlay_arr[:,:,int(overlay_arr.shape[-1]*2/4)], cmap = 'jet', alpha=0.5)

        axarr[2,0].imshow(f_arr[:,:,int(f_arr.shape[-1]*3/4)], cmap='gray')
        axarr[2,1].imshow(overlay_arr[:,:,int(overlay_arr.shape[-1]*3/4)], cmap='gray')
        axarr[2,2].imshow(f_arr[:,:,int(f_arr.shape[-1]*3/4)], cmap='gray')
        axarr[2,2].imshow(overlay_arr[:,:,int(overlay_arr.shape[-1]*3/4)], cmap='jet', alpha=0.5)
        plt.show()
    
    def hist_plot(self, img_arr):
        '''
        plot histogram of a img of (0-255) graylevel
        '''
        grayDict={}
        for key in range(1,256):
            grayDict[key] = np.sum(img_arr[:,:,:]==key)
        plt.figure(dpi = 300)
        plt.xticks(range(0,255,25))
        plt.bar(list(grayDict.keys()),list(grayDict.values()))
        plt.show()

    def smoothing(self, img_itk):
        '''
        smooth the diff mask
        '''
        rgsmootherfilter = sitk.SmoothingRecursiveGaussianImageFilter()
        rgsmootherfilter.SetSigma(2.5)
        rgsmootherfilter.SetNormalizeAcrossScale(True)
        rgsmoothedimage  = rgsmootherfilter.Execute(img_itk)
        return rgsmoothedimage
    
    def mask_generator(self, fname, mname_info, 
        if_smooth = True, if_plot = True, if_hist = True, if_mask = True):
        '''
        Generate the diff mask according to the registered images
        if_plot: True to plot some slices of the fimg and mimg and overlayed diff mask
        if_hist: True to plot the histogram of diff mask
        '''
        fitk = sitk.ReadImage(os.path.join(self.scan_path, fname))
        mitk = sitk.ReadImage(os.path.join(self.scan_savepath, mname_info[0] ,mname_info[1], mname_info[2]))
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(sitk.sitkUInt8)
        mitk = cast_filter.Execute(mitk)
        marr = sitk.GetArrayFromImage(mitk)
        if if_mask:
            fmask_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mask_path,fname)))
            marr[fmask_arr[:,:,:]==0] = 0
        subimg_arr = sitk.GetArrayFromImage(fitk) - marr
        subimg_itk = sitk.GetImageFromArray(subimg_arr) 
        subimg_itk.SetOrigin(fitk.GetOrigin())
        subimg_itk.SetDirection(fitk.GetDirection())
        subimg_itk.SetSpacing(fitk.GetSpacing())
        if if_smooth:
            subimg_itk = self.smoothing(subimg_itk)
        if if_plot:
            self.plot_registered_img(fname, sitk.GetArrayFromImage(fitk).transpose([1,2,0]), subimg_arr.transpose([1,2,0]))
        if if_hist:
            subimg_arr[fmask_arr[:,:,:]==0] = 0
            self.hist_plot(subimg_arr)
        return subimg_itk


        



    

