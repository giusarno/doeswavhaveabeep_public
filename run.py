import audioprocess as AP
import audio_csv_util as CV
import tensor_util as TU
import argparse

import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

moduletolog = ['__main__','audioprocess','tensor_util','audio_csv_util']

# create file handler which logs even debug messages
fh = logging.FileHandler('runtime.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
#for module in moduletolog:
#    print(module)
#    fh.addFilter(logging.Filter(module))
#    ch.addFilter(logging.Filter(module))

logger.addHandler(fh)
#logger.addHandler(ch)
log = logger

def process_master_file_record(rootpath,outputpath,recordname,ext,startbeep,stopbeep,window,stride,shift,targetsamplerate,addeffect=False):

    filename = rootpath+recordname+"."+ext

    log.info(f'Processing file {filename} startbeep {startbeep} stopbeep {stopbeep} window {window} stride {stride} shift {shift}')

    waveformpre, sample_rate, duration = AP.load_audio_from_file(filename)
    waveformpre, sample_rate = AP.resample_wav_plus_mono(waveformpre,sample_rate,targetsamplerate)
    
    realshift = round(shift*sample_rate)
    realstartbeep = (startbeep * sample_rate) - realshift
    realstopbeep =  (stopbeep * sample_rate) - realshift


    #What is the file is too smal.
    audiolength= waveformpre.size(1)

    if (startbeep > 0 ) and ((startbeep * sample_rate) < (window *sample_rate)):
        log.info(f'startbeep {startbeep} < window {window}')
        #AP.plot_waveform(waveformpre,sample_rate,recordname)
        waveformpre = AP.waveform_padding_before(waveformpre,int((window-startbeep)*sample_rate))
        #AP.plot_waveform(waveformpre,sample_rate,recordname)

    waveform = TU.sub_from_begin_2nd_tensor(waveformpre,realshift)

    outputshiftedfile = outputpath+recordname+"_shift_"+str(realshift)+"_sr_"+str(sample_rate)+"_test."+ext
    #noise, waveform = AP.preprocess_wav(waveform,sample_rate,10)
    isfound = False
    if addeffect:
        ef,isfound = AP.load_sount_effect("background_voices",targetsamplerate)
        
    AP.save_audio_from_file(waveform,sample_rate,outputshiftedfile)
    
    #wavform_new = AP.resample_wav_plus_mono(waveform,sample_rate,targetsamplerate)
    #outputshiftedfile_new = outputpath+recordname+"_shift_"+str(realshift)+"_sr_"+str(targetsamplerate)+"_test."+ext
    #AP.save_audio_from_file(wavform_new,sample_rate,outputshiftedfile_new)
    
    
    log.debug(f'waveform shape {waveformpre.shape} after shift {waveform.shape}')



    log.info(f'Processing file {filename} sample rate {sample_rate} duration {duration} milli seconds')

    slices,number = AP.slice_the_audio_w_0_pad(waveform,sample_rate,window,stride)
    log.info(f'Generated {number} slices')

    #outfileroot = rootpath+recordname

    csvfilenames = []
    arebeep = []

    for i in range (number):
        log.debug("slice n {}".format(i))
        log.debug("the slice {}".format(slices.select(1,i)))
        wavtensor = slices.select(1,i)
        outputfile = outputpath+recordname+"_shift_"+str(realshift)+"_sr_"+str(sample_rate)+"_slice_"+str(i)+"."+ext
        log.debug(f"appending filename {outputfile} sample_rate {sample_rate}")
        csvfilenames.append(outputfile)
        #noise, signal = AP.preprocess_wav(wavtensor,sample_rate,2)
        #print('beforenoise')
        #AP.plot_waveform(wavtensor,sample_rate)
        #wavtensor = signal
        if isfound:
            wavtensor = AP.add_sound_effect(wavtensor,ef,2)
            #print('noise')
            #AP.plot_waveform(ef,sample_rate)
        #AP.plot_waveform(wavtensor,sample_rate)

        AP.save_audio_from_file(wavtensor,sample_rate,outputfile)
        
        #wavtensor_new = AP.resample_wav_plus_mono(wavtensor,sample_rate,targetsamplerate)
        #outputfile_new = outputpath+recordname+"_shift_"+str(realshift)+"_sr_"+str(targetsamplerate)+"_slice_"+str(i)+"."+ext
        #AP.save_audio_from_file(wavtensor_new,sample_rate,outputfile_new)
        
        startwindow = i * (stride * sample_rate)
        endwindow = startwindow + (window * sample_rate)

        isbeep = True
        if (realstopbeep < startwindow) or  (realstartbeep > endwindow ):
            isbeep = False

        log.debug(f'realstartbeep {realstartbeep} realstopbeep {realstopbeep} startwindow {startwindow} endwindow {endwindow} isbeep {isbeep}')
        arebeep.append(isbeep)

    result = {'slices':csvfilenames,'isbeep':arebeep}
    return result

def process_master_file(file,inputdir,outputdir,winlength,stride,shifts,targetsamplerate,addeffect=False):
    global outputdirename
    print(outputdirename)
    filenames = CV.read_audio_info_from_csv(file)
    first = True
    inputdir="./"+inputdir+"/"
    outputdir="./"+outputdir+"/"
    
    for shift in shifts:
        log.info(f"starting Shift {shift}")
        for index,row in filenames.iterrows():

            filename_all =  row["filename"]
            startbeep = row["startbeep"]
            stopbeep =  row["stopbeep"]

            filename = filename_all.split('.')[0]
            ext = filename_all.split('.')[1]

            result = process_master_file_record(inputdir,outputdir,filename,ext,startbeep,stopbeep,winlength,stride,shift,targetsamplerate,addeffect)

            csvfile =outputdir+"wavlist.csv"
            
            if first:
                CV.write_audio_info_to_csv(csvfile,result)
                first = False
            else:
                CV.append_audio_info_to_csv(csvfile,result)



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--mode", default="train", help="data generation mode")
    parser.add_argument("-ct", "--cleantarget", help="Clean target directory",action="store_true")
    parser.add_argument("-w", "--window", default=2, help="overal window")
    parser.add_argument("-s", "--stride", default=0.2, help="the stride")

    args = parser.parse_args()

    # Set up parameters
    mode = args.mode
              
    outputdirename = "outputpretrainwav"
    inputdirname = "inputrawwav"
    outputdirenameval = "outputpretrainwav_val"
    inputdirnameval = "inputrawwav_val"
    
    win = float(args.window)
    stride = float(args.stride)
    
    print(f'window {win} stride {stride}')
    addeffect = True
    if mode == 'val':
        outputdirename = outputdirenameval
        inputdirname = inputdirnameval
        addeffect= False
    elif mode == 'train':
        inputdirname = inputdirname
        outputdirename =outputdirename
    else:
        sys.exit(f'mode value {mode} does not exists')  
    
    
    
    CV.create_dir_if_not_exists("./"+outputdirename)
    
    log.info(f'mode {mode} output folder {outputdirename} input folder {inputdirname}')

    if args.cleantarget:
        CV.remove_files_in_dir(outputdirename)
        log.info(f'Cleantarget: cleaning output folder {outputdirename} ')

    

    process_master_file("./"+inputdirname+"/wavlist.csv",inputdirname,outputdirename,win,stride,[0,0.03,0.05,0.07],16000,addeffect)
    #process_master_file("./"+inputdirname+"/wavlist.csv",inputdirname,outputdirename,2,0.1,[0,0.1,0.2],32000)






#log.debug (slice2.shape)


#AP.plot_multiple_waveform(slice2,sample_rate2)

#ten =AP.waveform_padding(waveform2,1)

#print (ten.shape)
#print (ten)

