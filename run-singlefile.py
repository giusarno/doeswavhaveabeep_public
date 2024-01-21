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

def process_master_file_record(file,outputpath,window,stride,targetsamplerate):
    
    log.info(f'Processing file {file} window {window} stride {stride} ')
    outputpath = outputpath+"/"
    basefile = os.path.basename(file)    
    filename =basefile.split('.')[0]
    ext = basefile.split('.')[1]
    
    waveformpre, sample_rate, duration = AP.load_audio_from_file(file)
    waveformpre, sample_rate = AP.resample_wav_plus_mono(waveformpre,sample_rate,targetsamplerate)

    realshift = 0

    waveform = TU.sub_from_begin_2nd_tensor(waveformpre,realshift)
    
    
    log.debug(f'waveform shape {waveformpre.shape} after shift {waveform.shape}')
    log.info(f'Processing file {filename} sample rate {sample_rate} duration {duration} milli seconds')

    slices,number = AP.slice_the_audio_w_0_pad(waveform,sample_rate,window,stride)
    log.info(f'Generated {number} slices')

    csvfilenames = []
    arebeep = []

    for i in range (number):
        log.debug("slice n {}".format(i))
        log.debug("the slice {}".format(slices.select(1,i)))
        wavtensor = slices.select(1,i)
        outputfile = outputpath+filename+"_shift_"+str(realshift)+"_sr_"+str(sample_rate)+"_slice_"+str(i)+"."+ext
        log.debug(f"appending filename {outputfile} sample_rate {sample_rate}")
        csvfilenames.append(outputfile)
        AP.save_audio_from_file(wavtensor,sample_rate,outputfile)
        #wavtensor_new = AP.resample_wav_plus_mono(wavtensor,sample_rate,targetsamplerate)
        #outputfile_new = outputpath+recordname+"_shift_"+str(realshift)+"_sr_"+str(targetsamplerate)+"_slice_"+str(i)+"."+ext
        #AP.save_audio_from_file(wavtensor_new,sample_rate,outputfile_new)

    result = {'slices':csvfilenames}
    return result

def process_master_file(file,outputdir,winlength,stride,targetsamplerate):

    result = process_master_file_record(file,outputdir,winlength,stride,targetsamplerate)

    csvfile =outputdir+"/wavlist.csv"

    CV.write_audio_info_to_csv(csvfile,result)




if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ct", "--cleantarget", help="Clean target directory",action="store_true")
    parser.add_argument("-f", "--filename", help="filename")
    parser.add_argument("-od", "--outputdir", help="outputdir")

    parser.add_argument("-sr", "--samplerate", help="samplerate")
    parser.add_argument("-w", "--window", default=2, help="overal window")
    parser.add_argument("-s", "--stride", default=0.2, help="the stride")

    args = parser.parse_args()
              

    win = float(args.window)
    stride = float(args.stride)
    file = args.filename
    samplerate = int(args.samplerate)
    outputdirename = args.outputdir
    
    print(f'filename {file} samplerate {samplerate} window {win} stride {stride}')
    
    CV.create_dir_if_not_exists("./"+outputdirename)
    
    log.info(f'output folder {outputdirename}')

    if args.cleantarget:
        CV.remove_files_in_dir(outputdirename)
        log.info(f'Cleantarget: cleaning output folder {outputdirename} ')


    process_master_file(file,outputdirename,win,stride,samplerate)
    
    print(f'processing done !!!')
    #process_master_file("./"+inputdirname+"/wavlist.csv",inputdirname,outputdirename,2,0.1,[0,0.1,0.2],32000)


#log.debug (slice2.shape)


#AP.plot_multiple_waveform(slice2,sample_rate2)

#ten =AP.waveform_padding(waveform2,1)

#print (ten.shape)
#print (ten)

