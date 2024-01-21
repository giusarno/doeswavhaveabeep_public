import torch
import torchaudio
from torch import nn
from torchvision import models
import audioprocess as AP
import tensor_util as TU
import queue
from datetime import datetime
import torchvision.transforms as T
import torchaudio.functional as F
import copy



class Net2(nn.Module):

    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=False)
        # Replace last layer
        #for param in self.network.parameters():
        #    param.requires_grad = False
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 2)
    def forward(self, xb):
        return self.network(xb)
    
def _freeze_norm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                #m.track_running_stats = False
                m.train()

    except ValueError:  
        print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
        return
    
def _set_batch_momentum(net,mom):
    try:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = mom
                #m.train()

    except ValueError:  
        print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
        return

def resample_if_necessary(signal,target_sample_rate, sr):
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal)
        return signal

def mix_down_if_necessary(signal):
        if signal.shape[0] > 1:
            print ("MULTIPLE CHANNELS")
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
def getfinalsignal(wavetensor,samplerate,transformation,debug,plot):
        
        signal = resample_if_necessary(wavetensor,samplerate,samplerate)
        signal = mix_down_if_necessary(signal)

        smax = (torch.max(signal))
        smin = (torch.min(signal))

        if  debug: 
            print(f'samplerate {samplerate}' )
            print (f' S Max {smax} S Min {smin}')

        if smax < 0.01 and smin > -0.01 :
                #print (signal)
                signal = signal.clone()
                signal[torch.logical_and(signal>=-0.01, signal<=0.01)] = 0
                #print (signal)
        if  plot:
            AP.plot_waveform(signal,samplerate)
        
        ns,wav = AP.preprocess_wav(signal,samplerate,2)
        pitch = F.detect_pitch_frequency(wav, samplerate)
        
        if  plot:
            AP.plot_waveform(wav,samplerate)
            AP.plot_waveform(pitch,samplerate)

        
        #signal = torch.round(signal, decimals=2) 
        
            
        signal = transformation(wav)

        fmax = (torch.max(signal))
        fmin = (torch.min(signal))
        signal = torch.round(signal, decimals=1)
        
        #signal = round(signal,1)
        #signal[signal < 200] = 0
        
        signal = signal.clone()

        #signal[torch.logical_and(signal>=0, signal<=1e-1)] = 1e-4 

        if signal.min() != signal.max():

            signal -= signal.min()
            signal /= signal.max()
        
        #if  plot:
        #    AP.plot_waveform(signal,samplerate)
        #signal = signal.log2()

        #self.fmax = (torch.max(signal))
        #self.fmin = (torch.min(signal))

        if debug:
            print (f' F Max {fmax} F Min {fmin}')
        if plot:
            print(f'SHAPE SIGNAL {signal.shape}')
            AP.plot_spectrogram(signal)
        preprocessedsignal = signal
        #print (signal.shape)
        signal = signal.repeat(3, 1, 1) 
        signal = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(signal)
        signal = signal[None]
        #print (signal.shape)
        return signal


def countvaluesinwin(slwin):
    count = 0
    slwinc = queue.Queue() 
    slwinc.queue = copy.deepcopy(slwin.queue)
    
    while not slwinc.empty():
        value = slwinc.get() 
        #print("value",value)
        if value == 0:
            count += 1
    return count
    
def preparedata (filename,samplerate,transformation,debug,plot):
            signal, sr = torchaudio.load(filename)
            return getfinalsignal(signal,samplerate,transformation,debug,plot)

def doeswavhavebeep(outputdir):
        annotations = pd.read_csv(outputdir+"/wavlist.csv")
        files = annotations['slices'].tolist()
        li = []
        for file in files:
            res = preparedata (file,16000,transform_spectra,False,False)
            predicted, expected,predictedsoft = predict(new_model_pred,res,0)
            predicted = int(predicted.int())
            print(f'file {file} predicted {predicted}')
            li.append(predicted)
        return scanfileforbeep(li)

def validate_model(model,data_loader,loss_fn,device,debug):
      train(model, data_loader, loss_fn, None , device,1,debug,True)
        
def predict(model, input, expected):
    with torch.no_grad():
        prediction = model(input)
        _, predicted = torch.max(prediction, 1) 
        predictedsoft = torch.softmax(prediction,1)
        #predictedsoft = predicted
    return predicted, expected ,predictedsoft

def scan_file_for_beep(model,device,file,window,stride,targetsamplerate,debug1,debug,plot,retimmediately=False):
    
    #print(f'Processing file {file} window {window} stride {stride} ')

    waveformpre, sample_rate, duration = AP.load_audio_from_file(file)
    #print(f'sample_rate {sample_rate} duration {duration}')
    waveformpre, sample_rate = AP.resample_wav_plus_mono(waveformpre,sample_rate,targetsamplerate)
    #print(f'sample_rate {sample_rate}')

    realshift = 0

    waveform = TU.sub_from_begin_2nd_tensor(waveformpre,realshift)
    slices,number = AP.slice_the_audio_w_0_pad(waveform,sample_rate,window,stride)
    
    #print(f'Generated {number} slices')
    
    tslices = []

    slidingwin = queue.Queue()

    slidingwin.put(1)
    slidingwin.put(1)
    slidingwin.put(1)
    slidingwin.put(1)
    slidingwin.put(1)
    
    istherebeep = False
    timeofbeep = -1
    
    dtstart = datetime.now()
    
    for i in range (number):
        wavtensor = slices.select(1,i)
        #tslices.append(wavtensor)
        signal = getfinalsignal(wavtensor,targetsamplerate,transform_spectra,debug,plot)
        signal = signal.to(device)
        predicted, expected,predictedsoft = predict(model,signal,0)
        predicted = int(predicted.int())
        
        win = window*targetsamplerate
        st = stride*targetsamplerate
        
        fwin = win+st*i
        time = fwin/targetsamplerate
        
        rem = slidingwin.get(False)
        slidingwin.put(predicted)
        countof0 = countvaluesinwin(slidingwin)
        if countof0 > 3:
            istherebeep = True
            if timeofbeep == -1:
                timeofbeep = time
                if retimmediately:
                    return istherebeep, timeofbeep
        if debug1:
            print (f'slice {i} beepinwindow {predicted==0} beepfound {istherebeep} time {time} timeofbeep {timeofbeep}' )
    
    dtstop = datetime.now()
    if retimmediately == False:
        AP.plot_waveform_withline(waveformpre,sample_rate,timeofbeep,"beep")
    if debug1:
        print(f'Processing Start {dtstart} Stop {dtstop} duration {dtstop-dtstart}')
    return istherebeep, timeofbeep

'''
def round(x, decimals=0):
    b = 10**decimals
    return torch.round(x*b)/b
'''
    
    
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

STD_MODEL_WEIGHTS="feedforwardnet.pth"
STD_MODEL_WEIGHTS_BEST='feedforwardnet-best.pth'
SAMPLE_RATE=16000
size=224

#device="cpu"
# instantiating our dataset object and create data loader
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    win_length = 2048,
    n_fft=2048,
    hop_length=256,
    n_mels=128,
    f_max=4500,
    f_min=200,
    normalized = False
)


transform_spectra = T.Compose([
    mel_spectrogram,
    T.Resize((size,size)),
])
 
        

#model_weights = STD_MODEL_WEIGHTS
model_weights =STD_MODEL_WEIGHTS_BEST
#model_weights = "feedforwardnet-v.0.0.1-cuda-batch-enable.pth"

new_state_dict = torch.load(model_weights)

new_model_pred= Net2().to(device)
new_model_pred.load_state_dict(new_state_dict)




files = {
        './testwav/machine-tape-035a.wav':'-1',
        './testwav/nextel.wav':'11.021',
        './testwav/sprint.wav':'20.690',
        './testwav/verizon.wav':'20.600'
         }
debug=False
single=False
new_model_pred.eval()
_freeze_norm_stats(new_model_pred)

count_faild=0
failed_files=[]

if single:
    isabeep, time = scan_file_for_beep(new_model_pred,device,File_name,0.7,0.08,16000,True,debug,debug,False)   
    print (f'File {File_name} isbeep {isabeep} predicted time {time}') 
else:
    tests = files.keys()
    for f in tests:
        File_name=f
        #isabeep, time = scan_file_for_beep(new_model_pred,File_name,1,0.15,16000,False,debug,debug,True)   
        isabeep, time = scan_file_for_beep(new_model_pred,device,File_name,0.7,0.08,16000,False,debug,False,True)   
        
        timediff=abs(float(time)-float(files[f]))
        
        if timediff > 0.08*5:
            count_faild=count_faild+1
            failed_files.append(File_name)
        print (f'File {File_name} isbeep {isabeep} predicted time {time} expected {files[f]}') 
    if count_faild > 0:
        print("Test run FAILED - n of TC failed & filenames",count_faild,failed_files)
    else:
        print("Test run SUCCEDED")
