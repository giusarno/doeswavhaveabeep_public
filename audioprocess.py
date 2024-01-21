import torch
import torchaudio
import matplotlib.pyplot as plt
import logging
import librosa

import torchaudio.functional as F
import torchaudio.transforms as T
import torch.nn.functional as TF

log = logging.getLogger(__name__)

def load_sount_effect(effect,sample_rate):
    isfound=False
    if effect == "background_voices":
            wav,sr = torchaudio.load('./std_effects/background_voices.wav')
            log.debug(f'loading background_voices......')
            isfound=True
    else:
            log.debug(f'Sound effect {effect} NOT FOUND ......')
    if isfound:
        if sr != sample_rate:
                resampler = T.Resample(sr,sample_rate)
                wav = resampler(wav)
        #wav = (wav[:, : inputshape])*snr
    return wav,isfound

def add_sound_effect(wave,sf,snr):
    soundeffect = sf
    soundeffect = (soundeffect[:, : wave.shape[1]])*snr
    return soundeffect+wave
    
def preprocess_wav(wave,sample_rate,snr):
    noise,sr = torchaudio.load("./std_effects/background_voices.wav")
    if sr != sample_rate:
            resampler = T.Resample(sr,sample_rate)
            noise = resampler(noice)
    noise = (noise[:, : wave.shape[1]])*snr
    return noise, noise+wave 

def plot_multiple_waveform(waveforms,sample_rate,titlein="Waveform"):
    log.debug("plot_multiple_waveform...start")
    nslices=waveforms.size(1)
    #print(f'nslice = {nslices}')
    for i in range (nslices):
        log.debug("slice n {}".format(i))
        log.debug("the waveform {}".format(waveforms.select(1,i)))
        plot_waveform(waveforms.select(1,i),sample_rate,title=titlein)
    log.debug("plot_multiple_waveform...end")
    return

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):

    log.debug("plot_waveform...start")
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    log.debug("plot_waveform...stop")

def plot_waveform_withline(waveform, sample_rate, secs,text, title="Waveform", xlim=None, ylim=None):

    log.debug("plot_waveform...start")
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].axvline(x=secs, ymin=0.05, ymax=0.95, color='purple',  ls='--', lw=2, label=text)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    log.debug("plot_waveform...stop")

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()

def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")

def save_audio_from_file(audio,rate,filepath):

    log.debug("save_audio_from_file...start")
    log.debug(f'audio shape {audio.shape} rate {rate} path {filepath}')

    torchaudio.save(filepath,audio,rate)


def load_audio_from_file(filepath):

    log.debug("load_audio_from_file...start")

    audio, sample_rate = torchaudio.load(filepath)

    nchannels = audio.size(dim=0)

    if (nchannels > 1):
        log.debug("nofchannels = {}".format(nchannels))
        #print(audio.shape)
        audio = audio[:1]
        log.debug("new shape {}".format(audio.shape))
    # print duration in seconds:
    audio_len = audio.size(dim=1)
    duration = audio_len/ float(sample_rate)*1000

    #print(f'audio len = {audio_len} sameple_rate {sample_rate}')
    #print_stats(audio,sample_rate=sample_rate)
    log.debug("load_audio_from_file...stop")

    return audio,sample_rate,duration

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):

    plt.figure()
    plt.title(title)
    plt.imshow(spec.log2()[0, :, :].numpy())
    plt.show()

def slice_the_audio(audio,sample_rate,window,stride):
    log.debug("slice_the_audio...start")

    actualwin = round(window * sample_rate)
    actualstr = round(stride * sample_rate)

    log.debug(f'actualwin = {actualwin} actualstr {actualstr}')
        
    aus = audio.size(dim=1)
    
    log.debug(f'tensor length {aus}')

    
    if aus < actualwin:
        audio = waveform_padding(audio,actualwin-aus)
    
    log.debug("slice_the_audio...stop")
    
    return audio.unfold(1,actualwin,actualstr)

def slice_the_audio_w_0_pad(audio,sample_rate,window,stride):

    log.debug("slice_the_audio_w_0_pad...start")
    actualstr = round(stride * sample_rate)
    actualwin = round(window * sample_rate)
    result = slice_the_audio(audio,sample_rate,window,stride)
    noslices= result.size(1)
    length= audio.size(1)

    log.debug("original audio waveform {}".format(audio))
    log.debug('shape of the result slices {}'.format(result.shape))

    log.debug("number of slices {}".format(noslices))
    log.debug("stride {}".format(actualstr))
    log.debug("window {}".format(actualwin))
    log.debug("how many strides for the windows and the audio length {}".format((length - actualwin)/actualstr))

    if noslices == (length - actualwin)/actualstr:
        log.debug("noslices matches exatly the length of the audio")
        return result
    log.debug("noslices does NOT matches exatly the length of the audio")

    nextstr = noslices * actualstr
    log.debug("position of the next stride {}".format(nextstr))
    padlen = nextstr + actualwin - length

    tenofnextslice= audio[0][ nextstr : length]
    #tensor1= audio[0][1 : length]
    log.debug("next slice from the postion of the next stride to the length of the shape {} audio {}" .format(tenofnextslice.shape,tenofnextslice))

    temptensor = waveform_padding(tenofnextslice,padlen)
    temptensor = torch.unsqueeze(temptensor, dim=0)
    temptensor = torch.unsqueeze(temptensor, dim=1)


    log.debug("the next slice {}".format((temptensor)))
    log.debug("the next slice shape {}".format((temptensor.shape)))

    final = torch.cat((result,temptensor),1)
    log.debug("new set of slices {}".format(final))
    log.debug("new set of slices shape {}".format( final.shape))
    log.debug("slice_the_audio_w_0_pad...stop")

    return final,final.size(1)

def waveform_padding(audio,length):
    log.debug("waveform_padding...start")

    log.debug("audio shape {}".format(audio.shape))
    log.debug("audio {}".format(audio))

    result = TF.pad(audio,(0,length))
    log.debug("waveform_padding...stop")

    return result
def waveform_padding_before(audio,length):
    log.debug("waveform_padding...start")

    log.debug("audio shape {}".format(audio.shape))
    log.debug("audio {}".format(audio))

    result = TF.pad(audio,(length,0))
    log.debug("waveform_padding...stop")

    return result

def resample_wav_plus_mono(wav,old_sr,new_sr):
    resample_transform = torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=new_sr)    
    audio_mono = torch.mean(resample_transform(wav),dim=0, keepdim=True)
    return audio_mono, new_sr
    



