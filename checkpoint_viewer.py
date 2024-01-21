import torch
import torchaudio
import torchvision
import os
import argparse

def save_model(model,location):
    torch.save(model, location)

def load_ckp(checkpoint):
    #f_path = './checkpoint/checkpoint.pt'
    checkpoint = torch.load(checkpoint)
    #model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['state_dict'], checkpoint['optimizer'], checkpoint['epoch'],checkpoint['loss'],checkpoint['epochloss'],checkpoint['n0'],checkpoint['n1'],checkpoint['nsp0'],checkpoint['nsp1'],checkpoint['n']

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ck", "--checkpoint", default="checkpoint.pt", help="checkpoint file")
    parser.add_argument("-gm", "--get_model", default="model_default.pth", help="model file")

    args = parser.parse_args()

    # Set up parameters
    checkpoint = args.checkpoint
    modelfile = args.get_model

    if os.path.isfile(checkpoint):
        print(f"Loading checkpoint file:{checkpoint} ....")
        model, optimiser, start_epoch, lowestloss,epochloss,n0,n1,nsp0,nsp1,n = load_ckp(checkpoint)
        print("start_epoch",start_epoch)
        print("lowestloss",lowestloss)
        print("epochloss")
        for l in epochloss:
            print(l)
        print(n0,n1,nsp0,nsp1,n)
        
        if args.get_model is not None:
            save_model(model,modelfile)
        
    else:
        print (f"checkpoint file {checkpoint} does not exist !!")



