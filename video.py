import sys
import torch
import torchvision
import torch.onnx
import echonet
from echonet import datasets

sys.path.append('/path/to/file') #paths are different for the demonstration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #enable gpu cuda accelarator if possible

mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train")) #MAE and STD from the dataset based on echonet algorithm

test_dataset = echonet.datasets.Echo(split = 'val', target_type = 'EF', clips = 'all', mean = mean, std = std, length = 32, period = 2, max_length = 250, pad =None) #set test dataset

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, num_workers = 1, shuffle = False, pin_memory = False) #load test dataset

input = next(iter(test_dataloader)) #initialize iteration for the dataset

model = torchvision.models.video.__dict__['deepLabV3'](pretrained=False) #set torchvision model for video analysis

model.fc = torch.nn.Linear(model.fc.in_features, 1) #set activation function for the nn model

model = torch.nn.DataParallel(model) #set data parallelism at the module level

onnxin = input[0] #start iteration

model.to(device) #pass model to device for conversion / cuda if gpu enabled
print(model)

weights = '/home/georgios/dynamic/output/video/deepLabV3_pretrained/checkpoint.pt' #set initial weights for the nn model

checkpoint = torch.load(weights)
    
model.load_state_dict(checkpoint['state_dict']) #load weights

model.eval() #convert model to inference mode after training

#create onnx inference model
torch.onnx.export(model.module, 
                      
                onnxin[:,0,...].to(device),
                      
                "video.onnx",
                      
                export_params = True,
                      
                opset_version = 10,
                      
                do_constant_folding = True,
                      
                input_names = ['input'],

                output_names = ['output']
                )

print("Model has been converted to onnx format")
