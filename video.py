import sys
import torch
import torchvision
import torch.onnx
import echonet
from echonet import datasets

sys.path.append('/path/to/file') #paths are different for the demonstration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

test_dataset = echonet.datasets.Echo(split = 'val', target_type = 'EF', clips = 'all', mean = mean, std = std, length = 32, period = 2, max_length = 250, pad =None)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, num_workers = 1, shuffle = False, pin_memory = False)

input = next(iter(test_dataloader))

model = torchvision.models.video.__dict__['deepLabV3'](pretrained=False)

model.fc = torch.nn.Linear(model.fc.in_features, 1)

model = torch.nn.DataParallel(model)

onnxin = input[0]

model.to(device)

weights = '/home/georgios/dynamic/output/video/deepLabV3_pretrained/checkpoint.pt'

checkpoint = torch.load(weights)
    
model.load_state_dict(checkpoint['state_dict'])

model.eval()

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
