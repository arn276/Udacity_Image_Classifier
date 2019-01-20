import argparse
import torch
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image

def command_line_args():
    '''Command line arguements.  Defaults are set but can be overruled.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default = '/home/workspace/aipnd-project/flowers/test/12/image_04014.jpg',
                        dest = 'image', help = 'Image to make a prediction')
    parser.add_argument('--labels', default = '/home/workspace/aipnd-project/cat_to_name.json',
                        dest = 'labels', help = 'Labels for the prediction')
    parser.add_argument('--topk', type = int, default = 5,
                        dest = 'topk', help = 'Return to k predictions')
    parser.add_argument('--json', type = str, 
                        help='JSON file containing label names')
    parser.add_argument('--gpu', action = 'store_false', 
                         dest = 'gpu', help = 'Train with GPU')
    args = parser.parse_args()
    return args

def load_checkpoint(file):
    '''Loading checkpoint file and the model.
    '''
    checkpoint = torch.load(file, map_location=lambda storage, loc: storage)
    # Checking architecture and loading the model.
    if checkpoint['arch'] == 'vgg16_bn': 
        model = models.vgg16(pretrained = True)
    else:
        model = models.alexnet(pretrained = True)
    print("'Model read in using " + checkpoint['arch'])
    for param in model.parameters():
        paramrequires_grad = False
    model.class_to_idx = checkpoint['image_datasets']
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'],checkpoint['hidden_layers'])),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(checkpoint['hidden_layers'],checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    model.classifier = classifier
    epoch = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer, epoch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    img_trans = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor()])
    test_image = img_trans(pil_image).float()
    np_image = np.array(test_image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229,0.224, 0.225])
    np_image = (np.transpose(np_image, (1,2,0)) - mean) / std
    np_image = np.transpose(np_image,(2,0,1))
    return np_image

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    # Pull in image and prepare for prediction
    np_image = process_image(image_path) 
    image_var = Variable(torch.FloatTensor(np_image), requires_grad = True)
    image_var = image_var.unsqueeze(0).float()
    if gpu:
        #change to CUDA
        model.cuda()
        image_var = image_var.to('cuda')
    output = model.forward(image_var)
    ps = torch.exp(output).data.topk(topk)
    if gpu:
        #change to CUDA
        probs = ps[0].cpu()
        classes = ps[1].cpu()
    else:
        probs = ps[0]
        classes = ps[1]
    # Setting the top class predictions
    class_convert = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_convert[label])
    top_probability, top_classes = probs.numpy()[0], mapped_classes
    max_index = np.argmax(top_probability)
    max_prob = top_probability[max_index]
    label = top_classes[max_index]
    # Reading in json to match classification number to names.
    with open(arg.labels, 'r') as f:
        cat_to_name = json.load(f)
    labels = []
    for classes in top_classes:
        labels.append(cat_to_name[classes])
    return labels,top_probability


arg = command_line_args()
print("Image path to be used for prediction: {}\nNumber of top class probabilities to be listed {}".format(arg.image, arg.topk))
model, optimizer, epochs = load_checkpoint('/home/workspace/paind-project/checkpoint.pth')
labels,top_probability = predict(arg.image, model, arg.topk, arg.gpu)
print("\nTop results: ")
for p in range(arg.topk):
    print('{}: {:.4f}%'.format(labels[p], top_probability[p]*100))  