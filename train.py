import argparse
import torch
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
    parser.add_argument('--data_dir', dest = "data_dir", type = str,
                         default ='/home/workspace/aipnd-project/flowers',
                        help = 'Parent directory of images (should contain "train", "valid", and "test" folders')
    parser.add_argument('--gpu', action = 'store_true', default = True,
                         dest = 'gpu', help = 'Train with GPU')
    architectures = {'alexnet', 'vgg16_bn'}
    parser.add_argument('--arch', type = str, dest = 'arch',
                         action = 'store', choices = architectures,
                         default = 'vgg16_bn',
                         help = 'Model architecture')
    parser.add_argument('--learning_rate', type=float, default = 0.001,
                         dest = 'learning_rate', help = 'Learning rate')
    parser.add_argument('--epochs', type = int, default = 3,
                         dest = 'epochs', help = 'Number of epochs')
    parser.add_argument('--hidden_units', type = int, default = 512,
                         dest = 'hidden_units', help = 'Hidden units')
    parser.add_argument('--checkpoint', type = str, 
                         dest = 'checkpoint', help = 'Save trained model checkpoint to file')
    return parser.parse_args()

def load_data(data_dir):
    '''Transforming and loading the training, validation, and testing datasets.
    '''
    # Creating train, valid, and test directories
    train_dir, valid_dir, test_dir = data_dir+'/train', data_dir+'/valid', data_dir+'/test'
    # Defining transformations for training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    # Loading datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    # Using image datasets and transforms to define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = 64, shuffle = True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = True)
        
    return dataloaders, validloaders, testloaders, image_datasets, len(image_datasets), len(valid_datasets), len(test_datasets)
 
def load_model(arch):
    ''' Load the pretrained models depending which 
    architecture was selected in the commandline.
    '''
    if arch == 'vgg16_bn':
        model = models.vgg16(pretrained = True)
        features = list(model.classifier.children())[:-1]
        num_filters = model.classifier[0].in_features
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        features = list(model.classifier.children())[:-1]
        num_filters = model.classifier[1].in_features
    else:
        raise ValueError('Unexpected network architecture selected: ', arch)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    return model, num_filters

def build_model(model, num_filters, output_size, hidden_units):
    ''' Build the model by layer, setup the classifier and optimizer.
    '''
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_filters, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    return model, criterion, optimizer

def model_check (model, criterion, optimizer, data, data_size, gpu):
    '''Check model performance of validation or test phases.
    '''
    running_loss = 0
    running_correct = 0
    for d in data:
        inputs, labels = d
        if gpu:
            #change to CUDA
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
    loss = running_loss / data_size
    accuracy = 100* (running_correct / data_size)
    
    return loss, accuracy

def train_network(model, criterion, optimizer, epochs, dataloaders, validloaders, data_size, valid_size, gpu):
    ''' Train the network and validate after each epoch
    '''
    best_accuracy = 0
    if gpu:
        #change to CUDA
        model.to('cuda')
        print("Use GPU: "+str(gpu))
    else:
        print("Using CPU; GPU is not available/configured")
    # Loops to run through each epoch    
    for e in range(epochs):
        # Loop to run between training and validating
        for phase in ['training', 'validating']:
            # Training
            if phase == 'training':
                model.train(True)
                running_loss = 0
                print_every = 15
                steps = 0
                for ii, (inputs, labels) in enumerate(dataloaders):
                    steps += 1
                    #change to CUDA
                    if gpu:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    optimizer.zero_grad()
                    #Forward & backward passes
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # Printing epoch and loss during training.
                    if steps % print_every == 0:
                        print("Epoch: {}/{}...".format(e+1, epochs),
                            "Loss: {:.4f}".format(running_loss/print_every))
                        running_loss = 0
            # Validation 
            else:
                model.train(False)
                loss, accuracy = model_check(model, criterion, optimizer, validloaders, valid_size, gpu)
                print('\nValidation after epoch ' + str(e+1) + ':\n Loss: {:.4f} \n Accuracy: {:.4f}%\n'.format(loss, accuracy))  
    return model
  
def save_model(model, arch, image_datasets, input_size, output_size, hidden_units, epochs, learning_rate):
    ''' Save the trained model.
    '''
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {'arch': arch,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_units,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'state_dict': model.state_dict(),
                  'image_datasets': model.class_to_idx,
                  'optimizer_state': optimizer.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')
    return

args = command_line_args()
dataloaders, validloaders, testloaders, image_datasets, data_size, valid_size, test_size = load_data(args.data_dir)
model, num_filters = load_model(args.arch)
model, criterion, optimizer = build_model(model, num_filters, len(dataloaders), args.hidden_units) 
print('Pretrained architecture to train model: ' + str(args.arch) +'\n'\
      'Input size: ' + str(num_filters) +'\n'\
      'Hidden units: ' + str(args.hidden_units) +'\n'\
      'Learning rate: ' + str(args.learning_rate) +'\n'\
      'Epochs: ' + str(args.epochs) +'\n')
model = train_network(model, 
                      criterion, optimizer, 
                      args.epochs, 
                      dataloaders, validloaders,
                      data_size, valid_size,
                      args.gpu)
test_loss, test_accuracy = model_check(model, criterion, optimizer, testloaders, test_size, args.gpu)
print('Test dataset performance\n Loss: {:.4f} \n Accuracy: {:.4f}%'.format(test_loss, test_accuracy))
save_model(model, args.arch, image_datasets, num_filters, len(dataloaders), args.hidden_units, args.epochs, args.learning_rate)
print('Model saved for later use in making predictions.')