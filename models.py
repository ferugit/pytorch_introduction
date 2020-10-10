import torch
import torch.nn as nn
import json

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename, metrics, lr,
     epochs, batch_size, criterion, optimizer):

        # Store model data
        model_data = {
            filename:{
                'hyperparameters':{
                    'learning_rate':lr,
                    'epochs':epochs,
                    'batch_size':batch_size
                },
                'metrics':{
                    'training':{
                        'loss':metrics[0][-1],
                        'accuracy':metrics[2][-1]
                    },
                    'validation':{
                        'loss':metrics[1][-1],
                        'accuracy':metrics[3][-1]
                    }
                },
                'network_architecture' : str(type(self)),
                'optimizer':str(type(optimizer)),
                'criterion':str(type(criterion))
            }
        }
        with open(filename + ".json", "w") as write_file:
            json.dump(model_data, write_file)

        # Store model
        torch.save(self.state_dict(), filename +'.pt')

    def save_scripted(self, filename):
        scripted_module = torch.jit.script(self)
        scripted_module.save(filename + '.zip')

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class LeNet(SerializableModule):
    """
    LeNet network to classify handwritten characters
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    