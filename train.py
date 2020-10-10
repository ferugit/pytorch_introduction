import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch

def right_predictions(out, label, threshold=0.5):
    """
    From a given set of labes and predictions 
    it uses a threshols to determine if the prediction
    is right
    """
    counter = 0
    if(out.shape[1] > 1):
        predictions_index = out.data.cpu().max(1, keepdim=True)[1]
        #label_index = label.data.cpu().max(1, keepdim=True)[1]
        counter = predictions_index.eq(label.view_as(predictions_index)).sum().item()
    else:
        for i in range(out.shape[0]):
            prediction = 0.0
            if (out[i] >= threshold):
                prediction = 1.0
            
            if (label[i] == prediction):
                counter += 1
    return counter

def train(model, train_loader, validation_loader, optimizer, criterion,
 device, epochs, batch_size, lr, net_class='rnn', one_hot=False):
    """
    Trainer for wuw detection models
    """
    # Metrics
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_times = []

    # Print model information
    print(model)

    # Get trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ' + str(trainable_params))

    print('Starting trainig...')

    # For present intermediate information
    n_intermediate_steps = int(len(train_loader)/3)
    
    # Start training loop
    for epoch in range(1, epochs+1):
        start_time = time.process_time()
        
        # Train model
        train_loss = 0.0
        train_accuracy = 0.0
        counter = 0

        model.train()
        for x, labels in train_loader:
            counter += 1
            model.zero_grad()
            
            # Model forward
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())

            # Backward and optimization
            if(one_hot):
                loss = criterion(out, labels)
            else:
                loss = criterion(out, labels.to(device).float())
            loss.backward()
            optimizer.step()

            # Store metrics
            if(one_hot):
                train_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), labels)
            else:
                train_accuracy += right_predictions(out, labels)
            train_loss += loss.item()

            # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Average Loss, Accuracy for Step: {}, {}".format(
                    epoch,
                    counter,
                    len(train_loader),
                    round(train_loss/counter, 4),
                    round(train_accuracy/(counter * batch_size), 4)
                    ))

        # Validate model
        validation_loss = 0.0
        validation_accuracy = 0.0

        model.eval()
        for x, labels in validation_loader:

            # Model forward
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())
            
            # Store metrics: loss and accuracy
            if(one_hot):
                loss = criterion(out, labels)
                validation_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), labels)
            else:
                loss = criterion(out, labels.to(device).float())
                validation_accuracy += right_predictions(out, labels)
            validation_loss += loss.item()
        
        # Calculate average losses
        train_loss = train_loss/len(train_loader)
        validation_loss = validation_loss/len(validation_loader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        # Calculate accuracies
        train_accuracy = train_accuracy/len(train_loader.sampler)
        validation_accuracy = validation_accuracy/len(validation_loader.sampler)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        
        # Print epoch information
        current_time = time.process_time()
        print("")
        print("Epoch {}/{} Done.".format(epoch, epochs))
        print("\t Tain Loss: {}  Validation Loss: {}".format(train_loss, validation_loss))
        print("\t Train accuracy: {}    Validation accuracy: {}".format(train_accuracy, validation_accuracy))
        print("\t Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
        print("")
        epoch_times.append(current_time-start_time)
        
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    metrics = (train_losses, validation_losses, train_accuracies, validation_accuracies)

    return model, metrics

def plot_train_metrics(metrics):
    """
    Plot loss and accuracy metrics generated 
    on a training
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(metrics[0], label='training loss')
    plt.plot(metrics[1], label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.subplot(122)
    plt.plot(metrics[2], label='training accuracy')
    plt.plot(metrics[3], label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()


def test_model(model, test_loader, criterion, device, batch_size, net_class='rnn', one_hot=False):
    
    # Metrics initialization
    test_loss = 0.0
    test_accuracy = 0.0

    labels = []
    predictions = []

    with torch.no_grad():

        model.eval()
        
        for x, label in test_loader:

            # Model forward
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())

            # Store metrics: loss and accuracy
            if(one_hot):
                loss = criterion(out, label)
                test_accuracy += right_predictions(torch.nn.functional.softmax(out, 1), label)
            else:
                loss = criterion(out, label.to(device).float())
                test_accuracy += right_predictions(out, label)
            test_loss += loss.item()

            # Store labels and predictions
            labels += label.squeeze().tolist()
            if (one_hot):
                predictions += torch.nn.functional.softmax(out.squeeze(), 1).tolist()
            else:
                predictions += out.squeeze().tolist()

        # Calculate average losses
        test_loss = test_loss/len(test_loader)

        # Calculate accuracies
        test_accuracy = test_accuracy/len(test_loader.sampler)

        metrics = (test_loss, test_accuracy)

    return labels, predictions, metrics


def set_seed(seed):
    """
    Fix seed of torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)