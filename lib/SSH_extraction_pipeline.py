import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from lib.functions import get_dataset_file_paths, custom_collate_fn, get_sampler
import numpy as np
import matplotlib.pyplot as plt
import binascii
from datetime import datetime
import os

from lib.custom_datasets import ByteDataset, Unlabeled_ByteDataset, PositiveClassByteDataset


class lstm_classifier(nn.Module):
    def __init__(self, input_size ,num_classes):
        super(lstm_classifier, self).__init__()
        self.hidden_size = 512
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size , num_layers=self.num_layers, batch_first=True, bidirectional = True)
        self.fc = nn.Linear(2*self.hidden_size, num_classes)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = x.unsqueeze(1) 
        output, _ = self.lstm(x)
        output = self.flatten(output)
        output = self.fc(output)
        #output = self.fc(output[:, -1, :])
        output = torch.sigmoid(output)

        return output
    
class lstm_regressor(nn.Module):
    def __init__(self, input_size,  output_size):
        super(lstm_regressor, self).__init__()
        self.hidden_size = 512
        self.num_layers = 2
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2* self.hidden_size, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)  
        extra_dimension = 1 

        x = x.view(batch_size, extra_dimension, -1)

        h0 = torch.zeros(2*self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(2*self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
    
        out = self.flatten(out)
        
        out = self.fc(out)
        
        return out
    

class SHH_extractor:
    def __init__(self, window_size, stride, cuda_index=None ):
        super(SHH_extractor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        if cuda_index is not None:
            self.device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = lstm_classifier(input_size= window_size, num_classes= 2).to(self.device)
        self.regressor = lstm_regressor(input_size= window_size, output_size= 1).to(self.device)
        self.weights = None
        self.regression_optimizer = None
        self.regression_criterion = None
        self.classification_criterion = None
        self.classification_optimizer = None
        
    def classification_train(self,training_dataset, num_epochs, batch_size,learning_rate, print_flag = False, sampling = False):
        print('classification training start')
        self.classification_criterion = nn.BCEWithLogitsLoss()
        self.classification_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        if sampling:
            training_sampler = get_sampler(training_dataset)
            train_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                collate_fn=custom_collate_fn,
                sampler= training_sampler
                )
        else:
            train_loader = DataLoader(
                training_dataset,
                batch_size=batch_size,
                collate_fn=custom_collate_fn,
                shuffle= False
                )
        losses = []
        accuracies = [] 
        f1_scores = []
 
        for epoch in range(num_epochs):
            self.classifier.train()
            epoch_loss = 0.0 
            #epoch_f1 = 0.0
            epoch_accuracy = 0
            epoch_label_class = []
            epoch_prediction_class = []

            for batch, label, position, pattern, length in train_loader:
                batch = batch.to(self.device)
                label = label.to(self.device)

                self.classification_optimizer.zero_grad()


                prediction = self.classifier(batch)

                rounded_prediction = torch.round(prediction)

                label = label.float() # convert label to float

                loss = self.classification_criterion(prediction, label)
                
                epoch_loss += loss.item() 

                label_cls = np.argmax(label.cpu().detach().numpy(), axis=1)
                pred_cls = np.argmax(rounded_prediction.cpu().detach().numpy(), axis=1)
                
                epoch_label_class.extend(label_cls)
                epoch_prediction_class.extend(pred_cls)

                loss.backward()
                self.classification_optimizer.step()
    
            epoch_f1 = f1_score(epoch_label_class, epoch_prediction_class)
            epoch_accuracy = accuracy_score(epoch_label_class, epoch_prediction_class)
            epoch_cm = confusion_matrix(epoch_label_class, epoch_prediction_class)
            epoch_precision=precision_score(epoch_label_class, epoch_prediction_class, average='macro')
            epoch_recall= recall_score(epoch_label_class, epoch_prediction_class, average='macro')
            print(epoch_cm)
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy) 
            f1_scores.append(epoch_f1)
            if print_flag:
                results = f'Epoch {epoch+1}/{num_epochs} | Epoch Accuracy: {epoch_accuracy:<23}| Epoch f1_score: {epoch_f1:<23}| Epoch precision: {epoch_precision:<23}| Epoch recall: {epoch_recall:<23}'
                print(results)
        if training_dataset.encryption_length is not None and training_dataset.version is not None:
                save_path = "saved_models/classifier_"+str(self.window_size)+"B_"+str(self.stride)+"B_"+str(training_dataset.version)+"_"+str(training_dataset.encryption_length)+".pth"
        elif training_dataset.encryption_length is None and training_dataset.version is not None:
                save_path = "saved_models/classifier_"+str(self.window_size)+"B_"+str(self.stride)+"B_"+str(training_dataset.version)+".pth"
        elif training_dataset.encryption_length is None and training_dataset.version is None:
                save_path = "saved_models/classifier_"+str(self.window_size)+"B_"+str(self.stride)+"B.pth"

        self.save_model(model_type='classifier', path=save_path)
        print("Classification training complete.")
        return
    
    def classification_test(self, validation_dataset, sampling= False):
        print('Classification test start')
        self.classifier.eval()
        true_labels = []
        predicted_labels = []
        if sampling:
            validation_sampler = get_sampler(validation_dataset)
            test_loader = DataLoader(
                validation_dataset,
                sampler= validation_sampler
                )
        else:
            test_loader = DataLoader(
                validation_dataset,
                shuffle= False
                )
        with torch.no_grad():
            for batch, label, position ,pattern, length in test_loader:
                
                batch = batch.to(self.device)
                label = label.to(self.device)
                label = label.view(-1).float()
                
                prediction = self.classifier(batch).squeeze()
                #predicted, _ = torch.max(outputs.data, 1)
                rounded_prediction = torch.round(prediction)


                true_labels.append(label.cpu().numpy())
                predicted_labels.append(rounded_prediction.cpu().numpy())
            
            true_cls = np.argmax(true_labels, axis=1)
            pred_cls = np.argmax(predicted_labels, axis=1)
            test_acc = accuracy_score(true_cls, pred_cls)
            print(f"Test accuracy: {test_acc:.4f}")

            cm = confusion_matrix(true_cls, pred_cls)
            print(f'Test Confusion matrix: \n{cm}\n')
            precision = precision_score(true_cls, pred_cls)
            recall = recall_score(true_cls, pred_cls)
            f1 = f1_score(true_cls, pred_cls)

            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 score: {f1:.4f}") 
            print('Test Complete') 

    def regression_train(self, training_dataset, num_epochs, batch_size,learning_rate ):
        print('regression training start')     
        self.regression_criterion = nn.L1Loss()
        self.regression_optimizer = torch.optim.Adam(self.regressor.parameters(), lr=learning_rate) 
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False) 
        losses = []
        predictions =  []
        true_labels = []
        for epoch in range(num_epochs):
            self.regressor.train()
            epoch_loss = 0.0 
            epoch_mse= 0.0
            epoch_rmse= 0.0
            epoch_r2= 0.0
            for batch, _, label, pattern, length in train_loader:
                
                batch = batch.to(self.device)
                label = label.to(self.device)
                label = label.view(-1,1)

                self.regression_optimizer.zero_grad()

                prediction = self.regressor(batch)
                label = label.float() # convert label to float
                

                loss = self.regression_criterion(prediction, label)
                loss.backward()
                self.regression_optimizer.step()
                epoch_loss += loss.item() 
            avg_epoch_loss = epoch_loss/len(train_loader)
            losses.append(avg_epoch_loss)
            results = f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}"
            print(results)
        
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_idx= 'version_'+str(training_dataset.version)+'-length_'+str(training_dataset.encryption_length)+'-window_'+str(self.window_size)+'-stride_'+str(self.stride)+'-lr_'+str(learning_rate)+'-epoch_'+str(num_epochs)+'-batch_'+str(batch_size)
        filename = f"img/training_loss_{name_idx}_timestamp_{timestamp}.png"
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Epoch L1Loss')
        plt.title('Training Loss')
        fig1 = plt.gcf()
        plt.show()
    
        fig1.savefig(filename, dpi=100)
        if training_dataset.encryption_length is not None and training_dataset.version is not None:
                save_path = "saved_models/regressor_"+str(self.window_size)+"B_"+str(self.stride)+"B_"+str(training_dataset.version)+"_"+str(training_dataset.encryption_length)+".pth"
        elif training_dataset.encryption_length is None and training_dataset.version is not None:
                save_path = "saved_models/regressor_"+str(self.window_size)+"B_"+str(self.stride)+"B_"+str(training_dataset.version)+".pth"
        elif training_dataset.encryption_length is None and training_dataset.version is None:
                save_path = "saved_models/regressor_"+str(self.window_size)+"B_"+str(self.stride)+"B.pth"

        self.save_model(model_type='regressor', path=save_path)
        print("Regression training complete.")

    def regression_test(self, validation_test):
        print('Regression test start')
        test_loader = DataLoader(validation_test, shuffle=False)
        self.regressor.eval()
        predictions =  []
        true_labels = []

        with torch.no_grad():
            for batch, _, label, pattern, length in test_loader:
                batch = batch.to(self.device)
                label = label.to(self.device) 
                label = label.view(-1, 1)
                prediction = self.regressor(batch)
                prediction = (prediction // 8) * 8 + (8 if prediction % 8 >= 4 else 0)
                predictions.extend(prediction.cpu().numpy())
                true_labels.extend(label.cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate evaluation metrics
        mse = mean_squared_error(true_labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)

        print("Regression testing results:")
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("Mean Absolute Error (MAE):", mae)
        print("R-squared (R2) Score:", r2)
        print('Test Complete')

    def classification_regression_pipeline(self, dataset, label_flag=True):
        if label_flag: 
            loader = DataLoader(dataset, batch_size=1, shuffle= False)
            with torch.no_grad():
                index= -1
                true_classification_labels = []
                predicted_classification_labels = []
                true_regression_positions= []
                predicted_regressionç_positions= []
                for batch, classification_label, position, pattern, length in loader:
                    index= index+1
                    batch = batch.to(self.device)

                    classification_result = self.classifier(batch)
                    classification_label = classification_label.to(self.device).view(-1).float()

                    true_class= np.argmax(classification_label.cpu().numpy())
                    predicted_class= np.argmax(classification_result.cpu().numpy())

                    true_classification_labels.append(true_class)
                    predicted_classification_labels.append(predicted_class)

                    if predicted_class == 1 and true_class == 1:
                        if position > self.window_size-self.stride:
                            #print('Correct classification, Key is duplicated by stride in the next data window')
                            pass
                        length = int(length[0])
                        data = batch.squeeze().tolist()

                        prediction = self.regressor(batch)
                        prediction = int(torch.round(prediction).item()) + 8
                        prediction = (prediction // 8) * 8 + (8 if prediction % 8 >= 4 else 0)

                        predicted_key = data[prediction: prediction+length]
                        predicted_key = [int(x) for x in predicted_key]
                        predicted_bytearrray= bytearray(predicted_key)
                        predicted_hex_string = binascii.hexlify(predicted_bytearrray).decode('utf-8')

                        actual_position = int(position.item() ) + 8
                        actual_key = data[actual_position:actual_position+length]
                        actual_key = [int(x) for x in actual_key]
                        actual_bytearrray= bytearray(actual_key)
                        actual_hex_string = binascii.hexlify(actual_bytearrray).decode('utf-8')

                        true_regression_positions.append(actual_position)
                        predicted_regressionç_positions.append(position)

                        print(f"Data segment #{index} contains a key in position: {prediction}, The true position is {actual_position}")
                        print(f"The predicted key is: {predicted_key},\n in hexadecimal format: {predicted_hex_string.upper()}")
                        print(f"The actual key is: {actual_key},\n in hexadecimal format: {actual_hex_string.upper()}")
                        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    
                    elif predicted_class != true_class:
                        #print('Incorrect Classification, the prediction is skipped for this data window')
                        pass
                    else:
                        pass
            classification_cm= confusion_matrix(true_classification_labels, predicted_classification_labels)
            regressor_mse = mean_squared_error(true_classification_labels, predicted_classification_labels)
            regressor_rmse = np.sqrt(regressor_mse)
            regressor_mae = mean_absolute_error(true_classification_labels, predicted_classification_labels)
            regressor_r2 = r2_score(true_classification_labels, predicted_classification_labels)

            print("classification_confusion_matrix:",classification_cm)
            print("Mean Squared Error (MSE):", regressor_mse)
            print("Root Mean Squared Error (RMSE):", regressor_rmse)
            print("Mean Absolute Error (MAE):", regressor_mae)
            print("R-squared (R2) Score:", regressor_r2)
            print('Complete')


        else:
            loader = DataLoader(dataset, batch_size=1, shuffle= False)
            with torch.no_grad():
                index= -1
                for batch in loader:
                    index = index +1
                    batch = batch.to(self.device)
                    classification_result = self.classifier(batch)
                    predicted_class= np.argmax(classification_result.cpu().numpy())
                    if predicted_class == 1:
                        data = batch.squeeze().tolist()
                        prediction = self.regressor(batch)
                        prediction = int(torch.round(prediction).item()) + 8
                        prediction = (prediction // 8) * 8 + (8 if prediction % 8 >= 4 else 0)
                        if prediction > self.window_size - self.stride:
                            continue
                        predicted_key = data[prediction: prediction+ 64]
                        predicted_key = [int(x) for x in predicted_key]
                        predicted_bytearrray= bytearray(predicted_key)
                        predicted_hex_string = binascii.hexlify(predicted_bytearrray).decode('utf-8')
                        print(f"Data segment #{index} contains a key in position: {prediction}")
                        print(f"The predicted key is in: {predicted_key},\n in hexadecimal format: {predicted_hex_string.upper()}")
                        print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")

    def save_model(self,model_type ,path):
        if not os.path.exists('./saved_models/'):
            os.mkdir('./saved_models/')
            print(f"Folder '{'./saved_models/'}' created successfully.")
        if model_type == "regressor":
            torch.save(self.regressor.state_dict(), path)
            print(f"regression unit saved in {path}")
        elif model_type == "classifier":
            torch.save(self.classifier.state_dict(), path)
            print(f"classification unit saved in {path}")
        else:
            raise Exception('model type is not regressor or classifier')
        
    def load_model(self, model_type, path):
        if model_type == "regressor":
            self.regressor.load_state_dict(torch.load(path))
            print(f'Loading regression unit from {path} ')
        elif model_type == "classifier":
            self.classifier.load_state_dict(torch.load(path))
            print(f'Loading classification unit from {path} ')
        else:
            raise Exception('model type is not regressor or classifier')

    def predict(self, dataset):
        loader = DataLoader(dataset, batch_size= 1,shuffle=False)
        with torch.no_grad():
            predicted_keys= []
            predicted_positions = []
            actual_keys = []
            actual_positions = []
            hex_predicted= []
            hex_actual = []
            lengths = []
            for batch, classification_label, position, pattern, length in loader:
                if position > self.window_size-self.stride:
                    continue
                length = int(length[0])
                data = batch.squeeze().tolist() 
                batch = batch.to(self.device)
                prediction = self.regressor(batch)
                prediction = int(torch.round(prediction).item()) + 8
                prediction = (prediction // 8) * 8 + (8 if prediction % 8 >= 4 else 0)

                predicted_key = data[prediction: prediction+length]
                predicted_key = [int(x) for x in predicted_key]
                predicted_bytearrray= bytearray(predicted_key)
                predicted_hex_string = binascii.hexlify(predicted_bytearrray).decode('utf-8')

                actual_position = int(position.item() ) + 8 
                actual_key = data[actual_position:actual_position+length]
                actual_key = [int(x) for x in actual_key]
                actual_bytearrray= bytearray(actual_key)
                actual_hex_string = binascii.hexlify(actual_bytearrray).decode('utf-8')

                predicted_keys.append(predicted_key)
                predicted_positions.append(prediction)
                actual_keys.append(actual_key)
                actual_positions.append(actual_position)
                lengths.append(length)
                hex_predicted.append(predicted_hex_string)
                hex_actual.append(actual_hex_string)
            counter = 0
            for idx in range(len(dataset.files)) :
                print(f"File {dataset.files[idx]} contains {dataset.nbr_keys_per_file[idx]} keys as follows")
                print("\n")
                for pos in range(dataset.nbr_keys_per_file[idx]):
                    print(f"Key#{pos} has a length of {lengths[counter]}")
                    print(f"The predicted position is {predicted_positions[counter]} which gives the following key : {predicted_keys[counter]}")
                    print(f"having a Hex value of : {hex_predicted[counter]}")
                    
                    print(f"The actual position is {actual_positions[counter]} which gives the following key : {actual_keys[counter]}")
                    print(f"having a Hex value of : {hex_actual[counter]}")
                    
                    counter = counter + 1
                    print("\n")
                print("---------------------------------------------------------------------------------------------------------------------------------")

    def sequential_train(self, dataset, parameter_space, unit, classification_sampling= False):
        if (unit != 'regressor') and (unit != 'classifier'):
            raise ValueError('unit must me regressor or classifier')
        index = 0
        print('sequential training init')
        for parameters in parameter_space:
            index = index +1
            print(f"__Training round #{index}")
            num_epochs = parameters[0]
            batch_size = parameters[1]
            learning_rate = parameters[2]
            if unit == 'classifier':
                self.classification_train(dataset, num_epochs= num_epochs, batch_size= batch_size,learning_rate=learning_rate,print_flag= True,sampling= classification_sampling)
            elif unit == 'regressor':
                self.regression_train(dataset, num_epochs= num_epochs, batch_size= batch_size, learning_rate= learning_rate)

        print('Sequential training done.')
             
    def load_units(self,window_size, stride, data_version, length):
        if length is not None and data_version is not None:
            classifier_path = f'saved_models/classifier_{window_size}B_{stride}B_{data_version}_{length}.pth'
            regressor_path = f'saved_models/regressor_{window_size}B_{stride}B_{data_version}_{length}.pth'
        elif length is None and data_version is not None:
            classifier_path = f'saved_models/classifier_{window_size}B_{stride}B_{data_version}.pth'
            regressor_path = f'saved_models/regressor_{window_size}B_{stride}B_{data_version}.pth'
        elif length is None and data_version is None:
            classifier_path = f'saved_models/classifier_{window_size}B_{stride}B.pth'
            regressor_path = f'saved_models/regressor_{window_size}B_{stride}B.pth'
        
        try:
            self.load_model(model_type= 'classifier', path= classifier_path)
        except:
            raise ValueError("There is no saved classification unit for the provided data configuration")
        try:
            self.load_model(model_type= 'regressor', path= regressor_path)  
        except:
            raise ValueError("There is no saved regression unit for the provided data configuration")
                
def get_labeled_datasets(path, window_size, stride, version= None, length= None):
    if length is not None and version is not None:
        verified_path = path + '/' +version+'/'+length
    elif length is None and version is not None:
        verified_path = path + '/' +version
    elif length is None and version is None:
        verified_path = path
    file_paths = get_dataset_file_paths(verified_path)
    print(f'Loading files in : {verified_path}')
    classification_set = ByteDataset(path = file_paths, window_size= window_size, stride= stride, enc_length= length, version=version)
    regression_set = PositiveClassByteDataset(path = file_paths, window_size= window_size, stride= stride, enc_length= length, version= version)
    return classification_set, regression_set

def get_Unlabeled_dataset(path, window_size, stride, version, length):
    if length is not None and version is not None:
        verified_path = path + '/' +version+'/'+length
    elif length is None and version is not None:
        verified_path = path + '/' +version
    elif length is None and version is None:
        verified_path = path
    file_paths = get_dataset_file_paths(verified_path)
    print(f'Loading files in : {verified_path}')
    unlabeled_set = Unlabeled_ByteDataset(path = file_paths, window_size= window_size, stride= stride, enc_length= length, version= version)
    return unlabeled_set
