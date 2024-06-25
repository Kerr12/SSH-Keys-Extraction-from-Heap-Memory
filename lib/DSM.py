from torch.utils.data import  DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from itertools import chain
import binascii
import uuid
import json
import subprocess
import pandas as pd

class EmbeddingFunction(nn.Module):
    def __init__(self, embedding_input_dim, hidden_dim):
        super(EmbeddingFunction, self).__init__()

        self.lstm_layer1 = nn.LSTM(embedding_input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_layer2 = nn.LSTM(8, hidden_dim, num_layers=1, batch_first=True)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(2 * hidden_dim, embedding_input_dim)

    def forward(self, input_tensor, input_embedding):
        input_embedding = input_embedding.unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        lstm_out1, _ = self.lstm_layer1(input_embedding)
        lstm_out2, _ = self.lstm_layer2(input_tensor)
        
        # Concatenate the last hidden states from both LSTM layers
        combined_output = torch.cat((lstm_out1[:, -1, :], lstm_out2[:, -1, :]), dim=1)
        final_embedding = self.output_layer(combined_output).squeeze()
        activated_output = self.activation(final_embedding)

        return activated_output

    
class Agent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, zero_probs=[]):
        super(Agent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, output_size)
        self.flatten = nn.Flatten()
        self.zero_probs = zero_probs

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        out, _ = self.lstm(x)
        out = self.flatten(out)
        out = self.fc(out)
        
        for class_idx in self.zero_probs:
            out[:, class_idx] = float('-inf')
        
        output = F.softmax(out, dim=1)
        return output.squeeze(0)
    
class Generator(nn.Module):
    def __init__(self, embedding_input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()

        self.lstm_layer1 = nn.LSTM(embedding_input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.lstm_layer2 = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=1, batch_first=True)  # Reverse of the encoder
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim//2,output_dim)  # Output should match the input dimension

    def forward(self, input_embedding):
        input_embedding = input_embedding.unsqueeze(0).unsqueeze(0)
        lstm_out1, _ = self.lstm_layer1(input_embedding)

        lstm_out2, _ = self.lstm_layer2(lstm_out1)

        # combined_output = torch.cat((lstm_out1[:, -1, :], lstm_out2[:, -1, :]), dim=1)
        final_output = self.output_layer(lstm_out2).squeeze().squeeze()
        activated_output = self.activation(final_output)

        return activated_output

class DSM:
    def __init__(self, embedding_dim, agent_hidden_state, cuda_index=None):
        self.cursor = 0
        self.gen_cursor= 0
        self.uuid_counter = 0
        if cuda_index is not None:
            self.device = torch.device(f'cuda:{cuda_index}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_state = agent_hidden_state
        self.embedding_size= embedding_dim
        
        self.generator = Generator(embedding_input_dim= self.embedding_size, hidden_dim=2* self.embedding_size, output_dim=8).to(self.device)
        self.embedding_function = EmbeddingFunction(embedding_input_dim=self.embedding_size, hidden_dim=2* self.embedding_size).to(self.device)

        self.agent_0 = Agent(input_size=self.embedding_size , hidden_size= self.hidden_state, output_size= 4, zero_probs=[2,3]).to(self.device)
        self.agent_1 = Agent(input_size=self.embedding_size , hidden_size= self.hidden_state, output_size= 4, zero_probs=[0,3]).to(self.device)
        self.agent_2 = Agent(input_size=self.embedding_size , hidden_size= self.hidden_state, output_size= 4, zero_probs=[0]).to(self.device)
        self.agent_3 = Agent(input_size=self.embedding_size , hidden_size= self.hidden_state, output_size= 4, zero_probs=[2,3]).to(self.device)

        # Create a list of agents
        self.agents = [self.agent_0, self.agent_1, self.agent_2, self.agent_3]

        self.parameters = (
                            list(self.embedding_function.parameters()) +
                            list(self.agent_0.parameters()) +
                            list(self.agent_1.parameters()) +
                            list(self.agent_2.parameters()) +
                            list(self.agent_3.parameters()) 
                        )

        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.generator_criterion = nn.L1Loss()
        self.EOF = False
        self.embedding = None
        self.action = None
        self.current_state= 0
        self.previous_state = 0
               
    def train(self, dataset, lr, shuffle= True, patience= None, suppress_output=False, correction = False):
        print('Training start')
        self.embedding_function.train()
        self.agent_0.train()
        self.agent_1.train()
        self.agent_2.train() 
        self.agent_3.train()
        
        # Initialize the optimizer and learning rate scheduler
        self.optimizer= optim.Adam(self.parameters, lr=lr)
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)

        # Create a DataLoader for the files
        loader = DataLoader(dataset, batch_size= 1, shuffle= shuffle)

        losses=[]
        file_index= 0
        data_predictions =  []

        for file_data, file_labels in loader:
            if not suppress_output:
                print(f"file #{file_index}")

            # Read File and it's labels
            file_data = torch.tensor(file_data, dtype=torch.float32).to(self.device)
            file_labels = torch.tensor(file_labels, dtype=torch.long).to(self.device)

            # File initisalisations
            running_loss = 0
            averaged_running_loss = 0
            self.embedding = torch.rand(self.embedding_size).to(self.device)
            self.cursor = 0
            prediction_count = {0:0, 1:0, 2:0, 3:0}
            label_count = {0:0, 1:0, 2:0, 3:0}
            file_predictions= []
            self.current_state = 0
            self.previous_state = 0
            key_len_counter = 0
            self.optimizer.zero_grad()
            for _ in tqdm(range(len(file_data))):
                
                #Data and Labels
                current_segment = file_data[self.cursor]
                current_label = file_labels[self.cursor]
                label = current_label.item() 
                

                #Generating the memory embedding
                self.embedding = self.embedding_function(current_segment, self.embedding).to(self.device)

                self.action = self.agents[self.current_state](self.embedding)
                self.previous_state = self.current_state
                self.current_state = torch.argmax(self.action.cpu()).item()

                if correction:
                    if self.current_state == 2: 
                        key_len_counter += 1 
                    elif self.current_state == 3:
                        if key_len_counter<2:
                            file_predictions[-2:] = [0, 0]
                            prediction_count[1] -= 1
                            prediction_count[2] -= 1
                            prediction_count[0] += 2
                            self.current_state = 0
                        elif key_len_counter not in [2,3,4,8] :
                            file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)
                            prediction_count[0] += (key_len_counter+1)
                            prediction_count[1] -= 1
                            prediction_count[2] -= key_len_counter
                            self.current_state = 0
                        key_len_counter = 0
                    elif self.current_state ==1:
                        if self.previous_state ==2:
                            file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)  
                            prediction_count[0] += (key_len_counter+1)                 
                            prediction_count[1] -= 1
                            prediction_count[2] -= key_len_counter
                            key_len_counter = 0
                    if (self.current_state == 0 or self.current_state == 1 ) and  self.previous_state == 1:
                        file_predictions[-1] = 0 
                        prediction_count[1] -= 1
                        prediction_count[0] += 1


                #saving the prediction and counts  
                file_predictions.append(self.current_state)  
                prediction_count[self.current_state] += 1  
                label_count[label] += 1  

                #Loss calculation and backpropagation through the embedding function and the agents
                loss = self.criterion(input=self.action.unsqueeze(0), target=current_label.unsqueeze(0))
                running_loss +=  loss
                
                self.cursor = self.cursor +  1

            data_predictions.append(file_predictions)
            averaged_running_loss= running_loss/(len(file_data))
            averaged_running_loss.backward()
            self.optimizer.step()
            if not suppress_output:
                print(f"File total loss = {running_loss}  | Average file loss = {running_loss /len(file_data)}")
                print(f"labels distributions: {label_count}")
                print(f"predictions distributions: {prediction_count}")
            file_index += 1
            losses.append(running_loss.detach().cpu()/len(file_data))
            if patience:
                scheduler.step(running_loss)
                if scheduler.num_bad_epochs >= scheduler.patience:
                    print("Early stopping!")
                    return data_predictions
        print('Training complete')    
        return data_predictions
         
    def infer(self, dataset, labeled_dataset= True, suppress_output=False, evaluation_approach='total', correction= True, early_stop_index= None):
        print('Inference start')
        self.embedding_function.eval()
        self.agent_0.eval()
        self.agent_1.eval()
        self.agent_2.eval() 
        self.agent_3.eval()

        loader = DataLoader(dataset, batch_size= 1, shuffle= False)
        
        file_index= 0
        total_predictions= []
        total_labels = []

        if labeled_dataset:
            for file_data, file_labels in loader:
                if not suppress_output:
                    print(f"file #{file_index}")
                file_data = torch.tensor(file_data, dtype=torch.float32).to(self.device)
                file_labels = torch.tensor(file_labels, dtype=torch.long).to(self.device)

                prediction_count = {0:0, 1:0, 2:0, 3:0}
                label_count = {0:0, 1:0, 2:0, 3:0}
                file_predictions= []
                labels = []
                self.current_state= 0
                self.previous_state = 0
                self.embedding = torch.rand(self.embedding_size).to(self.device)
                self.cursor = 0
                key_len_counter = 0
                if early_stop_index is not None and file_index == early_stop_index:
                    return total_predictions
                for indx in tqdm(range(len(file_data)), disable=suppress_output):
                    current_segment = file_data[self.cursor]
                    current_label = file_labels[self.cursor]
                    label = current_label.item() 
                    labels.append(label)

                    self.embedding = self.embedding_function(current_segment, self.embedding).to(self.device)
                    self.action = self.agents[self.current_state](self.embedding)
                    self.previous_state = self.current_state
                    self.current_state = torch.argmax(self.action.cpu()).item()
                    
                        #correction mechanism
                    if correction:
                        if self.current_state == 2: 
                            key_len_counter += 1 
                        elif self.current_state == 3:
                            if key_len_counter<2:
                                file_predictions[-2:] = [0, 0]
                                prediction_count[1] -= 1
                                prediction_count[2] -= 1
                                prediction_count[0] += 2
                                self.current_state = 0
                            elif key_len_counter not in [2,3,4,8] :
                                file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)
                                prediction_count[0] += (key_len_counter+1)
                                prediction_count[1] -= 1
                                prediction_count[2] -= key_len_counter
                                self.current_state = 0
                            key_len_counter = 0
                        elif self.current_state ==1:
                            if self.previous_state ==2:
                                file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)  
                                prediction_count[0] += (key_len_counter+1)                 
                                prediction_count[1] -= 1
                                prediction_count[2] -= key_len_counter
                                key_len_counter = 0
                        if (self.current_state == 0 or self.current_state == 1 ) and  self.previous_state == 1:
                            file_predictions[-1] = 0 
                            prediction_count[1] -= 1
                            prediction_count[0] += 1



                    file_predictions.append(self.current_state) 
                    prediction_count[self.current_state] += 1 
                    label_count[label] += 1         
                    self.cursor = self.cursor +  1

                total_predictions.append(file_predictions)
                total_labels.append(labels)
                if not suppress_output:
                    print(f"labels distributions: {label_count}")
                    print(f"predictions distributions: {prediction_count}")
                    if evaluation_approach == "per_file":
                        self.evaluate(predictions= file_predictions, labels= labels, evaluation_approach=evaluation_approach )
                file_index += 1
            if not suppress_output:
                self.evaluate(predictions= total_predictions, labels= total_labels, evaluation_approach='total' )
            print('Inference Complete')
            return total_predictions
        else:
            for file_data in loader:
                if not suppress_output:
                    print(f"file #{file_index}")
                file_data = torch.tensor(file_data, dtype=torch.float32).to(self.device)

                prediction_count = {0:0, 1:0, 2:0, 3:0}
                file_predictions= []
                self.current_state= 0
                self.embedding = torch.rand(self.embedding_size).to(self.device)
                self.cursor = 0
                key_len_counter = 0
                if early_stop_index is not None and file_index == early_stop_index:
                    return total_predictions
                for indx in tqdm(range(len(file_data)), disable=suppress_output):
                    current_segment = file_data[self.cursor]
                    self.embedding = self.embedding_function(current_segment, self.embedding).to(self.device)
                    self.action = self.agents[self.current_state](self.embedding)
                    self.previous_state = self.current_state
                    self.current_state = torch.argmax(self.action.cpu()).item()

                    if correction:
                    # #correction mechanism
                        if self.current_state == 2: 
                            key_len_counter += 1 
                        elif self.current_state == 3:
                            if key_len_counter<2:
                                file_predictions[-2:] = [0, 0]
                                prediction_count[1] -= 1
                                prediction_count[2] -= 1
                                prediction_count[0] += 2
                                self.current_state = 0
                            elif key_len_counter not in [2,3,4,8] :
                                file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)
                                prediction_count[0] += (key_len_counter+1)
                                prediction_count[1] -= 1
                                prediction_count[2] -= key_len_counter
                                self.current_state = 0
                            key_len_counter = 0
                        elif self.current_state ==1:
                            if self.previous_state ==2:
                                file_predictions[-(key_len_counter+1):] = [0] * (key_len_counter+1)  
                                prediction_count[0] += (key_len_counter+1)                 
                                prediction_count[1] -= 1
                                prediction_count[2] -= key_len_counter
                                key_len_counter = 0
                        if (self.current_state == 0 or self.current_state == 1 ) and  self.previous_state == 1:
                            file_predictions[-1] = 0 
                            prediction_count[1] -= 1
                            prediction_count[0] += 1

                    file_predictions.append(self.current_state) 
                    prediction_count[self.current_state] += 1      
                    self.cursor = self.cursor +  1

                total_predictions.append(file_predictions)
                if not suppress_output:
                    print(f"predictions distributions: {prediction_count}")
                file_index += 1
            print('Inference Complete')
            return total_predictions
    
    def evaluate(self, predictions, labels, evaluation_approach='total'):
        if evaluation_approach not in ['total', 'per_file']:
            raise ValueError("Invalid approach. Use 'total' or 'per_file'.")

        if evaluation_approach == 'per_file':          
            confusion_mat = confusion_matrix(labels, predictions)
            precision = precision_score(labels, predictions, average=None)
            recall = recall_score(labels, predictions, average=None)
            f1 = f1_score(labels, predictions, average=None)

            print("Class-wise Precision:", precision)
            print("Class-wise Recall:", recall)
            print("Class-wise F1-score:", f1)
            print(confusion_mat)
            print("-----------------------------------------------")

        elif evaluation_approach == 'total':
            flattened_predictions = list(chain(*predictions))
            flattened_labels = list(chain(*labels))
            
            confusion_mat = confusion_matrix(flattened_labels, flattened_predictions)
            precision = precision_score(flattened_labels, flattened_predictions, average=None)
            recall = recall_score(flattened_labels, flattened_predictions, average=None)
            f1 = f1_score(flattened_labels, flattened_predictions, average=None)
            print('===============================================')
            print("Overall Metrics (Summed):")
            print("Class-wise Precision:", precision)
            print("Class-wise Recall:", recall)
            print("Class-wise F1-score:", f1)
            print(confusion_mat)
       
    def train_generator(self, dataset, lr, suppress_output=False, shuffle= True):
        print('Generator training start')
        self.generator.train()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr= lr)
        loader = DataLoader(dataset, batch_size= 1, shuffle= shuffle)
        file_index= 0
        generated_data = []
        predicted_states = []
        for file_data, file_labels in loader:
            if not suppress_output:
                print(f"file #{file_index}")
            # Read File and it's labels
            file_data = torch.tensor(file_data, dtype=torch.float32).to(self.device)
            file_labels = torch.tensor(file_labels, dtype=torch.long).to(self.device)
            self.current_state= 0
            self.embedding = torch.rand(self.embedding_size).to(self.device)
            self.cursor = 0
            prediction_count = {0:0, 1:0, 2:0, 3:0}
            file_generated_data = []
            file_predicted_states = []
            running_loss = 0
            key_len_counter = 0
            self.generator_optimizer.zero_grad()
            for _ in tqdm(range(len(file_data))):
                current_label = file_labels[self.cursor]
                current_segment = file_data[self.cursor]
                generated_segment = self.generator(self.embedding).to(self.device)
                generated_segment = generated_segment.round().type(torch.FloatTensor).to(self.device)
                file_generated_data.append(generated_segment.tolist())

                self.embedding = self.embedding_function(current_segment, self.embedding).to(self.device)
                self.action = self.agents[self.current_state](self.embedding)
                self.previous_state = self.current_state
                self.current_state = torch.argmax(self.action.cpu()).item()

                # #correction mechanism
                if self.current_state == 2: 
                    key_len_counter += 1 
                elif self.current_state == 3:
                    if key_len_counter<2:
                        file_predicted_states[-2:] = [0, 0]
                        prediction_count[1] -= 1
                        prediction_count[2] -= 1
                        prediction_count[0] += 2
                        self.current_state = 0
                    elif key_len_counter not in [2,3,4,8] :
                        file_predicted_states[-(key_len_counter+1):] = [0] * (key_len_counter+1)
                        prediction_count[0] += (key_len_counter+1)
                        prediction_count[1] -= 1
                        prediction_count[2] -= key_len_counter
                        self.current_state = 0
                    key_len_counter = 0
                elif self.current_state ==1:
                    if self.previous_state ==2:
                        file_predicted_states[-(key_len_counter+1):] = [0] * (key_len_counter+1)  
                        prediction_count[0] += (key_len_counter+1)                 
                        prediction_count[1] -= 1
                        prediction_count[2] -= key_len_counter
                        key_len_counter = 0
                if (self.current_state == 0 or self.current_state == 1 ) and  self.previous_state == 1:
                    file_predicted_states[-1] = 0 
                    prediction_count[1] -= 1
                    prediction_count[0] += 1

                file_predicted_states.append(self.current_state)
                prediction_count[self.current_state] += 1  
                print(f"generated  {generated_segment}")
                print(f"input {current_segment}")
                print(f"predicted state {self.current_state}")
                print(f"-----------------------------------------------------------------------------")
                loss = self.generator_criterion(input=generated_segment, target=current_segment)
                running_loss +=  loss
                self.cursor = self.cursor +  1
            running_loss.backward()
            self.generator_optimizer.step()
            if not suppress_output:
                print(f"File total loss = {running_loss}  | Average file loss = {running_loss /len(file_data)}")
                print(f"predictions distributions: {prediction_count}")
            file_index += 1

            generated_data.append(file_generated_data)
            predicted_states.append(file_predicted_states)
        print('Training complete')
        return generated_data, predicted_states

    def generate_file(self, path, file_length, last_file_index, supress_output= False, dump= False, embedding= None):
        self.generator.eval()
        self.embedding_function.eval()
        prediction_count = {0:0, 1:0, 2:0, 3:0}
        key_len_counter = 0
        self.gen_cursor = 0
        self.current_state = 0
        if embedding is not None:
            self.embedding = embedding.to(self.device)
        else:
            self.embedding = torch.rand(self.embedding_size).to(self.device)
        file_generated_data = []
        file_predicted_states = []
        for _ in tqdm(range(file_length)):
            generated_segment = self.generator(self.embedding).to(self.device)
            generated_segment = generated_segment.round().type(torch.FloatTensor).to(self.device)
            file_generated_data.append(generated_segment.tolist())

            self.embedding = self.embedding_function(generated_segment, self.embedding).to(self.device)
            self.action = self.agents[self.current_state](self.embedding)
            self.previous_state = self.current_state
            self.current_state = torch.argmax(self.action.cpu()).item()


            # #correction mechanism
            if self.current_state == 2: 
                key_len_counter += 1 
            elif self.current_state == 3:
                if key_len_counter<2:
                    file_predicted_states[-2:] = [0, 0]
                    prediction_count[1] -= 1
                    prediction_count[2] -= 1
                    prediction_count[0] += 2
                    self.current_state = 0
                elif key_len_counter not in [2,3,4,8] :
                    file_predicted_states[-(key_len_counter+1):] = [0] * (key_len_counter+1)
                    prediction_count[0] += (key_len_counter+1)
                    prediction_count[1] -= 1
                    prediction_count[2] -= key_len_counter
                    self.current_state = 0
                key_len_counter = 0
            elif self.current_state ==1:
                if self.previous_state ==2:
                    file_predicted_states[-(key_len_counter+1):] = [0] * (key_len_counter+1)  
                    prediction_count[0] += (key_len_counter+1)                 
                    prediction_count[1] -= 1
                    prediction_count[2] -= key_len_counter
                    key_len_counter = 0
            if (self.current_state == 0 or self.current_state == 1 ) and  self.previous_state == 1:
                file_predicted_states[-1] = 0 
                prediction_count[1] -= 1
                prediction_count[0] += 1

            file_predicted_states.append(self.current_state)
            prediction_count[self.current_state] += 1 
            self.gen_cursor+= 1 
        if not supress_output:
            print(f"predictions distributions: {prediction_count}")
        if dump:
            self.dump_file(path= path, data= file_generated_data, states= file_predicted_states, last_file_index= last_file_index, supress_output=supress_output )

        return file_generated_data, file_predicted_states
    
    def dump_file(self, path , data, states, last_file_index, supress_output= False):
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Folder '{path}' created successfully.")

        joined_data= list(chain(*data))
        joined_data = [int(element) for element in joined_data]
        byte_data= bytearray(joined_data)
        byte_data_hex = binascii.hexlify(byte_data).decode('utf-8')

        key_positions = [(index+1)*8 for index, value in enumerate(states) if value == 1]
        metadata = {}
        for index, pos in enumerate(key_positions):
            key = f'key_{index + 1}_pos'
            metadata[key] = pos

        self.uuid_counter =last_file_index + 1
        incremental_uuid = uuid.UUID(int=self.uuid_counter)
        raw_file_name= f"{incremental_uuid}.raw"
        json_file_name= f"{incremental_uuid}.json"
        txt_file_name= f"{incremental_uuid}.txt"

        raw_file_path= f"{path}/{raw_file_name}"
        json_file_path= f"{path}/{json_file_name}"
        txt_file_path= f"{path}/{txt_file_name}"

        xxd_command = ["xxd", "-c", "8", "-"]
        xxd_output = subprocess.check_output(xxd_command, input=byte_data_hex, universal_newlines=True)

        with open(raw_file_path, "wb") as file:
            file.write(byte_data)
        with open(json_file_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        with open(txt_file_path, "w") as txt_file:
            txt_file.write(xxd_output)

        if not supress_output:
            print(f"Data saved in:\n{raw_file_path}\n{txt_file_path}\nMetadata saved in:\n{json_file_path}")
        return

    def extract(self,dataset, predictions, labeled_dataset = True):
        loader = DataLoader(dataset, batch_size= 1, shuffle= False)
        prediction_index = 0
        extracted_keys_int=[]
        extracted_keys_hex=[]
        keys_masks= []
        file_index= -1
        for data in loader:
            file_index += 1
            keys_per_file_int = []
            keys_per_file_hex= []
            keys_label_mask=[]
            if labeled_dataset:
                file_data= data[0]
                file_label = data[1]
            else:
                file_data= data

            file_predictions = predictions[prediction_index]
            malloc_positions = [index for index, value in enumerate(file_predictions) if value == 1]

            for position in malloc_positions:

                potential_key = []
                key_position = position +1

                while file_predictions[key_position] == 2:
                    potential_key.append(file_data[key_position])
                    key_position += 1

                if  1 < len(potential_key) <= 8:
                    flattened_potential_key = list(chain(*potential_key))
                    flattened_potential_key = [int(tensor.item()) for tensor in flattened_potential_key]
    
                    potential_bytearray= bytearray(flattened_potential_key)
                    flattened_potential_key_hex = binascii.hexlify(potential_bytearray).decode('utf-8')

                    keys_per_file_int.append(flattened_potential_key)
                    keys_per_file_hex.append(flattened_potential_key_hex)
                    if labeled_dataset:
                        if file_label[key_position] == 3: 
                            keys_label_mask.append(True)
                        else :
                            keys_label_mask.append(False)
            
            extracted_keys_int.append(keys_per_file_int)
            extracted_keys_hex.append(keys_per_file_hex)
            keys_masks.append(keys_label_mask)
            prediction_index += 1
        if labeled_dataset:
            data = {
                'extracted_keys_int': [item for sublist in extracted_keys_int for item in sublist],
                'extracted_keys_hex': [item for sublist in extracted_keys_hex for item in sublist],
                'key_label': [item for sublist in keys_masks for item in sublist],
                'file': [i for i, sublist in enumerate(extracted_keys_int) for _ in sublist]
            }
            df = pd.DataFrame(data)
            return df
        else:
            data = {
                'extracted_keys_int': [item for sublist in extracted_keys_int for item in sublist],
                'extracted_keys_hex': [item for sublist in extracted_keys_hex for item in sublist],
                'file': [i for i, sublist in enumerate(extracted_keys_int) for _ in sublist]
            }
            df = pd.DataFrame(data)
            return df
     
    def save_model(self,directory_path, data_version, length ,trial, type):
        if type not in ['prediction', 'generation']:
            raise ValueError("Invalid approach. Use 'prediction' or 'generation'.")
        if length is not None and data_version is not None:
            config=data_version + "__" + length
        elif length is None and data_version is not None:
            config=data_version 
        elif length is None and data_version is None:
            config=''
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
            print(f"Folder '{directory_path}' created successfully.")

        if not os.path.exists(f'{directory_path}/{config}'):
            os.mkdir(f'{directory_path}/{config}/')
            print(f"Folder {directory_path}/{config}/ created successfully.")

        if not os.path.exists(f'{directory_path}/{config}/{trial}/'):
            os.mkdir(f'{directory_path}/{config}/{trial}/')
            print(f"Folder {directory_path}/{config}/{trial}/ created successfully.")
        path= f"{directory_path}/{config}/{trial}/"

        embed_path= f'{path}/embedding.pth'
        agent_0_path= f'{path}/agent_0.pth'
        agent_1_path= f'{path}/agent_1.pth'
        agent_2_path= f'{path}/agent_2.pth'
        agent_3_path= f'{path}/agent_3.pth'
        generator_path= f'{path}/generator.pth'
        if type == 'prediction':
            torch.save(self.embedding_function.state_dict(), embed_path)
            torch.save(self.agent_0.state_dict(), agent_0_path)
            torch.save(self.agent_1.state_dict(), agent_1_path)
            torch.save(self.agent_2.state_dict(), agent_2_path)
            torch.save(self.agent_3.state_dict(), agent_3_path)
            print(f"embedding function and agents saved in '{path}'")
        elif type == 'generation':
            torch.save(self.generator.state_dict(), generator_path)
            print(f"generator saved in '{path}'")

    def load_model(self,directory_path, data_version, length, trial):
        if length is not None and data_version is not None:
            config=data_version + "__" + length
        elif length is None and data_version is not None:
            config=data_version 
        elif length is None and data_version is None:
            config=''
        path= f"{directory_path}/{config}/{trial}"  

        embed_path= f'{path}/embedding.pth'
        agent_0_path= f'{path}/agent_0.pth'
        agent_1_path= f'{path}/agent_1.pth'
        agent_2_path= f'{path}/agent_2.pth'
        agent_3_path= f'{path}/agent_3.pth'
        generator_path= f'{path}/generator.pth'
        print(f'Loading modules from {path}')
        print('-----------------------------------------------------------------------------')
        try:
            self.embedding_function.load_state_dict(torch.load(embed_path))
            self.agent_0.load_state_dict(torch.load(agent_0_path))
            self.agent_1.load_state_dict(torch.load(agent_1_path))
            self.agent_2.load_state_dict(torch.load(agent_2_path))
            self.agent_3.load_state_dict(torch.load(agent_3_path))
            print(f"Embedding function and Agents loaded.")
        except Exception as e:
            print(f'Error loading agents and embedding functions: {str(e)}')
        print('-----------------------------------------------------------------------------')
        try:
            self.generator.load_state_dict(torch.load(generator_path))
            print(f"Generator loaded.")
        except Exception as e:
            print(f'Error loading generator: {str(e)}')
        print('-----------------------------------------------------------------------------')


