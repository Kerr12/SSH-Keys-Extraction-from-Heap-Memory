from lib.SSH_extraction_pipeline import get_labeled_datasets, SHH_extractor
import datetime

# Function to print a timestamp
def print_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp)

print_timestamp()
print('Training/Validation script start over versions V_8_8_P1, V_8_7_P1, V_8_1_P1, V_8_0_P1, V_7_9_P1, V_7_8_P1, V_7_2_P1, V_7_1_P1, V_7_0_P1, V_6_9_P1, V_6_8_P1 ')
data_version = None
length = None

training_path = 'Data_V2/Training/basic'
validation_path = 'Data_V2/Validation/basic'
testing_path = 'Data_V2/Performance Test/basic'

# parameter_space has the following format [num_epochs, batch_size, learning_rate]
classification_parameter_space = [[5, 256, 0.001], [5, 512, 0.001], [5, 512, 0.0001], [5, 1024, 0.0001], [5, 2048, 0.0001]]
regression_parameter_space = [[200, 256, 0.001], [200, 512, 0.001], [200, 512, 0.0001], [200, 1024, 0.0001],[200, 2048, 0.0001]]

window_size = 512
stride = 64 
print(f'Window_size = {window_size}, stride_size (overlap) = {stride}')

print(f'Regression parameter space {regression_parameter_space}')
print(f'Classification parameter space {classification_parameter_space}')

print_timestamp()
classification_training_set, regression_training_set = get_labeled_datasets(path=training_path, window_size=window_size, stride=stride, version=data_version, length=length)
print_timestamp()
classification_validation_set, regression_validation_set = get_labeled_datasets(path=validation_path, window_size=window_size, stride=stride, version=data_version, length=length)

print('Loading Model')
model = SHH_extractor(window_size=window_size, stride=stride, cuda_index= 7)
print(f'running on {model.device}')
print_timestamp()
model.sequential_train(regression_validation_set, regression_parameter_space, 'regressor', True)

print_timestamp()
model.regression_test(regression_validation_set)

print_timestamp()
model.sequential_train(classification_training_set, classification_parameter_space, 'classifier', True)

print_timestamp()
model.classification_test(classification_validation_set, False)


print('Script end')
print('-------------------------------------------------------------------------------------------------------------------------------------------------------------------')
