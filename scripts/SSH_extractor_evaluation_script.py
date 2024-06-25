from lib.SSH_extraction_pipeline import get_labeled_datasets, SHH_extractor, get_Unlabeled_dataset

data_version=None
length= None
testing_path= 'Data_V2/Performance Test/basic'
window_size= 512
stride= 64

print('Model Trained on V_8_8_P1, V_8_7_P1, V_8_1_P1, V_8_0_P1, V_7_9_P1,V_7_8_P1, V_7_2_P1, V_7_1_P1, V_7_0_P1, V_6_9_P1, V_6_8_P1')
print('Testing on V_8_8_P1, V_8_7_P1, V_8_1_P1, V_8_0_P1, V_7_9_P1, V_7_8_P1, V_7_2_P1, V_7_1_P1, V_7_0_P1, V_6_9_P1, V_6_8_P1')
print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('Window_size = 512, stride_size (overlap) = 64')
classification_testing_set, regression_testing_set= get_labeled_datasets(path= testing_path , window_size= window_size, stride= stride, version= data_version, length= length)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
model = SHH_extractor( window_size= window_size, stride= stride)
model.load_units(window_size, stride, data_version, length)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
model.classification_test(classification_testing_set, sampling=False)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
model.regression_test(regression_testing_set)