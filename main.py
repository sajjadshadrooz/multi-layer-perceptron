from utils.import_data.get_data_from_csv import get_data_from_csv 
from utils.preprocess.preprocess_data import preprocess_data
from utils.process_engine.mlp import MLP


file_name = input("Please enter filenameof your CSV file: ")
data_frame = get_data_from_csv(file_name)

target_columns = input("Please enter target columns: ").split(" ")
test_size_precent = int(input("Please enter precent of test size: "))
train_x, train_y, test_x, test_y = preprocess_data(data_frame, target_columns, test_size_precent) 

hidden_layers = input("Please enter hidden layers: ").split(" ")
layers = [len(train_x.columns)]
layers.extend([int(i) for i in hidden_layers])
layers.append(len(train_y.columns))

learning_rate = float(input("Please enter learning rate: "))
activation_function = input("Please enter activation function: ")
model = MLP(layers, learning_rate, activation_function)

epochs = int(input("Please enter epochs: "))
model.train(train_x, train_y, epochs)

model.predict(train_x, train_y)
