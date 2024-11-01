import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Neural Network Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def load_cross_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        while True:
            identifier = file.readline().strip()
            if not identifier:
                break  # End of file
            values = file.readline().strip().split()
            label_values = file.readline().strip().split()
            data.append([float(values[0]), float(values[1])])
            labels.append([int(label_values[0]), int(label_values[1])])
    return np.array(data), np.array(labels)

class MLP_cross:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.hidden_bias = np.random.uniform(-1, 1, hidden_size)
        self.output_bias = np.random.uniform(-1, 1, output_size)

        self.delta_weights_input_hidden = np.zeros((input_size, hidden_size))
        self.delta_weights_hidden_output = np.zeros((hidden_size, output_size))

    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.hidden_bias
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.output_bias
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, inputs, expected_output):
        output_errors = expected_output - self.final_output
        output_delta = output_errors * sigmoid_derivative(self.final_input)

        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_errors * sigmoid_derivative(self.hidden_input)

        self.delta_weights_hidden_output = (self.learning_rate * np.outer(self.hidden_output, output_delta) +
                                            self.momentum * self.delta_weights_hidden_output)
        self.delta_weights_input_hidden = (self.learning_rate * np.outer(inputs, hidden_delta) +
                                           self.momentum * self.delta_weights_input_hidden)

        self.weights_hidden_output += self.delta_weights_hidden_output
        self.weights_input_hidden += self.delta_weights_input_hidden
        self.output_bias += self.learning_rate * output_delta
        self.hidden_bias += self.learning_rate * hidden_delta

    def train(self, inputs, expected_output):
        self.forward(inputs)
        self.backward(inputs, expected_output)

    def predict(self, inputs):
        return self.forward(inputs)

def cross_validation_cross_pat(data_input, data_output, k_folds, learning_rate, momentum, epochs):
    fold_size = len(data_input) // k_folds
    all_true_labels = []
    all_pred_labels = []
    for fold in range(k_folds):
        validation_start = fold * fold_size
        validation_end = validation_start + fold_size

        validation_input = data_input[validation_start:validation_end]
        validation_output = data_output[validation_start:validation_end]

        train_input = np.concatenate((data_input[:validation_start], data_input[validation_end:]))
        train_output = np.concatenate((data_output[:validation_start], data_output[validation_end:]))

        mlp = MLP_cross(input_size=2, hidden_size=2, output_size=2, learning_rate=learning_rate, momentum=momentum)

        for epoch in range(epochs):
            for i in range(len(train_input)):
                mlp.train(train_input[i], train_output[i])

        for i in range(len(validation_input)):
            pred = mlp.predict(validation_input[i])
            all_true_labels.append(np.argmax(validation_output[i]))
            all_pred_labels.append(np.argmax(pred))

    return all_true_labels, all_pred_labels

def plot_confusion_matrix(true_labels, pred_labels):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center", color="red")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# FloodPredictionModel Class
class FloodPredictionModel:
    def __init__(self, train_data_path, test_data_path, hidden_size=5, learning_rate=0.0003, momentum_rate=0.5, epochs=80000):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.epochs = epochs
        self.input_data, self.output_data = self.load_and_normalize_data(train_data_path, 252, 9)
        self.unseen_input_data, self.unseen_output_data = self.load_and_normalize_data(test_data_path, 63, 9)
        self.input_size = self.input_data.shape[1]
        self.output_size = self.output_data.shape[1]
        self.initialize_weights()
        
    def load_and_normalize_data(self, file_path, rows, columns):
        data = np.zeros((rows, columns))
        with open(file_path, 'r') as file:
            for l, line in enumerate(file):
                if l < rows:
                    values = line.strip().split()
                    for column in range(columns):
                        data[l, column] = float(values[column])
        return data[:, :columns-1] / 600, data[:, columns-1:] / 600
    
    def initialize_weights(self):
        np.random.seed(42)
        self.weights_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.biases_hidden = np.random.rand(self.hidden_size)
        self.weights_output = np.random.rand(self.hidden_size, self.output_size)
        self.biases_output = np.random.rand(self.output_size)
        self.prev_weights_output_change = np.zeros_like(self.weights_output)
        self.prev_biases_output_change = np.zeros_like(self.biases_output)
        self.prev_weights_hidden_change = np.zeros_like(self.weights_hidden)
        self.prev_biases_hidden_change = np.zeros_like(self.biases_hidden)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def train(self):
        plot_epoch = []
        plot_loss = []
        
        for epoch in range(self.epochs):
            
            hidden_layer_input = np.dot(self.input_data, self.weights_hidden) + self.biases_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.biases_output
            predicted_output = output_layer_input
          
            loss = np.mean((predicted_output - self.output_data) ** 2)
            loss = round(loss, 8)
           
            output_error = predicted_output - self.output_data
            output_gradient = output_error
            weights_output_change = (self.learning_rate * np.dot(hidden_layer_output.T, output_gradient)) + (self.momentum_rate * self.prev_weights_output_change)
            biases_output_change = (self.learning_rate * np.sum(output_gradient, axis=0)) + (self.momentum_rate * self.prev_biases_output_change)
            self.weights_output -= weights_output_change
            self.biases_output -= biases_output_change

            hidden_error = np.dot(output_gradient, self.weights_output.T) * self.sigmoid_derivative(hidden_layer_input)
            weights_hidden_change = (self.learning_rate * np.dot(self.input_data.T, hidden_error)) + (self.momentum_rate * self.prev_weights_hidden_change)
            biases_hidden_change = (self.learning_rate * np.sum(hidden_error, axis=0)) + (self.momentum_rate * self.prev_biases_hidden_change)
            self.weights_hidden -= weights_hidden_change
            self.biases_hidden -= biases_hidden_change

            self.prev_weights_output_change = weights_output_change
            self.prev_biases_output_change = biases_output_change
            self.prev_weights_hidden_change = weights_hidden_change
            self.prev_biases_hidden_change = biases_hidden_change

            if epoch % 100 == 0:
                plot_epoch.append(epoch)
                plot_loss.append(loss)

        print(f"Final loss: {round(loss, 8)}")
        predicted_output = predicted_output * 600
        return plot_epoch, plot_loss, predicted_output
        
    def plot_results(self, parameter_results):
        plt.figure(figsize=(12, 12))

        # Plot loss vs. epochs
        plt.subplot(3, 1, 1)
        for params_key, results in parameter_results.items():
            epochs, losses = results['loss']
            plt.plot(epochs, losses, label=f"LR: {params_key[0]}, HS: {params_key[1]}, MR: {params_key[2]}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0.0001, 0.009)
        plt.title("Loss vs Epochs for Different Parameters")
        plt.legend()
        plt.grid(True)

        # Plot predicted output vs. desired output
        plt.subplot(3, 1, 2)
        for params_key, results in parameter_results.items():
            _, predicted_output = results['prediction']
            plt.plot(predicted_output / 600, label=f"LR: {params_key[0]}, HS: {params_key[1]}, MR: {params_key[2]}")
        plt.xlabel("Data Index")
        plt.ylabel("Output")
        plt.title("Predicted Output for Different Parameters")
        plt.legend()
        plt.grid(True)

        # Plot test data results
        plt.subplot(3, 1, 3)
        hidden_layer_input = np.dot(self.unseen_input_data, self.weights_hidden) + self.biases_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_output) + self.biases_output
        predicted_test_output = output_layer_input * 600
        plt.plot(self.unseen_output_data * 600, label="Desired Test Output", marker='x', color='red')
        plt.plot(predicted_test_output, label="Predicted Test Output", marker='o', linestyle='--', color='blue')
        plt.xlabel("Data Index")
        plt.ylabel("Output")
        plt.title("Desired vs Predicted Output for Test Data")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Cross-validation for MLP_cross
def run_cross_validation_and_flood_model():
    # Load cross.pat data
    cross_data, cross_labels = load_cross_data('cross')

    # Normalize cross data
    cross_data_normalized = (cross_data - np.min(cross_data, axis=0)) / (np.max(cross_data, axis=0) - np.min(cross_data, axis=0))

    # Cross-validation for cross.pat data
    learning_rates = [0.01]
    momentum = 0.9
    epochs = 5000
    k_folds = 10

    for lr in learning_rates:
        true_labels, pred_labels = cross_validation_cross_pat(cross_data_normalized, cross_labels, k_folds, lr, momentum, epochs)
        accuracy = np.sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)
        print(f"Learning Rate: {lr}, Accuracy: {accuracy}")
        
        # Plot confusion matrix
        plot_confusion_matrix(true_labels, pred_labels)

    # File paths for flood prediction
    train_data_path = 'flood_data_set'
    test_data_path = 'flood_data_test'

    # Parameter sets to test
    parameter_sets = [
        (0.0001, 2, 0.1, 80000),  # (learning_rate, hidden_nodes, momentum_rate, epochs)
        (0.0001, 2, 0.5, 80000), 
        (0.0001, 2, 0.9, 80000)
    ]

    parameter_results = {}
    for params in parameter_sets:
        learning_rate, hidden_size, momentum_rate, epochs = params
        print(f"Training model with parameters: hidden_size={hidden_size}, learning_rate={learning_rate}, momentum_rate={momentum_rate}, epochs={epochs}")
        model = FloodPredictionModel(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            momentum_rate=momentum_rate,
            epochs=epochs
        )
        plot_epoch, plot_loss, predicted_output = model.train()
        parameter_results[params] = {
            'loss': (plot_epoch, plot_loss),
            'prediction': (plot_epoch, predicted_output)  # Store plot_epoch if needed
        }

    # Plot the results
    model.plot_results(parameter_results)

# Run the combined function
run_cross_validation_and_flood_model()
