import time
import numpy as np
from tqdm import tqdm, trange
from utils import train_test_split


class NeuralNetwork:
    def __init__(self, input_neurons, hidden_layers_neurons, hidden_layers_activations,
                 output_neurons, init_method='random', output_activation='linear',
                 leaky_relu_slope=0.1):

        self.params = {}
        assert len(hidden_layers_activations) == len(hidden_layers_neurons)
        self.activations = [None] + hidden_layers_activations + [output_activation]
        self.num_layers = len([input_neurons] + hidden_layers_neurons + [output_neurons])-1
        self.num_hidden_layers = len(hidden_layers_neurons)
        self.layer_sizes = [input_neurons] + hidden_layers_neurons + [output_neurons]
        self.output_activation = output_activation
        self.norm = None
        self.leaky_relu_slope = leaky_relu_slope
        np.random.seed(0)


        # Defining all the methods for intialising the weights for the forward pass
        if init_method == 'random':
            for i in range(1, self.num_layers + 1):
                self.params[f'W{i}'] = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i])
                self.params[f'B{i}'] = np.random.randn(1, self.layer_sizes[i])

        elif init_method == 'he':
            for i in range(1, self.num_layers + 1):
                self.params[f'W{i}'] = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) * np.sqrt(
                    2 / self.layer_sizes[i - 1])
                self.params[f'B{i}'] = np.random.randn(1, self.layer_sizes[i])

        elif init_method == 'xavier':
            for i in range(1, self.num_layers + 1):
                self.params[f'W{i}'] = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) * np.sqrt(
                    1 / self.layer_sizes[i - 1])
                self.params[f'B{i}'] = np.random.randn(1, self.layer_sizes[i])

        self.forward_pass_values = {}
        self.gradients = {}

    # Defining all the Activation functions
    def activation(self, X, activation_fun):
        if activation_fun == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-X))
        elif activation_fun == 'tanh':
            return np.tanh(X)
        elif activation_fun == 'relu':
            return np.maximum(0, X)
        elif activation_fun == 'leaky_relu':
            return np.maximum(self.leaky_relu_slope * X, X)
        elif activation_fun == 'linear':
            return X

    # Activation functions for the Backpropagation
    def grad_activation(self, X, activation_fun):
        if activation_fun == 'sigmoid':
            return X * (1 - X)
        elif activation_fun == 'tanh':
            return (1 - np.square(X))
        elif activation_fun == 'relu':
            return 1.0 * (X > 0)
        elif activation_fun == 'leaky_relu':
            derivatives = np.zeros_like(X)
            derivatives[X <= 0] = self.leaky_relu_slope
            derivatives[X > 0] = 1
            return derivatives
        elif activation_fun == 'linear':
            return np.ones_like(X)

    # Forward pass 
    def forward(self, X):
        params = self.params

        self.forward_pass_values['A1'] = np.matmul(X, params['W1']) + params['B1']
        self.forward_pass_values['H1'] = self.activation(self.forward_pass_values['A1'], self.activations[1])

        for i in range(2, self.num_layers+1):
            self.forward_pass_values[f'A{i}'] = np.matmul(self.forward_pass_values[f'H{i-1}'], params[f'W{i}']) + params[f'B{i}']
            self.forward_pass_values[f'H{i}'] = self.activation(self.forward_pass_values[f'A{i}'], self.activations[i])

        return self.forward_pass_values[f'H{self.num_layers}']

    # Defining the Loss functions
    def get_loss(self, y_true, y_pred):
        if self.loss_fun == 'mse':
            return np.square(np.subtract(y_true, y_pred)).mean()
        elif self.loss_fun == 'rmse':
            mse = np.square(np.subtract(y_true, y_pred)).mean()
            return np.sqrt(mse)
        elif self.loss_fun == 'mae':
            return np.absolute(np.subtract(y_true, y_pred)).mean()

    # Gradients for the computation in the Neural Network
    def grad(self, X, Y):
        predictions = self.forward(X)

        params = self.params

        H_L = f'dH{self.num_layers}'
        A_L = f'dA{self.num_layers}'
        B_L = f'dB{self.num_layers}'
        W_L = f'dW{self.num_layers}'

        H_L_values = self.forward_pass_values[f'H{self.num_layers}']
        self.gradients[H_L] = 2*(H_L_values - Y)
        self.gradients[A_L] = np.multiply(self.gradients[H_L], self.grad_activation(H_L_values, self.output_activation))
        self.gradients[W_L] = None
        if self.num_hidden_layers == 0:
            self.gradients[W_L] = np.matmul(X.T, self.gradients[A_L])
        else:
            self.gradients[W_L] = np.matmul(self.forward_pass_values[f'H{self.num_layers-1}'].T, self.gradients[A_L])
        self.gradients[B_L] = np.sum(self.gradients[A_L], axis=0).reshape(1, -1)

        if self.num_hidden_layers == 0:
            return predictions

        for i in range(self.num_layers-1, 1, -1):
            dH_i = self.gradients[f'dH{i}'] = np.matmul(self.gradients[f'dA{i+1}'], params[f'W{i+1}'].T)
            H_i = self.forward_pass_values[f'H{i}']
            dA_i = self.gradients[f'dA{i}'] = np.multiply(dH_i, self.grad_activation(H_i, self.activations[i]))
            self.gradients[f'dW{i}'] = np.matmul(self.forward_pass_values[f'H{i-1}'].T, dA_i)
            self.gradients[f'dB{i}'] = np.sum(dA_i, axis=0).reshape(1, -1)

        dH_1 = self.gradients['dH1'] = np.matmul(self.gradients[f'dA2'], params[f'W2'].T)
        H_1 = self.forward_pass_values['H1']
        dA_1 =self.gradients['dA1'] = np.multiply(dH_1, self.grad_activation(H_1, self.activations[1]))
        self.gradients['dW1'] = np.matmul(X.T, dA_1)
        self.gradients['dB1'] = np.sum(dA_1, axis=0).reshape((1, -1))

        return predictions


    
    # Model fitting
    # Dividing the whole input data into batches 
    # Performing the predictions using the training data
    def fit(self, X, Y, X_eval=None, Y_eval=None, epochs=1,
            algo='MiniBatch', mini_batch_size=1, lr=0.01, loss_fun='mse'):
        self.loss_fun = loss_fun
        train_losses = {}
        val_losses = {}

        # X = self.normalise(X)

        for num_epoch in range(epochs):
            print(f'\nEpoch [{num_epoch+1} / {epochs}]')
            m = X.shape[0]
            indexes = np.arange(0, m)
            np.random.shuffle(indexes)
            X, Y = X[indexes], Y[indexes]

            if algo == "GD":
                t = trange(1, desc='Loss', leave=True)
                predictions = self.grad(X, Y)
                for i in range(1, self.num_layers + 1):
                    self.params[f'W{i}'] -= lr * (self.gradients[f'dW{i}'] / m)
                    self.params[f'B{i}'] -= lr * (self.gradients[f'dB{i}'] / m)
                train_loss = self.get_loss(y_true=Y, y_pred=predictions)
                t.set_description(f'Epoch Loss: {train_loss}')
                t.refresh()
                train_losses[num_epoch] = train_loss
            elif algo == "MiniBatch":
                predictions = []
                Ks = list(range(0, m, mini_batch_size))
                total_batches = len(Ks)
                t = trange(total_batches, desc='Loss', leave=True)
                for i in t:
                    k = Ks[i]
                    batch_predictions = self.grad(X[k:k + mini_batch_size], Y[k:k + mini_batch_size])
                    predictions += list(batch_predictions.ravel())
                    batch_loss = self.get_loss(Y[k:k+mini_batch_size], batch_predictions)
                    t.set_description(f'Batch loss [{i+1} / {total_batches}] ({self.loss_fun}): {batch_loss}')
                    t.refresh()
                    time.sleep(0.005)
                    for i in range(1, self.num_layers + 1):
                        self.params[f'W{i}'] -= lr * (self.gradients[f'dW{i}'] / mini_batch_size)
                        self.params[f'B{i}'] -= lr * (self.gradients[f'dB{i}'] / mini_batch_size)

                predictions = np.array(predictions).reshape(-1, 1)
                train_loss = self.get_loss(y_true=Y, y_pred=predictions)
                # print("Train Loss:",train_loss)
                # # print("Valid Loss:",val_loss)
                if X_eval is not None:
                    val_loss = self.evaluate(X_eval, Y_eval)
                    print(f'\nEpoch [{num_epoch+1} / {epochs}] Train loss ({self.loss_fun}): {train_loss} '
                          f'Validation loss ({self.loss_fun}): {val_loss}')
                    val_losses[num_epoch] = val_loss
                else:
                    print(f'\nEpoch [{num_epoch+1} / {epochs}] Train loss ({self.loss_fun}): {train_loss}')
                train_losses[num_epoch] = train_loss
      
                print("Train Losses", train_losses)
                print("Valid Losses", val_losses)
    
    
    def normalise(self, X):
        norm = np.linalg.norm(X)
        self.norm = norm
        X = X / norm

        return X

    def predict(self, x):
        if self.norm:
            x = x/self.norm
        return self.forward(x).item()

    def evaluate(self, x, y):
        predictions = self.forward(x)
        return self.get_loss(y, predictions)


if __name__ == '__main__':
    np.random.seed(0)
    X_total = np.random.randn(20000, 5)
    Y_total = np.sum(np.square(X_total), axis=1) - 10
    Y_total = Y_total.reshape(-1, 1)
    print(Y_total.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.2)
    print(f'Train data (X_train) size: {X_train.shape}')
    print(f'Train data (Y_train) size: {Y_train.shape}')
    print(f'Test data (X_test) size: {X_test.shape}')
    print(f'Test data (Y_test) size: {Y_test.shape}')

    model = NeuralNetwork(input_neurons=X_train.shape[-1],
                          hidden_layers_neurons=[2],
                          hidden_layers_activations=['sigmoid'],
                          output_neurons=1,
                          output_activation='linear')

    model.fit(X_train, Y_train, X_test, Y_test, algo='MiniBatch', mini_batch_size=256, epochs=30, lr=0.001)

    test_point = np.array([[1, 2, 3, 4, 5]]).reshape(1, -1)
    test_y = np.sum(np.square(test_point), axis=1) - 10
    test_y = test_y.reshape(-1, 1)
    model.predict(test_point)

    for param in model.gradients.keys():
        if 'W' in param or 'B' in param:
            print(param, model.gradients[param])
