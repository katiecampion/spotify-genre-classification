import numpy as np


class NeuralNetwork:
    """
    multi-layer NN:
    -input layer
    -hidden layers with ReLU activation
    -output layer with softmax activation
    -cross-entropy loss
    """

    def __init__(self, layer_sizes, learning_rate=0.01, random_state=42):
        """
        initialize the neural network
        parameters:
        -layer_sizes: list of layer sizes
        -learning_rate: learning rate for gradient descent
        -random_state: random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes)

        np.random.seed(random_state)

        #initialise weights and biases for each layer
        self.weights = []
        self.biases = []

        for i in range(self.n_layers - 1):
            #initialization for relu
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        #training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    #relu activation
    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        #derivative of relu for backpropagation
        return (z > 0).astype(float)

    #turns outputs into probabilities
    def softmax(self, z): #
        #softmax activation function for output layer
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        """
        cross-entropy
        parameters:
        -y_pred: predicted probabilities (n_samples, n_classes)
        -y_true: true labels (one-hot encoded)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def one_hot_encode(self, y, n_classes):
        #convert integer labels to one-hot encoded vectors
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    #foward propagation 
    def forward(self, X):
        """
        forward propagation
        parameters:
        -X: Input data (n_samples, n_features)

        returns:
        -output probabilities (n_samples, n_classes)
        """
        #store activations and z values for backpropagation
        self.activations = [X]
        self.z_values = []

        current = X

        #hidden layers with relu
        for i in range(self.n_layers - 2):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = self.relu(z)
            self.activations.append(current)

        #output layers with softmax
        z = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = self.softmax(z)
        self.activations.append(output)

        return output

    #backward propagation 
    def backward(self, y_true):
        """
        backward propagation to compute gradients
        parameters:
        -y_true: true labels 
        returns:
        -Lists of weight and bias gradients
        """
        m = y_true.shape[0]
        weight_gradients = []
        bias_gradients = []

        #output layer gradient softmax + cross-entropy
        delta = self.activations[-1] - y_true

        #backpropagate through layers
        for i in range(self.n_layers - 2, -1, -1):
            #gradients for weights and biases
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)

            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)

            #propagate to previous layer if not in input layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients):
        #update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_size=32, verbose=True):
        """
        train the neural network

        parameters:
        -X_train: training features
        -y_train: training labels
        -X_val: validation features
        -y_val: validation labels
        -epochs: number of training epochs
        -batch_size: mini-batch size
        -verbose: print progress

        returns:
        -training history dictionary
        """
        n_samples = X_train.shape[0]
        n_classes = self.layer_sizes[-1]

        #one hot encode labels
        y_train_onehot = self.one_hot_encode(y_train, n_classes)
        if y_val is not None:
            y_val_onehot = self.one_hot_encode(y_val, n_classes)

        #training loop
        for epoch in range(epochs):
            #shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]

            #mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                #forward pass
                self.forward(X_batch)

                #backward pass
                weight_gradients, bias_gradients = self.backward(y_batch)

                #update weights
                self.update_weights(weight_gradients, bias_gradients)

            #calculate metrics for this epoch
            train_pred = self.forward(X_train)
            train_loss = self.cross_entropy_loss(train_pred, y_train_onehot)
            train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)

            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.cross_entropy_loss(val_pred, y_val_onehot)
                val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)

                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")

        return self.history

    def predict(self, X):
        """
        predict class labels
        parameters:
        -X: Input features
        returns:
        -predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        """
        predict class probabilities
        parameters:
        -x: Input features

        returns:
        -class probabilities (n_samples, n_classes)
        """
        return self.forward(X)

    def score(self, X, y):
        """
        calculate accuracy score
        parameters:
        -x: Input features
        -y: true labels 
        returns:
        -accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def create_network(n_features, n_classes, hidden_layers=[32, 16], learning_rate=0.01):
    """
    function to create a neural network
    parameters:
    -n_features: number of input features
    -n_classes: number of output classes
    -hidden_layers: list of hidden layer sizes
    -learning_rate: learning rate

    returns:
    -neuralNetwork instance
    """
    layer_sizes = [n_features] + hidden_layers + [n_classes]
    return NeuralNetwork(layer_sizes, learning_rate=learning_rate)


if __name__ == "__main__":
    #test with dummy data
    print("Neural network module")

    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 14)  # 1000 samples, 14 features
    y = np.random.randint(0, 12, 1000)  # 12 classes

    # Split into train/val
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    # Create and train network
    nn = create_network(n_features=14, n_classes=12, hidden_layers=[64, 32])
    history = nn.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    print(f"\ntraining accuracy: {nn.score(X_train, y_train):.4f}")
    print(f"validation accuracy: {nn.score(X_val, y_val):.4f}")
