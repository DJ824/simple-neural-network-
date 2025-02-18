import copy
import os
import urllib
import urllib.request
import numpy as np
import cv2
import pickle
import copy
import matplotlib.pyplot as plt
import tarfile
from numpy.random import default_rng

rng = default_rng()
os.environ['ACCELERATE_DISABLE_VFORCE'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0.01, weight_regularizer_l2=0.01,
                 bias_regularizer_l1=0.01, bias_regularizer_l2=0.01):
        self.weights = np.ascontiguousarray(0.01 * rng.standard_normal((n_inputs, n_neurons)))
        self.biases = np.ascontiguousarray(np.zeros((1, n_neurons)))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        inputs = np.ascontiguousarray(inputs)
        self.inputs = inputs
        self.output = np.ascontiguousarray(np.dot(inputs, self.weights) + self.biases)

    def backward(self, dvalues):
        dvalues = np.ascontiguousarray(dvalues)

        self.dweights = np.ascontiguousarray(np.dot(self.inputs.T, dvalues))
        self.dbiases = np.ascontiguousarray(np.sum(dvalues, axis=0, keepdims=True))

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.ascontiguousarray(np.dot(dvalues, self.weights.T))


class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs


class Activation_ReLU:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.ascontiguousarray(np.maximum(0, inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        self.dinputs = np.ascontiguousarray(self.dinputs)

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs = inputs
        # unnormalized prob
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobain matrix of output, results in pos values on main diag (when i == j)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # chain rule gradient of outputs by incoming gradient to get gradient wrt inputs
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs


class Loss:
    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + \
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - \
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs, training=True)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = np.ascontiguousarray(
            self.beta_1 * layer.weight_momentums +
            (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = np.ascontiguousarray(
            self.beta_1 * layer.bias_momentums +
            (1 - self.beta_1) * layer.dbiases
        )

        weight_momentums_corrected = layer.weight_momentums / \
                                     (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
                                   (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = np.ascontiguousarray(
            self.beta_2 * layer.weight_cache +
            (1 - self.beta_2) * layer.dweights ** 2
        )
        layer.bias_cache = np.ascontiguousarray(
            self.beta_2 * layer.bias_cache +
            (1 - self.beta_2) * layer.dbiases ** 2
        )

        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights = np.ascontiguousarray(
            layer.weights - self.current_learning_rate *
            weight_momentums_corrected /
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases = np.ascontiguousarray(
            layer.biases - self.current_learning_rate *
            bias_momentums_corrected /
            (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        self.iterations += 1


class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        import gc

        self.accuracy.init(y)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
            X_val = np.ascontiguousarray(X_val, dtype=np.float32)
            y_val = np.ascontiguousarray(y_val, dtype=np.float32)

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                    batch_X = np.ascontiguousarray(batch_X)
                    batch_y = np.ascontiguousarray(batch_y)

                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

                gc.collect()

            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

            gc.collect()

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        # remove data from input layer and gradients from loss
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)


def load_dataset(dataset, path, image_size=(64, 64), augment=False):
    X = []
    y = []

    labels = [f for f in os.listdir(os.path.join(path, dataset))
              if not f.startswith('.') and
              os.path.isdir(os.path.join(path, dataset, f))]

    label_to_idx = {label: idx for idx, label in enumerate(sorted(labels))}

    for label in labels:
        label_path = os.path.join(path, dataset, label)
        files = [f for f in os.listdir(label_path)
                 if not f.startswith('.') and
                 os.path.isfile(os.path.join(label_path, f))]

        for file in files:
            image_path = os.path.join(label_path, file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, image_size)

            X.append(image)
            y.append(label_to_idx[label])

            if augment:
                flipped = cv2.flip(image, 1)
                X.append(flipped)
                y.append(label_to_idx[label])

                M = cv2.getRotationMatrix2D((image_size[0] / 2, image_size[1] / 2), 10, 1.0)
                rotated = cv2.warpAffine(image, M, image_size)
                X.append(rotated)
                y.append(label_to_idx[label])

                adjusted = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
                X.append(adjusted)
                y.append(label_to_idx[label])

    X = np.array(X)
    y = np.array(y)

    num_classes = len(labels)
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    return X, y_one_hot


def create_model():
    model = Model()

    model.add(Layer_Dense(3072, 1024,
                          weight_regularizer_l1=0, weight_regularizer_l2=5e-4,
                          bias_regularizer_l1=0, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.2))

    model.add(Layer_Dense(1024, 512,
                          weight_regularizer_l1=0, weight_regularizer_l2=5e-4,
                          bias_regularizer_l1=0, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.3))

    model.add(Layer_Dense(512, 256,
                          weight_regularizer_l1=0, weight_regularizer_l2=5e-4,
                          bias_regularizer_l1=0, bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.3))

    model.add(Layer_Dense(256, 10,
                          weight_regularizer_l1=0, weight_regularizer_l2=5e-4,
                          bias_regularizer_l1=0, bias_regularizer_l2=5e-4))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(
            learning_rate=0.001,
            decay=1e-4
        ),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()
    return model


def download_cifar10(path='./data'):
    if not os.path.exists(path):
        os.makedirs(path)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = os.path.join(path, "cifar-10-python.tar.gz")

    if not os.path.exists(os.path.join(path, 'cifar-10-batches-py')):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)

        print("Extracting files...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=path)

        print("Cleaning up...")
        os.remove(filename)
        print("Done!")


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar10_data(path='./data'):
    download_cifar10(path)

    cifar_path = os.path.join(path, 'cifar-10-batches-py')

    X_train = []
    y_train = []

    for batch in range(1, 6):
        batch_file = os.path.join(cifar_path, f'data_batch_{batch}')
        data_dict = unpickle(batch_file)

        X_batch = np.ascontiguousarray(data_dict[b'data'])
        y_batch = data_dict[b'labels']

        X_train.append(X_batch)
        y_train.extend(y_batch)

    X_train = np.ascontiguousarray(np.concatenate(X_train, axis=0))
    y_train = np.array(y_train)

    test_file = os.path.join(cifar_path, 'test_batch')
    test_dict = unpickle(test_file)

    X_test = np.ascontiguousarray(test_dict[b'data'])
    y_test = np.array(test_dict[b'labels'])

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    num_classes = 10
    y_train_one_hot = np.zeros((y_train.shape[0], num_classes), dtype=np.float32)
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    y_train_one_hot = np.ascontiguousarray(y_train_one_hot)

    y_test_one_hot = np.zeros((y_test.shape[0], num_classes), dtype=np.float32)
    y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1
    y_test_one_hot = np.ascontiguousarray(y_test_one_hot)

    return X_train, y_train_one_hot, X_test, y_test_one_hot


# if __name__ == "__main__":
#     X_train, y_train, X_test, y_test = load_cifar10_data()
#
#     print("Data shapes:")
#     print(f"X_train: {X_train.shape}")
#     print(f"y_train: {y_train.shape}")
#     print(f"X_test: {X_test.shape}")
#     print(f"y_test: {y_test.shape}")
#
#     model = create_model()
#
#     model.train(X_train, y_train,
#                 validation_data=(X_test, y_test),
#                 epochs=50,
#                 batch_size=128,
#                 print_every=100)
#
#     model.evaluate(X_test, y_test)

# parameters = model.get_parameters()


fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

image_data = cv2.imread('test3.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) -
              127.5) / 127.5


model = Model.load('fashion_mnist.model')

confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
print(fashion_mnist_labels[predictions[0]])
