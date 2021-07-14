import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate,
                 hidden_layer_bias, output_layer_bias):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate

        # add bias
        self.hidden_layer_bias = hidden_layer_bias
        self.output_layer_bias = output_layer_bias

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        output = 1/(1+np.exp(0-input))  # TODO!
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = inputs[0] * self.hidden_layer_weights[0, i]  \
                           + inputs[1] * self.hidden_layer_weights[1, i] \
                           + inputs[2] * self.hidden_layer_weights[2, i] \
                           + inputs[3] * self.hidden_layer_weights[3, i]
            output = self.sigmoid(weighted_sum + self.hidden_layer_bias[i])
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = hidden_layer_outputs[0] * self.output_layer_weights[0, i] \
                           + hidden_layer_outputs[1] * self.output_layer_weights[1, i]
            output = self.sigmoid(weighted_sum+ self.output_layer_bias[i])
            output_layer_outputs.append(output)
        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):


        output_layer_betas = desired_outputs - output_layer_outputs

        # TODO! Calculate output layer betas.
        #print('OL betas: ', output_layer_betas)

        hidden_layer_betas = []
        #print(output_layer_outputs)
        for i in range(self.num_hidden):
            hlb = self.output_layer_weights[i, 0] * output_layer_outputs[0] * (1 - output_layer_outputs[0]) * output_layer_betas[0] \
                  + self.output_layer_weights[i, 1] * output_layer_outputs[1] * (1 - output_layer_outputs[1]) * output_layer_betas[1] \
                  + self.output_layer_weights[i, 2] * output_layer_outputs[2] * (1 - output_layer_outputs[2]) * output_layer_betas[2]
            hidden_layer_betas.append(hlb)
        # TODO! Calculate hidden layer betas.
        #print('HL betas: ', hidden_layer_betas)

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = []
        for i in range(self.num_hidden):
            dolw = [
                self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[0] * (1 - output_layer_outputs[0]) *
                output_layer_betas[0],
                self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[1] * (1 - output_layer_outputs[1]) *
                output_layer_betas[1],
                self.learning_rate * hidden_layer_outputs[i] * output_layer_outputs[2] * (1 - output_layer_outputs[2]) *
                output_layer_betas[2]
                ]
            delta_output_layer_weights.append(dolw)
        # TODO! Calculate output layer weight changes.
        #print(delta_output_layer_weights)

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = []
        for i in range(self.num_inputs):
            dhlw = [
                self.learning_rate * inputs[i] * hidden_layer_outputs[0] * (1 - hidden_layer_outputs[0]) *
                hidden_layer_betas[0],
                self.learning_rate * inputs[i] * hidden_layer_outputs[1] * (1 - hidden_layer_outputs[1]) *
                hidden_layer_betas[1]
                ]
            delta_hidden_layer_weights.append(dhlw)
        # TODO! Calculate hidden layer weight changes.

        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights):
        # TODO! Update the weights.
        self.hidden_layer_weights += delta_hidden_layer_weights
        self.output_layer_weights += delta_output_layer_weights
        #print('Placeholder')

    def train(self, instances, desired_outputs, epochs):

        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights, = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                max = np.max(output_layer_outputs)
                predicted_class = output_layer_outputs.index(max)  # TODO!
                predictions.append(predicted_class)
                #if predicted_class == 0:
                #    predictions.append([1, 0, 0])
                #if predicted_class == 1:
                #    predictions.append([0, 1, 0])
                #if predicted_class == 2:
                #    predictions.append([0, 0, 1])


                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)

            # TODO: Print accuracy achieved over this epoch

            # remake the desired_output into 0 1 2 to compare with prediction
            n = len(desired_outputs)
            d_outputs = []
            for i in range(n):
                if desired_outputs[i][0] == 1:
                    d_outputs.append(0)
                if desired_outputs[i][1] == 1:
                    d_outputs.append(1)
                if desired_outputs[i][2] == 1:
                    d_outputs.append(2)

            correct = 0
            for j in range(n):
                if d_outputs[j] == predictions[j]:
                    correct += 1

            acc = correct / n
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            max = np.max(output_layer_outputs)
            predicted_class = output_layer_outputs.index(max)
              # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions
