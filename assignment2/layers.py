import numpy as np


def l2_regularization(W, reg_strength):
    loss = reg_strength*np.sum(np.square(W))
    grad = 2*reg_strength*W
    
    return loss, grad

def cross_entropy_loss(probs, target_index):
    loss= {}
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_arr = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(loss_arr) / batch_size    
    return loss

def softmax(predictions):
    pred = np.copy(predictions)
    probs = {}
    if (pred.ndim == 1):
        pred -= np.max(predictions)
        probs = np.exp(pred)/np.sum(np.exp(pred))
    else:
        pred = [pred[i] - np.max(pred[i]) for i in range(pred.shape[0])]
        exp_pred = np.exp(pred)
        exp_sum = np.sum(exp_pred, axis = 1)
        probs = np.asarray([exp_pred[i]/exp_sum[i] for i in range(exp_pred.shape[0])])
    return probs
def softmax_with_cross_entropy(predictions, target_index):
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    #print(loss)
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[range(batch_size), target_index] -= 1
        dprediction /= batch_size    
    #print(dprediction)
    return loss, dprediction

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return np.vectorize(lambda x: 0 if x <= 0 else x)(X.reshape(-1)).reshape(X.shape[0], X.shape[1])

        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        return frwrd
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.vectorize(lambda x: 0 if x < 0 else 1)(self.X.reshape(-1)).reshape(self.X.shape[0], self.X.shape[1])
        return d_result * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X,self.W.value)+self.B.value
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)
        
        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
