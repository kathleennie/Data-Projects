import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.get_weights().update(x, nn.as_scalar(y))
                    flag = False
            if flag:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(1, 40)
        self.b1 = nn.Parameter(1, 40)

        self.w2 = nn.Parameter(40, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first = nn.Linear(x, self.w1)
        k = nn.ReLU(nn.AddBias(first, self.b1))

        second = nn.Linear(k, self.w2)
        k2 = nn.AddBias(second, self.b2)

        return k2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while (True):
            for x, y in dataset.iterate_once(2):
                loss = self.get_loss(x, y)
                g = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.update(g[0], -0.009)
                self.w2.update(g[1], -0.009)
                self.b1.update(g[2], -0.009)
                self.b2.update(g[3], -0.009)

            l = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(l) < 0.018:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(784, 250)
        self.b1 = nn.Parameter(1, 250)

        self.w2 = nn.Parameter(250, 150)
        self.b2 = nn.Parameter(1, 150)

        self.w3 = nn.Parameter(150, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first = nn.Linear(x, self.w1)
        k1 = nn.ReLU(nn.AddBias(first, self.b1))

        second = nn.Linear(k1, self.w2)
        k2 = nn.AddBias(second, self.b2)

        third = nn.Linear(k2, self.w3)
        k3 = nn.AddBias(third, self.b3)
        return k3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learn = -0.01
        while True:
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                grad = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                self.w1.update(grad[0], learn)
                self.w2.update(grad[1], learn)
                self.w3.update(grad[2], learn)

                self.b1.update(grad[3], learn)
                self.b2.update(grad[4], learn)
                self.b3.update(grad[5], learn)
            print(dataset.get_validation_accuracy())
            if dataset.get_validation_accuracy() >= 0.97:
                return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.dim = 5
        self.hDim = 256

        self.b1 = nn.Parameter(1, self.hDim)
        self.b2 = nn.Parameter(1, 256)
        self.b3 = nn.Parameter(1, 5)

        self.w = nn.Parameter(self.num_chars, self.hDim)
        self.w_H = nn.Parameter(self.hDim, self.hDim)
        self.w_F = nn.Parameter(self.hDim, self.dim)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        flag = True
        for i in xs:
            first = nn.AddBias(nn.Linear(i, self.w), self.b1)
            k = nn.Linear(nn.ReLU(first), self.w_H)
            second = nn.AddBias(k, self.b2)

            if flag:
                result = nn.ReLU(second)
                flag = False
            else:
                result = nn.Add(second, nn.Linear(nn.ReLU(result), self.w_H))

        return nn.AddBias(nn.Linear(nn.ReLU(result), self.w_F), self.b3)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learn = -0.008

        while True:
            for x, y in dataset.iterate_once(2):
                g = nn.gradients(self.get_loss(x, y), [self.w, self.w_H, self.w_F, self.b1, self.b2, self.b3])

                self.b1.update(g[3], learn)
                self.b2.update(g[4], learn)
                self.b3.update(g[5], learn)

                self.w.update(g[0], learn)
                self.w_H.update(g[1], learn)
                self.w_F.update(g[2], learn)

            print(dataset.get_validation_accuracy())

            if dataset.get_validation_accuracy() >= 0.81:
                return
