import random
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import Ridge


# This script implements a simple illustration of the use of Echo State Networks (ESNs).
# Followed basic set-up and training procedure described in: https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf


########################################################################################################################
# Some helper functions to generate training and testing data
########################################################################################################################

def get_square_wave(time_points=10000, period=20, show_plot=False):
    # Select starting value for pseudo-random square wave
    value = random.sample([0, 1], k=1)

    # Construct square wave signal
    wave = np.zeros([time_points])
    start = 0
    for stop in range(0, time_points, period):
        wave[start:stop] = value
        start = stop
        value = 0 if value == 1 else 1

    if show_plot:
        plot_sequence(y_sequence=wave, title='Square wave')

    return wave


def get_sin_wave(time_points=10000, div_factor=5, show_plot=False):
    time = np.arange(0, time_points, 1)
    wave = np.sin(time/div_factor)  # Div factor to vary frequency

    if show_plot:
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plot_sequence(y_sequence=wave, title='Sine wave')

    return wave


def combine_waves(wave1, wave2, time_points=10000, transitions=100, show_plot=False):
    # Generate time steps on which output wave alternates
    transition_points = sorted(random.sample(range(time_points), k=transitions))

    # Keep track of ground truth outputs
    y_target = np.zeros(time_points)

    # Select starting wave
    wave_id = random.sample([1, 2], k=1)

    # Construct square wave signal
    wave = np.zeros([time_points, 1])
    start = 0
    for stop in transition_points:
        wave[start:stop, 0] = wave1[start:stop] if wave_id == 1 else wave2[start:stop]
        y_target[start:stop] = wave_id
        start = stop
        wave_id = 2 if wave_id == 1 else 1

    if show_plot:
        plot_sequence(y_sequence=wave)

    return wave, y_target


########################################################################################################################
# Plotting functions
########################################################################################################################

def plot_neural_activity(state_history, num_nodes, num_disp=10):
    nodes = np.random.randint(0, num_nodes, size=num_disp)

    sequences, labels = [], []
    for node in nodes:
        sequences.append(state_history[:, node])
        labels.append('Node' + str(node))

    plot_sequences(sequences, labels, title='Evolution of activity of selected nodes', y_label='Activation')


def plot_sequence(y_sequence, title='Combined wave', time_points=10000):
    x = np.arange(0, time_points, 1)
    plot.plot(x, y_sequence)
    plot.title(title)
    plot.xlabel('Time')
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.show()


def plot_sequences(sequences, labels, title='Combined wave', x_label='Time', y_label='Amplitude', time_points=10000):

    x = np.arange(0, time_points, 1)
    for sequence in sequences:
        plot.plot(x, sequence)

    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    plot.legend(labels, loc='upper left')
    plot.show()


########################################################################################################################
# Calculate evaluation statistic
########################################################################################################################

def calc_NRMSE(pred, target):
    mu = np.mean(target)
    var = np.divide(np.sum(np.power(np.subtract(target, mu), 2)), target.size)
    nrmse = np.sqrt(
        np.divide(
            np.divide(
                np.sum(
                    np.power(
                        np.subtract(pred, target),
                        2
                    )
                ),
                target.size
            ),
            var
        )
    )
    return nrmse


########################################################################################################################
# Define model architecture
########################################################################################################################

class EchoStateNetwork:

    def __init__(self,
                 learning_rate=1.,
                 bias=True,
                 input_size=1,
                 reservoir_nodes=100,
                 nonlinearity=np.tanh,
                 reservoir_connectivity=0.2,  # How many percent of weights ought to be non-0
                 ridge_alpha=0.1
                 ):

        # Init some parameters
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.alpha = ridge_alpha
        self.learning_rate = learning_rate
        self.reservoir_nodes = reservoir_nodes

        # Compute network dimensions and stats
        reservoir_weights = reservoir_nodes ** 2
        reservoir_nonzeros = int(np.ceil(reservoir_weights * reservoir_connectivity))

        w_in_dim = [reservoir_nodes, input_size + (1 if bias else 0)]
        w_dim = [reservoir_nodes, reservoir_nodes]

        # Initialize input weights
        self.w_in = np.random.normal(loc=0.0, scale=.7, size=(w_in_dim[0], w_in_dim[1]))

        # Initialize reservoir's pseudo-random init state
        self.x_init = np.random.normal(loc=0.0, scale=.6, size=reservoir_nodes)

        # Initialize reservoir weights
        self.w = np.zeros(w_dim)
        nonzero_indices = np.random.randint(low=reservoir_nodes, size=(reservoir_nonzeros, 2))
        for index in nonzero_indices:
            self.w[index[0], index[1]] = np.random.normal(loc=0.0, scale=.2, size=None)

        # Placeholder init for ridge regression model
        self.ridge_model = None

        # Print stats:
        print('Average weight w_in:\n', np.mean(self.w_in))
        print('Average weight w (including 0s):\n', np.mean(self.w))
        print('Average weight w (excluding 0s):\n', np.true_divide(self.w.sum(), (self.w != 0).sum()))
        print('Average state value x_init:\n', np.mean(self.x_init))
        print('Average w*x_init:\n', np.mean(np.sum(self.w * self.x_init, axis=1)))


    def get_reservoir_states(self, sequence):
        # Get the evolution of the states of the reservoir's nodes as input sequence is presented to the reservoir

        x = self.x_init.copy()                                      # Reservoir init state
        x_history = np.zeros([time_steps, self.reservoir_nodes])    # Variable to be filled with reservoir states over time
        constant = np.array([1])                                    # Bias constant

        # Calculate reservoir's state for each time step t
        for t in range(time_steps):
            # Add bias constant to training input for t'th time step or not
            if self.bias:
                input = np.concatenate([constant, sequence[t]], axis=0)
            else:
                input = np.array(sequence[t])
            print('Input:\n', np.mean(input))
            print('w_in:\n', np.mean(self.w_in))
            print('self.w_in * input:\n', np.mean(self.w_in * input))

            # Compute components needed for updating reservoir
            in1 = np.sum(self.w_in * input, axis=1)                 # Input term (input weights * input)
            in2 = np.sum(self.w * x, axis=1)                        # Update term (recurrent reservoir weights * previous state)

            print('Input term:\t', np.mean(in1))
            print('Update term:\t', np.mean(in2))

            # Combine terms and apply non-linearity
            update = self.nonlinearity(in1 + in2)
            print('New x:\t\t', np.mean(update))

            # Update reservoir's state
            x = (1. - self.learning_rate) * x + self.learning_rate * update

            # Keep track of reservoir's states over course of processing input sequence
            x_history[t, :] = x

        return x_history


    def train(self, sequence, y_target):
        # Generate sequence of reservoir states as reservoir iterates over train input sequence
        x_history = self.get_reservoir_states(sequence)

        # Train output weights using ridge regression model
        self.ridge_model = Ridge(
            alpha=self.alpha,
            fit_intercept=True,
            copy_X=True
        ).fit(x_history, y_target)

        return x_history


    def predict(self, sequence):
        # Generate sequence of reservoir states as reservoir iterates over train input sequence
        x_history = self.get_reservoir_states(sequence)

        # For each time step, predict the driving signal using output weights
        y_predicted = self.ridge_model.predict(x_history)

        return y_predicted


########################################################################################################################
# Set hyperparameters
########################################################################################################################

time_steps = 10000
reservoir_nodes = 100
reservoir_connectivity = 0.25
ridge_alpha = 0.1
num_nodes_plotted = 5


########################################################################################################################
# Generate training and testing data
########################################################################################################################

# Get training input
square_wave = get_square_wave(time_points=time_steps, show_plot=True)
sine_wave = get_sin_wave(time_points=time_steps, show_plot=True)
wave, y_target = combine_waves(square_wave, sine_wave, show_plot=True)

# Get testing input
square_wave = get_square_wave(time_points=time_steps, show_plot=False)
sine_wave = get_sin_wave(time_points=time_steps, show_plot=False)
wave_test, y_target_test = combine_waves(square_wave, sine_wave, show_plot=False)


########################################################################################################################
# Train model
########################################################################################################################

esn = EchoStateNetwork(
    reservoir_nodes=reservoir_nodes,
    reservoir_connectivity=reservoir_connectivity,
    ridge_alpha=ridge_alpha
)
x_history = esn.train(wave, y_target)

plot_neural_activity(x_history, reservoir_nodes, num_disp=num_nodes_plotted)

########################################################################################################################
# Predict and plot on training data
########################################################################################################################

y_predicted = esn.predict(wave_test)
plot_sequences([wave, y_target, y_predicted], ['wave', 'y_target', 'y_predicted'], title='Train data')

########################################################################################################################
# Predict, plot, and evaluate on testing data
########################################################################################################################

y_predicted = esn.predict(wave_test)
plot_sequences([y_target_test, y_predicted], ['y_target_test', 'y_predicted'], title='Test data')

# Compute error
nrmse = calc_NRMSE(y_predicted, y_target_test)
print('NRMSE:', nrmse)
