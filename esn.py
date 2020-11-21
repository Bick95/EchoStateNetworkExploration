import random
import numpy as np
import matplotlib.pyplot as plot
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# This script implements a simple illustration of the use of Echo State Networks (ESNs).
# Followed set-up and training procedure described in: https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf

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
        print('Frrom ', start, 'to:', stop, 'is wave id:', wave_id)
        wave[start:stop, 0] = wave1[start:stop] if wave_id == 1 else wave2[start:stop]
        y_target[start:stop] = wave_id
        print('wave:\n', wave[start:stop, 0])
        print('id:\n', y_target[start:stop])
        start = stop
        wave_id = 2 if wave_id == 1 else 1
        print('New stop:', stop)
        print('New id:', wave_id)

    if show_plot:
        plot_sequence(y_sequence=wave)

    return wave, y_target


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
# Define model architecture
########################################################################################################################

class EchoStateNetwork:

    def __init__(self,
                 learning_rate=1.,
                 bias=True,
                 input_size=1,
                 reservoir_nodes=100,
                 nonlinearity=np.tanh,
                 reservoir_connectivity=0.8,  # How many percent of weights ought to be non-0
                 ridge_alpha=0.05
                 ):

        # Init some parameters
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.alpha = ridge_alpha
        self.learning_rate = learning_rate
        self.reservoir_nodes = reservoir_nodes

        # Compute network dimensions
        reservoir_weights = reservoir_nodes ** 2
        reservoir_nonzeros = int(np.ceil(reservoir_weights * reservoir_connectivity))

        w_in_dim = [reservoir_nodes, input_size + (1 if bias else 0)]
        w_dim = [reservoir_nodes, reservoir_nodes]

        # Initialize input- and reservoir weights
        self.w_in = np.random.rand(w_in_dim[0], w_in_dim[1])
        self.w = np.zeros(w_dim)

        # Reservoir's pseudo-random init state
        self.x_init = np.random.rand(reservoir_nodes)

        # Initialize reservoir weights
        nonzero_indices = np.random.randint(low=reservoir_nodes, size=(reservoir_nonzeros, 2))
        for index in nonzero_indices:
            self.w[index[0], index[1]] = random.random()

        # Placeholder init for ridge regression model
        self.ridge_model = None


    def get_reservoir_states(self, sequence):
        # Get input-output mappings
        x = self.x_init.copy()  # Reservoir state
        x_history = np.zeros([time_steps, self.reservoir_nodes])
        constant = np.array([1])

        # Calculate reservoir's state for each time step t
        for t in range(time_steps):
            # Add bias constant to training input for t'th time step or not
            if self.bias:
                input = np.concatenate([constant, sequence[t]], axis=0)
            else:
                input = np.array(sequence[t])

            # Compute components needed for updating reservoir
            in1 = np.sum(self.w_in * input, axis=1)
            in2 = np.sum(self.w * x, axis=1)

            # Combine terms and apply non-linearity
            update = self.nonlinearity(in1 + in2)

            # Update reservoir's state
            x = (1. - self.learning_rate) * x + self.learning_rate * update

            # Keep track of reservoir's states over course of processing input sequence
            x_history[t, :] = x

        return x_history


    def train(self, sequence, y_target):
        # Generate sequence of reservoir states as it iterates over train input sequence
        x_history = self.get_reservoir_states(sequence)

        # Train output weights
        self.ridge_model = Ridge(
            alpha=self.alpha,
            fit_intercept=True,
            copy_X=True
        ).fit(x_history, y_target)

        return x_history


    def predict(self, sequence):
        # Generate sequence of reservoir states as it iterates over train input sequence
        x_history = self.get_reservoir_states(sequence)

        # For each time step, predict the driving signal
        y_predicted = self.ridge_model.predict(x_history)

        return y_predicted


########################################################################################################################
# Generate training and testing data
########################################################################################################################

# Train-Hyperparameter
time_steps = 10000

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
    reservoir_nodes=100,
    reservoir_connectivity=0.3,
    ridge_alpha=0.00000000000000000000002
)
x_history = esn.train(wave, y_target)

plot_neural_activity(x_history, 100, num_disp=10)

########################################################################################################################
# Predict and plot on training data
########################################################################################################################

y_predicted = esn.predict(wave_test)
plot_sequences([wave, y_target, y_predicted], ['wave', 'y_target', 'y_predicted'], title='Train data')

########################################################################################################################
# Predict and plot on testing data
########################################################################################################################

y_predicted = esn.predict(wave_test)
plot_sequences([y_target_test, y_predicted], ['y_target_test', 'y_predicted'], title='Test data')

