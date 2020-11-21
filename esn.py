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

def get_square_wave(time_points=10000, transitions=100, show_plot=False):
    # Generate time steps on which square wave alternates
    transition_points = sorted(random.sample(range(time_points), k=transitions))

    # Select starting value for pseudo-random square wave
    value = random.sample([0, 1], k=1)

    # Construct square wave signal
    wave = np.zeros([time_points])
    start = 0
    for stop in transition_points:
        wave[start:stop] = value
        start = stop
        value = 0 if value == 1 else 1

    if show_plot:
        time = np.arange(0, time_points, 1)
        plot.plot(time, wave)
        plot.title('Square wave')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

    return wave


def get_sin_wave(time_points=10000, div_factor=25, show_plot=False):
    time = np.arange(0, time_points, 1)
    wave = np.sin(time/div_factor)  # Div factor to vary frequency

    if show_plot:
        # Plot a sine wave using time and amplitude obtained for the sine wave
        plot.plot(time, wave)
        plot.title('Sine wave')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

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
        time = np.arange(0, time_points, 1)
        plot.plot(time, wave)
        plot.title('Combined wave')
        plot.xlabel('Time')
        plot.ylabel('Amplitude')
        plot.grid(True, which='both')
        plot.axhline(y=0, color='k')
        plot.show()

    return wave, y_target


# Train-Hyperparameter
time_steps = 10000

########################################################################################################################
# Generate training and testing data
########################################################################################################################

# Get training input
square_wave = get_square_wave(time_points=time_steps, show_plot=False)
sine_wave = get_sin_wave(time_points=time_steps, show_plot=False)
wave, y_target = combine_waves(square_wave, sine_wave, show_plot=True)

# Get testing input
square_wave = get_square_wave(time_points=time_steps, show_plot=False)
sine_wave = get_sin_wave(time_points=time_steps, show_plot=False)
wave_test, y_target_test = combine_waves(square_wave, sine_wave, show_plot=True)


# Define model architecture
leaning_rate            = 1.
bias                    = True
input_size              = 1
output_size             = 1
reservoir_nodes         = 3
nonlinearity            = np.tanh
reservoir_connectivity  = 0.2        # How many percent of weights ought to be non-0
reservoir_weights       = reservoir_nodes ** 2
reservoir_nonzeros      = int(np.ceil(reservoir_weights * reservoir_connectivity))

w_in_dim                = [reservoir_nodes, input_size + (1 if bias else 0)]
w_dim                   = [reservoir_nodes, reservoir_nodes]
# w_out_dim             = [output_size, esn_nodes]

w_in                    = np.random.rand(w_in_dim[0], w_in_dim[1])
w                       = np.zeros(w_dim)
w_out                   = None

# Reservoir's init state
x_init = np.random.rand(reservoir_nodes)


# Initialize reservoir weights
nonzero_indices = np.random.randint(low=reservoir_nodes, size=(reservoir_nonzeros, 2))
for index in nonzero_indices:
    w[index[0], index[1]] = random.random()


# Get input-output mappings
x = x_init.copy()  # Reservoir state
x_history = np.zeros([time_steps, reservoir_nodes])
constant = np.array([1])

for t in range(time_steps):
    if bias:
        input = np.concatenate([constant, wave[t]], axis=0)
    else:
        input = np.array(wave[t])

    # Compute components needed for updating reservoir
    in1 = np.sum(w_in * input, axis=1)
    in2 = np.sum(w * x, axis=1)

    # Combine terms and apply non-linearity
    update = nonlinearity(in1 + in2)

    # Update reservoir's state
    x = (1. - leaning_rate) * x + leaning_rate * update

    # Keep track of reservoir's states over course of processing input sequence
    x_history[t, :] = x


# Train output weights
alpha = 0.5
model = Ridge(alpha=alpha).fit(x_history, y_target)

prediction = model.predict(x)

#class EchoStateNetwork():

#    def __init__(self):
#        self.w_in =
