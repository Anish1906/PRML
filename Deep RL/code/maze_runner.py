import time
import numpy as np
import matplotlib.pyplot as plt

# Function for printing the maze if you need it
def print_maze(maze):
    print("██████████")
    for row in maze:
        print("█", end='')
        for col in row:
            if (col == 0):
                print(' ', end='')
            elif (col == 1):
                print('█', end ='')
            elif (col == 2):
                print('O', end='')
        print("█")
    print("██████████")


# Return a clean copy of the maze
def get_new_maze():
    maze = [[2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0]]
    return maze


# Define actions
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
action_arr = ['U', 'D', 'L', 'R']


# Update the state of the maze and agent
def do_move(maze, current_location, end_location, move_idx):
    new_maze = maze
    y, x = current_location
    direction = actions[action_arr[move_idx]]
    new_x = x + direction[1]
    new_y = y + direction[0]
    new_maze[y][x] = 0
    new_maze[new_y][new_x] = 2
    new_location = (new_y, new_x)
    # Get reward
    if (new_location == end_location):
        current_reward = 0
    else:
        current_reward = -1
    return (new_maze, new_location, current_reward)


# Given an input vector, return the index of the 1 for one-hot encoding   
def get_position_index(loc, width):
    return loc[0] * width + loc[1]

# Try an iteration of the maze in exploit only mode
def try_maze(max_moves):
    num_moves = 0
    width = 8
    start_location = (0, 0)
    end_location = (7, 7)
    current_location = start_location
    x = []
    y = []
    x.append(current_location[1] + 0.5)
    y.append(current_location[0] + 0.5)
    maze = get_new_maze()
    while (current_location != end_location and num_moves < max_moves):
        # Choose next action
        max_expected = -1e12 # Some really big negative number
        for idx in range(4):
            #if (allowed_states[get_position_index(current_location, width)][idx] == 1):
                #new_state = tuple([sum(val) for val in zip(current_location, actions[action_arr[idx]])])
            if (current_expectations[get_position_index(current_location, width),idx] >= max_expected):
                max_expected = current_expectations[get_position_index(current_location, width),idx]
                next_move = idx
        # Update the state
        if (allowed_states[get_position_index(current_location, width)][next_move] == 0):
            maze = maze
            current_location=current_location
            current_reward = -1
        else:
            maze, current_location, current_reward = do_move(maze, current_location, end_location, next_move)
        x.append(current_location[1] + 0.5)
        y.append(current_location[0] + 0.5)
        num_moves += 1
    return num_moves, x, y


# Set up important variables
# Maze info
start_location = (0, 0)
end_location = (7, 7)
width = 8
height = 8
# Training parameters
alpha = 0.1
random_factor = 0.475
drop_rate = 0.0001
max_iterations = 5000
max_moves = 1000
test_interval = 50 # Once you change to deep learning, set this to 5
# Some data
allowed_states = np.zeros((width * height, 4))
current_expectations = np.zeros((width * height, 4))


# Construct allowed states
maze = get_new_maze()
for y in range(height):
    for x in range(width):
        pos_idx = get_position_index((y, x), width)
        # Skip wall tiles
        if (maze[y][x] == 1):
            continue

        for action_idx in range(len(action_arr)):
            action = action_arr[action_idx]
            new_x = x + actions[action][1]
            new_y = y + actions[action][0]
            if (new_x < 0 or new_x > width-1 or new_y < 0 or new_y > height-1):
                continue
            if maze[new_y][new_x] == 0 or maze[new_y][new_x] == 2:
                allowed_states[pos_idx][action_idx] = 1

# Initialize rewards
for idx in range(width * height):
    for j in range(4):
        current_expectations[idx,j] = np.random.uniform (-1.0, -0.1)


# Set up everything for rendering
fig = plt.figure(1)
fig.show()
fig.canvas.draw()
x = []
y = []

# Some variables for stats
train_move_counts = []
test_move_counts = []
consecutive_good_win = 0
# The training happens here
for iteration in range (max_iterations):
    if ((iteration + 1) % 50 == 0):
        print("{}".format(iteration + 1))
    # Reset important variables
    moves = 0
    current_location = start_location
    state_history = []
    maze = get_new_maze()
    # Do a run through the maze
    while (current_location != end_location):
        # Choose next action
        next_move = None
        # Explore
        if (np.random.random() < random_factor):
            temp_vec = []
            for idx in range(4):
                #if (allowed_states[get_position_index(current_location, width)][idx] == 1):
                temp_vec.append(idx)
            next_move = np.random.choice(temp_vec)
        # Exploit
        else:
            max_expected = -1e12 # Some really big negative number
            for idx in range(4):
                #if (allowed_states[get_position_index(current_location, width)][idx] == 1):
                    #new_state = tuple([sum(val) for val in zip(current_location, actions[action_arr[idx]])])
                if (current_expectations[get_position_index(current_location, width),idx] >= max_expected):
                    max_expected = current_expectations[get_position_index(current_location, width), idx]
                    next_move = idx
        # Update the state & get reward
        previous_location = current_location
        if (allowed_states[get_position_index(current_location, width)][next_move] == 0):
            maze = maze
            current_location=current_location
            current_reward = -1
        else:
            maze, current_location, current_reward = do_move(maze, current_location, end_location, next_move)
        moves += 1
        # Update state history
        state_history.append((previous_location, next_move, current_reward, current_location))
        # If agent takes too long, just end the run
        if (moves > max_moves):
            current_location = end_location
            
    # Do the learning
    target_reward = 0
    for previous,move, reward, current in reversed(state_history):
        previous_idx = get_position_index(previous, width)
        current_expectations[previous_idx, move] = current_expectations[previous_idx, move] + alpha * (target_reward - current_expectations[previous_idx, move])
        target_reward += reward
    random_factor -= 1e-4
    # Store number of moves for plotting
    train_move_counts.append(moves)

    # Test the model
    if ((iteration+1) % test_interval == 0):
        test_val, test_x, test_y = try_maze(max_moves)
        if test_val == max_moves:
            print("TEST FAIL")
            consecutive_good_win = 0
        else:
            print("TEST WIN ({} moves)".format(test_val))
            if (test_val < 25):
                consecutive_good_win += 1
            else:
                consecutive_good_win = 0

        # Store number of moves for plotting
        test_move_counts.append(test_val)

    # Every 250 iterations, visualize the paths of the agent
    for state, _, _, _ in state_history:
        x.append(state[1] + 0.5)
        y.append(state[0] + 0.5)
    # These will be needed when we are storing previous state instead of current state
    '''
    if (len(state_history) < 1000):
        x.append(7.5)
        y.append(7.5)
    '''
    if ((iteration + 1) % 250 == 0):
        xedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        yedges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        heatmap, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        extent = [0, 8, 0, 8]
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel("Agent x-coordinate", fontsize=20)
        plt.ylabel("Agent y-coordinate", fontsize=20)
        plt.gca().invert_yaxis()
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        cb = plt.colorbar(ticks=[0, heatmap.max()])
        cb.ax.set_yticklabels(['Never', 'Often'])
        cb.ax.tick_params(labelsize=16)
        fig.canvas.draw()
        time.sleep(0.25)
        plt.pause(1) # an alternate way to pause for a bit
        # Reset the storage
        x = []
        y = []

    # If we've done well enough three times in a row, stop training
    # For normal Q-Learning, this will get hit almost immediately, uncomment when you do deep Q-learning
    '''
    if (consecutive_good_win >= 3):
        break
    '''

fig2 = plt.figure(2)
plt.semilogy(train_move_counts, 'b', linewidth=0.5)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("Number of moves taken", fontsize=14)
plt.title("Training moves", fontsize=16)
plt.show()

fig3 = plt.figure(3)
plt.semilogy(test_move_counts, 'b', linewidth=0.5)
plt.xlabel("Iteration Number", fontsize=14)
plt.ylabel("Number of moves taken", fontsize=14)
plt.title("Testing moves", fontsize=16)
plt.show()