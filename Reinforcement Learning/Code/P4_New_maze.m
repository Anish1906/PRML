% Starter code for project 3: Deep Learning
% PRML, CSE583/EE552
% TA: Shimian Zhang, Feb 2022
% TA: Addison Petro, Feb 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Main code for smaller maze problem.
%}

%% Clean up the workspace
clear all;
close all;
clc;

%% Make the maze
maze =  [2, 0, 0, 0, 0, 0 
        0, 1, 0, 0, 0, 0 
        0, 1, 0, 1, 1, 0 
        0, 0, 0, 0, 1, 0 
        0, 0, 1, 0, 0, 0 
        0, 0, 1, 0, 0, 0];
%duplicating the maze to reset maze after every iteration    
maze2 = [2, 0, 0, 0, 0, 0 
        0, 1, 0, 0, 0, 0 
        0, 1, 0, 1, 1, 0 
        0, 0, 0, 0, 1, 0 
        0, 0, 1, 0, 0, 0 
        0, 0, 1, 0, 0, 0];
current_location = [1,1];
end_location = [6,6];

action_keys = {'U', 'D', 'L', 'R'};
action_vals = {[-1, 0], [1, 0], [0, -1], [0, 1]};
actions = containers.Map(action_keys, action_vals);

% You will need to figure out which data types to use for allowed states
allowed_states = containers.Map;

%setting actions for corners
allowed_states(make_coordinates(1,1))= action_keys;
allowed_states(make_coordinates(1,1))=setdiff(allowed_states(make_coordinates(1,1)), {'U'});
allowed_states(make_coordinates(1,1))=setdiff(allowed_states(make_coordinates(1,1)), {'L'});
allowed_states(make_coordinates(1,6))= action_keys;
allowed_states(make_coordinates(1,6))=setdiff(allowed_states(make_coordinates(1,6)), {'R'});
allowed_states(make_coordinates(1,6))=setdiff(allowed_states(make_coordinates(1,6)), {'U'});
allowed_states(make_coordinates(6,1))= action_keys;
allowed_states(make_coordinates(6,1))=setdiff(allowed_states(make_coordinates(6,1)), {'L'});
allowed_states(make_coordinates(6,1))=setdiff(allowed_states(make_coordinates(6,1)), {'D'});
allowed_states(make_coordinates(6,6))= action_keys;
allowed_states(make_coordinates(6,6))=setdiff(allowed_states(make_coordinates(6,6)), {'R'});
allowed_states(make_coordinates(6,6))=setdiff(allowed_states(make_coordinates(6,6)), {'D'});

%setting actions for all other states
for i = 1
    for j = 2:1:5
        allowed_states(make_coordinates(i,j))= action_keys;
        allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'U'});
        if maze(i,j-1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'L'});
        end
        if maze(i,j+1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'R'});
        end
        if maze(i+1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'D'});
        end
    end
end

for i = 6
    for j = 2:1:5
        allowed_states(make_coordinates(i,j))= action_keys;
        allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'D'});
        if maze(i,j-1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'L'});
        end
        if maze(i,j+1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'R'});
        end
        if maze(i-1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'U'});
        end
    end
end

for i = 2:1:5
    for j = 1
        allowed_states(make_coordinates(i,j))= action_keys;
        allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'L'});
        if maze(i,j+1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'R'});
        end
        if maze(i-1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'U'});
        end
        if maze(i+1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'D'});
        end
    end
end

for i = 2:1:5
    for j = 6
        allowed_states(make_coordinates(i,j))= action_keys;
        allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'R'});
        if maze(i,j-1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'L'});
        end
        if maze(i-1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'U'});
        end
        if maze(i+1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'D'});
        end
    end
end

for i = 2:1:5
    for j = 2:1:5
        allowed_states(make_coordinates(i,j))= action_keys;
        if maze(i,j-1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'L'});
        end
        if maze(i,j+1) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'R'});
        end
        if maze(i-1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'U'});
        end
        if maze(i+1,j) == 1
            allowed_states(make_coordinates(i,j)) = setdiff(allowed_states(make_coordinates(i,j)), {'D'});
        end
    end
end

%% Initializing rewards
rng('default') % For reproducibility
a = -0.1;
b = -1;

%Q-table
for i = 1:1:6
    for j = 1:1:6
        if maze(i,j) == 1
            q_table(i,j) = -Inf;
        else 
            q_table(i,j) = (b-a).*rand(1,1) + a;
        end
    end
end

%rewards loop
for i = 1:1:6
    for j = 1:1:6
        if (i*j)< 36
            reward(i,j) = -1;
        else 
            reward(i,j) = 0;
        end
    end
end

reward(1,1) = 0;

%% 2d to 1d representation
oneD = containers.Map;
%This will help us with state-table
k = 1;
for i = 1:1:6
    for j = 1:1:6
        oneD(make_coordinates(i,j)) = k;
        k = k+1;
    end 
end

m = 1;
for i = 1:1:6
    for j = 1:1:6
        coord(i,j) = m;
        m = m+1;
    end 
end
%% Move through maze and print it
%Initialising parameters
tot = prod(current_location);
count = 0;
print_maze(maze);
random_factor = 0.25;
a = 0;
b = 1;
itr = 0;
target_reward = 0;

for itr = 1:1:3000
     %resetting parmeters every loop
    maze = maze2;
    current_location = [1,1];
    count = 0;
    state_table = [];
    state_table(1) = 1;
    tot = 0;
    while (tot ~= 36) && count<500
        %generating random value for exploration and exploitation
        r = (b-a).*rand(1,1) + a;
        if r < random_factor
            %exploration
            states = cell2mat(allowed_states(make_coordinates(current_location(1),current_location(2))));
            str = states(randi(length(states)));%randomly choosing a state
            move = values(actions, {str});
            old_location(1) = current_location(1);
            old_location(2) = current_location(2);
            current_location = current_location + cell2mat(move);
            maze(old_location(1),old_location(2)) = 0;
            maze(current_location(1),current_location(2)) = 2;
            tot = prod(current_location);
            count = count + 1;
        else
            %exploitation
            val = [];
            %comparing neighbouring expectations
            states = cell2mat(allowed_states(make_coordinates(current_location(1),current_location(2))));
            for i = 1:1:length(states)
                temp_state = current_location + cell2mat(values(actions, {states(i)}));
                val(i) = q_table(temp_state(1),temp_state(2));
            end
            max_val = max(val);
            index = find(val == max_val);
            str = states(index);
            move = values(actions, {str});
            old_location(1) = current_location(1);
            old_location(2) = current_location(2);
            current_location = current_location + cell2mat(move);
            maze(old_location(1),old_location(2)) = 0;
            maze(current_location(1),current_location(2)) = 2;
            tot = prod(current_location);
            count = count + 1;
        end
        %make state table
        state_table(count) = oneD(make_coordinates(current_location(1),current_location(2)));
    end
    moves(itr) = count;
    %depreciating the random factor
    if rem(itr,250)==0
        random_factor = random_factor - 1e-4;
    end
    target_reward = 0;
    %updating current expectation based on state table
    for i = length(state_table):-1:1
        [row,col] = find(coord == state_table(i));
        q_table(row,col) = q_table(row,col) + 0.002*(target_reward - q_table(row,col));
        target_reward = target_reward + reward(row,col);
    end
    %printing heat map
    if rem(itr,250)==0
        mapping = zeros(6);
        for i = 1:1:length(state_table)
            r1 = rem(state_table(i),6);
            q1 = (state_table(i) - r1)/6;
            if r1 == 0
                r1 = 6;
            else
                q1 = q1 +1;
            end
            mapping(q1,r1) = mapping(q1,r1) + 1; 
        end
        figure, h = heatmap(mapping,'Colormap', parula);
    end
end
figure, plot(moves);%plotting number of moves taken

print_maze(maze);
fprintf("Congratulations! You made it to the end of the maze!\n");