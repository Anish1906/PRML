% Starter code for project 3: Deep Learning
% PRML, CSE583/EE552
% TA: Shimian Zhang, Feb 2022
% TA: Addison Petro, Feb 2022

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:
    PSU Email ID:
    Description: Main code for manual maze problem.
%}

%% Clean up the workspace
clear all;
close all;
clc;

%% Make the maze
maze = [2, 0, 0, 0, 0, 0, 0, 0
        0, 0, 0, 0, 0, 0, 0, 0
        0, 1, 0, 0, 0, 0, 1, 0
        0, 0, 0, 0, 1, 0, 0, 0
        0, 0, 0, 0, 1, 0, 0, 0
        0, 0, 1, 0, 0, 1, 0, 0
        0, 0, 0, 0, 0, 0, 1, 1
        0, 0, 0, 0, 1, 0, 0, 0];
current_location = [1,1];
end_location = [8, 8];

action_keys = {'U', 'D', 'L', 'R'};
action_vals = {[-1, 0], [1, 0], [0, -1], [0, 1]};
actions = containers.Map(action_keys, action_vals);

% You will need to figure out which data types to use for allowed states
allowed_states = containers.Map;

%setting actions for corners
allowed_states(make_coordinates(1,1))= {};
allowed_states(make_coordinates(1,1))=[allowed_states(make_coordinates(1,1));'R'];
allowed_states(make_coordinates(1,1))=[allowed_states(make_coordinates(1,1));'D'];
allowed_states(make_coordinates(1,8))= {};
allowed_states(make_coordinates(1,8))=[allowed_states(make_coordinates(1,8));'L'];
allowed_states(make_coordinates(1,8))=[allowed_states(make_coordinates(1,8));'D'];
allowed_states(make_coordinates(8,1))= {};
allowed_states(make_coordinates(8,1))=[allowed_states(make_coordinates(8,1));'R'];
allowed_states(make_coordinates(8,1))=[allowed_states(make_coordinates(8,1));'U'];
%setting actions for all other states
for i = 1
    for j = 2:1:7
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

for i = 8
    for j = 2:1:7
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

for i = 2:1:7
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

for i = 2:1:7
    for j = 8
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

for i = 2:1:7
    for j = 2:1:7
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

%% Move through maze and print it
%Initialising parameters
tot = prod(current_location);
count = 1;
print_maze(maze);

while (tot ~= 64)
    fprintf('Allowed moves:%s \n',string(allowed_states(make_coordinates(current_location(1),current_location(2))))');
    states = cell2mat(allowed_states(make_coordinates(current_location(1),current_location(2))));
    str = input("Next move?","s");%input from user
    if ismember(str, states) == 0
        fprintf('Invalid move. Try again.\n')%Checking if input is valid move
    else
        move = values(actions, {str});%If input is valid, robot moves
        old_location = current_location;
        current_location = current_location + cell2mat(move);
        maze(old_location(1),old_location(2)) = 0;
        maze(current_location(1),current_location(2)) = 2;
        tot = prod(current_location);
        print_maze(maze);
        fprintf('Valid moves made: %d \n', count);
        count = count + 1;
    end
end

print_maze(maze);
fprintf("Congratulations! You made it to the end of the maze!\n");