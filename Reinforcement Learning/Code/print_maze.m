function print_maze(maze)

for idx = 1:size(maze, 2) + 2
    fprintf("█");
end
fprintf("\n");
for row = 1:size(maze, 1)
    fprintf("█");
    for col = 1:size(maze, 2)
        if (maze(row, col) == 0)
            fprintf(" ");
        elseif (maze(row, col) == 1)
            fprintf("█");
        elseif (maze(row, col) == 2)
            fprintf("O");
        end
    end
    fprintf("█\n");
end
for idx = 1:size(maze, 2) + 2
    fprintf("█");
end
fprintf("\n");

end

