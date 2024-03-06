def unique_paths(x, y):
    dp = [[0] * y for _ in range(x)]

    for i in range(x):
        dp[i][0] = 1
    for j in range(y):
        dp[0][j] = 1

    # Fill the rest of the grid
    for i in range(1, x):
        for j in range(1, y):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]


    return dp[x-1][y-1]

x, y = map(int, input().split())
# Output
print(unique_paths(x, y))
