 Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')

# Add labels and title
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')

# Show the plot
plt.show()