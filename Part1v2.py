import matplotlib.pyplot as plt

def get_perceptron_params():
    """Get perceptron weights and bias from user input."""
    return (
        float(input("enter w1: ")),
        float(input("enter w2: ")),
        float(input("enter bias: "))
    )

def activation_function(z):
    """Binary step activation function."""
    return 1 if z > 0 else -1

def test_single_perceptron(x, y):
    """Test a single perceptron with given inputs."""
    w1, w2, s = get_perceptron_params()
    # Calculate outputs for each input pair
    outputs = [activation_function(w1 * xi + w2 * yi - s) for xi, yi in zip(x, y)]
    return outputs, w1, w2, s

def plot_decision_boundary():
    """Plot the decision boundary and data points."""
    # Input data for logical operations
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    
    classes, w1, w2, s = test_single_perceptron(x, y)
    
    # Calculate decision boundary line
    x_boundary = [-4, 4]
    y_boundary = [(s - (w1 * x)) / w2 for x in x_boundary]
    
    # Plot data points with appropriate colors
    for i in range(len(x)):
        color = 'orange' if classes[i] == 1 else 'blue'
        plt.scatter(x[i], y[i], color=color)
    
    # Plot decision boundary and setup plot
    plt.plot(x_boundary, y_boundary, 'r-', label='Decision Boundary')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Decision Boundary using Single Perceptron')
    plt.show()
    print("Classes:", classes)

def multi_layer():
    """Implement a multi-layer perceptron network."""
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    
    print("Layer 1:")
    y1, *_ = test_single_perceptron(x, y)
    print("Layer 2:")
    y2, *_ = test_single_perceptron(x, y)
    print("Layer 3:")
    y3, *_ = test_single_perceptron(y1, y2)
    
    print(f"y1 = {y1}\ny2 = {y2}\nY = {y3}")

def perceptron_learning():
    """Implement the perceptron learning algorithm."""
    num_inputs = int(input("Enter number of inputs: "))
    num_dimensions = int(input("Enter number of dimensions: "))
    learning_rate = 0.2
    
    # Initialize weights with zeros
    weights = [0] * (num_dimensions + 1)
    inputs = []
    targets = []
    
    # Get input data and target classes
    for i in range(num_inputs):
        features = [float(input(f"Enter d{j+1}: ")) for j in range(num_dimensions)]
        inputs.append(features)
        targets.append(int(input("Enter the class: ")))
    
    # Training loop - single iteration
    for i in range(num_inputs):
        # Calculate prediction
        y_val = weights[0]  # bias term
        for j in range(num_dimensions):
            y_val += weights[j + 1] * inputs[i][j]
        prediction = activation_function(y_val)
        
        # Update weights if prediction is wrong
        if prediction != targets[i]:
            error = targets[i] - prediction
            weights[0] += learning_rate * error  # Update bias
            for j in range(num_dimensions):
                weights[j + 1] += learning_rate * error * inputs[i][j]
    
    print(f"Final weights = {weights}")

# The Answers:
plot_decision_boundary()           # Question 1
# multi_layer()                    # Question 2
# perceptron_learning()           # Question 3