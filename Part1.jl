#import Pkg; Pkg.add("Plots")

using Plots  # For plotting functionality

"""
    test_single_perceptron(x::Vector{Float64}, y::Vector{Float64})

Tests a single perceptron with user-input weights and bias.
Returns the classified outputs and the weights/bias used.
"""
function test_single_perceptron(x::Vector{Float64}, y::Vector{Float64})
    # Get user input for weights and bias
    println("Enter w1:")
    w1 = nothing
    while w1 === nothing
        try
            w1 = parse(Float64, readline())
        catch
            println("Invalid input. Please enter a number for w1:")
        end
    end
    
    println("Enter w2:")
    w2 = nothing
    while w2 === nothing
        try
            w2 = parse(Float64, readline())
        catch
            println("Invalid input. Please enter a number for w2:")
        end
    end
    
    println("Enter bias:")
    s = nothing
    while s === nothing
        try
            s = parse(Float64, readline())
        catch
            println("Invalid input. Please enter a number for bias:")
        end
    end
    
    # Initialize array to store classifications
    classes = Float64[]
    
    # Classify each input pair
    for i in 1:length(x)
        # Calculate weighted sum and apply activation function
        r = (w1 * x[i] + w2 * y[i]) - s
        push!(classes, activ_fun(r))
    end
    
    return classes, w1, w2, s
end

"""
    activ_fun(z::Float64)

Activation function (step function) that returns 1 if input > 0, -1 otherwise.
"""
function activ_fun(z::Float64)
    return z > 0 ? 1.0 : -1.0
end

"""
    decision_boundary()

Plots the decision boundary and data points for binary classification.
"""
function decision_boundary()
    # Define input data points
    x = [0.0, 0.0, 1.0, 1.0]
    y = [0.0, 1.0, 0.0, 1.0]
    
    # Get classifications and weights
    classes, w1, w2, s = test_single_perceptron(x, y)
    
    # Calculate decision boundary line points
    x_val = [-4.0, 4.0]
    y_val = [(s - (w1 * x_val[1])) / w2, (s - (w1 * x_val[2])) / w2]
    
    # Create new plot
    plt = plot(
        xlabel="X-axis",
        ylabel="Y-axis",
        title="Decision Boundary using Single Perceptron",
        grid=true,
        xlims=(-1, 2),
        ylims=(-1, 2)
    )
    
    # Plot data points with different colors based on class
    for i in 1:length(x)
        color = classes[i] == 1 ? :orange : :blue
        label = classes[i] == 1 ? "Class 1" : "Class 2"
        # Only add label for first occurrence of each class
        if i == 1 || (i > 1 && classes[i] != classes[i-1])
            scatter!([x[i]], [y[i]], color=color, label=label)
        else
            scatter!([x[i]], [y[i]], color=color, label=nothing)
        end
    end
    
    # Plot decision boundary
    plot!(x_val, y_val, color=:red, label="Decision Boundary")
    
    # Add axes
    hline!([0], color=:black, linewidth=1.5, label=nothing)
    vline!([0], color=:black, linewidth=1.5, label=nothing)
    
    display(plt)
    println("Classes: ", classes)
end

"""
    multi_layer()

Implements a multi-layer perceptron for XOR problem.
"""
function multi_layer()
    # Input data
    x = [0.0, 1.0, 0.0, 1.0]
    y = [0.0, 0.0, 1.0, 1.0]
    
    # First layer
    println("Layer 1:")
    classes1, _, _, _ = test_single_perceptron(x, y)
    
    # Second layer
    println("Layer 2:")
    classes2, _, _, _ = test_single_perceptron(x, y)
    
    # Output layer
    println("Layer 3:")
    classes3, _, _, _ = test_single_perceptron(classes1, classes2)
    
    # Print results
    println("y1 = ", classes1)
    println("y2 = ", classes2)
    println("Y = ", classes3)
end

"""
    get_valid_input(prompt::String, type::Type)

Helper function to get valid numerical input from user.
"""
function get_valid_input(prompt::String, type::Type)
    while true
        println(prompt)
        try
            return parse(type, readline())
        catch
            println("Invalid input. Please enter a valid number.")
        end
    end
end

"""
    perceptron_learning()

Implements the perceptron learning algorithm for multi-dimensional data.
"""
function perceptron_learning()
    # Get problem dimensions from user
    num_inputs = get_valid_input("Enter number of inputs:", Int)
    num_dimensions = get_valid_input("Enter number of dimensions:", Int)
    
    # Initialize parameters
    rate = 0.2  # Learning rate
    bias = 1.0  # Bias term
    w = zeros(num_dimensions + 1)  # Weights vector (including bias weight)
    inputs = Vector{Vector{Float64}}()  # Input patterns
    classes = Vector{Float64}()  # Desired outputs
    num_iteration = 1  # Number of training iterations
    
    # Get training data from user
    for i in 1:num_inputs
        temp = Float64[]
        for j in 1:num_dimensions
            value = get_valid_input("Enter d$j:", Float64)
            push!(temp, value)
        end
        push!(inputs, temp)
        class = get_valid_input("Enter the class:", Float64)
        push!(classes, class)
    end
    
    # Training loop
    for iteration in 1:num_iteration
        for i in 1:num_inputs
            # Calculate weighted sum
            y_val = w[1] * bias
            for j in 1:num_dimensions
                y_val += w[j + 1] * inputs[i][j]
            end
            
            # Get prediction
            predicted = activ_fun(y_val)
            
            # Update weights if prediction is wrong
            if predicted != classes[i]
                error = classes[i] - predicted
                # Update input weights
                for j in 1:num_dimensions
                    w[j + 1] += rate * error * inputs[i][j]
                end
                # Update bias weight
                w[1] += rate * error * bias
            end
        end
    end
    
    println("Final weights = ", w)
end

# Main execution
# Uncomment the function you want to run
decision_boundary()           # Question 1
# multi_layer()              # Question 2
# perceptron_learning()      # Question 3