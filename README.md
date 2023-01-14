## Physically Informed MEural Network (PINN)

Here's the complete example of how you could use TensorFlow to solve the system of differential equations $$\frac{dx}{dt} = x*y$$ and $$\frac{dy}{dt} = x^2$$ with a PINN using classes and advanced features of Python.

## Description
In this example, we defined a class named PINN which contains the neural network, optimizer, and the train and predict methods. The __init__ method is called when a new instance of the class is created and it sets the architecture of the neural network and the optimizer. The loss method calculates the mean squared error between the predicted equations and zero. The train method trains the neural network by minimizing the loss function using the Adam optimizer. The predict method uses the trained network to predict the solution for new input.

We created an instance of the PINN class, defined the input, trained the model, and used the trained network to predict the solution for new input.

After you've used the trained network to predict the solution for new input, you can use the predicted dx/dt and dy/dt values to plot the solution using Matplotlib library.

## Result
<img src="https://github.com/Dartrisen/pinn/blob/main/result.png" width="50%" height="50%">
