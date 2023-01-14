import matplotlib.pyplot as plt
import tensorflow as tf

class PINN:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_shape=[2], activation="relu"),
            tf.keras.layers.Dense(2)
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    def loss(self, x, y):
        dy_dt = self.model(y)
        eq1 = dy_dt[:,0] - x * y[:,1]
        eq2 = dy_dt[:,1] + x**2
        return tf.reduce_mean(tf.square(eq1) + tf.square(eq2))
    
    def train(self, x, y, epochs=1000):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss_value = self.loss(x, y)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
    def predict(self, x):
        y_test = tf.stack([x, x], axis=-1)
        dy_dt_pred = self.model(y_test)
        return dy_dt_pred

# Create an instance of the PINN class
pinn = PINN()

# Define the input
x = tf.linspace(0., 10., 100)
y = tf.stack([x, x], axis=-1)

# Train the model
pinn.train(x, y)

# Use the trained network to predict the solution for new input
dy_dt_pred = pinn.predict(x)

# Plot the solution
plt.plot(y_test[:,0], dy_dt_pred[:,0], label='dx/dt')
plt.plot(y_test[:,1], dy_dt_pred[:,1], label='dy/dt')
plt.legend()
plt.xlabel('x,y')
plt.ylabel('dx/dt, dy/dt')
plt.show()
