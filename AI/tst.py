import matplotlib.pyplot as plt
import numpy as np

def plot_sine_wave():
    # Generate x values
    x = np.linspace(0, 2 * np.pi, 1000)
    # Compute y values (sine of x)
    y = np.sin(x)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Sine Wave')
    
    # Add title and labels
    plt.title('Sine Wave')
    plt.xlabel('x values')
    plt.ylabel('sin(x)')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.show()

# Call the function to plot the sine wave
plot_sine_wave()