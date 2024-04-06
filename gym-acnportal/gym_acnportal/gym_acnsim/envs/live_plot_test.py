# import random
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Create initial data values of x and y
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
#
# # Create the plot with initial data
# plt.figure(figsize=(10, 6))
# plt.title("Manual Live Plot Example")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# line, = plt.plot(x, y)
#
# # Function to update the plot
# def update_plot():
#     # Create new Y values
#     new_y = np.sin(x - random.uniform(1, 5))
#
#     newy2 = np.sin(x**2)
#     choic = np.random.choice([0,1])
#     # Update data values
#     if choic == 0:
#         line.set_ydata(new_y)
#     else:
#         line.set_ydata(newy2)
#
#     # Draw the updated plot
#     plt.draw()
#     plt.pause(0.01)  # Pause to update the plot
#
# # Show the initial plot
# plt.show()
#
# # Wait for user input to update the plot
# while True:
#     user_input = input("Press Enter to update the plot (or 'q' to quit): ")
#     if user_input.lower() == 'q':
#         break
#     else:
#         update_plot()
#
# plt.close()  # Close the plot when finished
