import dearpygui.dearpygui as dpg
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initializing DearPyGui context
dpg.create_context()


def solve_and_plot_linear_system(sender, app_data, user_data):
    try:
        # Initializing empty lists for matrix and b vector
        matrix = []
        b_vector = []

        # Collecting input values from the GUI
        for r in range(row_num):
            row = []
            for c in range(col_num):
                value = dpg.get_value(f"r{r + 1}c{c + 1}")
                row.append(value)
            matrix.append(row)
            b_value = dpg.get_value(f"b{r + 1}")
            b_vector.append(b_value)

        # Checking if matrix is 3x3
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("Matrix must be 3x3")

        # Converting lists to numpy arrays
        A = np.array(matrix, dtype=float)
        b = np.array(b_vector, dtype=float)

        # Calculating and display additional information
        rank = np.linalg.matrix_rank(A)
        det = np.linalg.det(A)
        dpg.set_value("matrix_info", f"Rank: {rank}, Determinant: {det:.2f}")

        # Solving Ax = 0 (homogeneous system)
        null_space = linalg.null_space(A)
        if null_space.size > 0:
            x_null = null_space[:, 0]  # Take the first basis vector of the null space
        else:
            x_null = "No non-trivial solution"

        # Solving Ax = b (non-homogeneous system)
        try:
            x_particular = linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x_particular = "No unique solution"

        # Displaying results in the GUI
        dpg.set_value("result_null", f"Solution for Ax = 0: {x_null}")
        dpg.set_value("result_particular", f"Solution for Ax = b: {x_particular}")

        # Plotting solutions
        fig = plt.figure(figsize=(12, 5))

        # Plotting Ax = 0
        ax1 = fig.add_subplot(121, projection='3d')
        if isinstance(x_null, np.ndarray):
            t = np.linspace(-10, 10, 100)
            x = x_null[0] * t
            y = x_null[1] * t
            z = x_null[2] * t
            ax1.plot(x, y, z, label='Null space')
        ax1.set_title('Solution for Ax = 0')
        ax1.set_xlabel('x');
        ax1.set_ylabel('y');
        ax1.set_zlabel('z')
        ax1.legend()
        ax1.grid(True)

        # Plotting Ax = b
        ax2 = fig.add_subplot(122, projection='3d')
        if isinstance(x_particular, np.ndarray):
            ax2.scatter(x_particular[0], x_particular[1], x_particular[2], color='red', s=100,
                        label='Particular solution')

            # If x_null is also a solution, plot the line of all solutions
            if isinstance(x_null, np.ndarray):
                t = np.linspace(-10, 10, 100)
                x = x_particular[0] + x_null[0] * t
                y = x_particular[1] + x_null[1] * t
                z = x_particular[2] + x_null[2] * t
                ax2.plot(x, y, z, label='All solutions')

        ax2.set_title('Solution for Ax = b')
        ax2.set_xlabel('x');
        ax2.set_ylabel('y');
        ax2.set_zlabel('z')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    except ValueError as e:
        dpg.set_value("error_message", f"Error: {str(e)}")
    except Exception as e:
        dpg.set_value("error_message", f"An unexpected error occurred: {str(e)}")


# The function to set the data type for input fields
def source_dtype():
    return 'float'


# The function to clear all input fields
def clear_inputs(sender, app_data, user_data):
    for r in range(row_num):
        for c in range(col_num):
            dpg.set_value(f"r{r + 1}c{c + 1}", 0.0)
        dpg.set_value(f"b{r + 1}", 0.0)
    dpg.set_value("result_null", "")
    dpg.set_value("result_particular", "")
    dpg.set_value("matrix_info", "")
    dpg.set_value("error_message", "")


# Creating the main window
with dpg.window(tag='pri'):
    dpg.add_text('Enter 3x3 augmented matrix [A b] for solution retrieval where Ax = 0 and Ax = b')

    # Setting up the table for matrix input
    col_num = 3
    row_num = 3
    with dpg.table(
            header_row=True,
            row_background=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
    ):
        # Adding column headers
        for i in range(col_num + 1):  # +1 for b column
            dpg.add_table_column(label=f'{"b" if i == col_num else f"c{i + 1}"}')

        # Adding input fields for matrix A and vector b
        for r in range(row_num):
            with dpg.table_row():
                for c in range(col_num):
                    dpg.add_input_float(tag=f"r{r + 1}c{c + 1}", label=f'r{r + 1}', source=source_dtype())
                dpg.add_input_float(tag=f"b{r + 1}", label=f'b{r + 1}', source=source_dtype())

    # Adding solve and clear buttons
    with dpg.group(horizontal=True):
        dpg.add_button(label="Solve and Plot", callback=solve_and_plot_linear_system)
        dpg.add_button(label="Clear", callback=clear_inputs)

    # Adding text areas for displaying results and additional information
    dpg.add_text("", tag="matrix_info")
    dpg.add_text("", tag="result_null")
    dpg.add_text("", tag="result_particular")
    dpg.add_text("", tag="error_message")

# Creating and configure the viewport
dpg.create_viewport(title='Linear algebra app (R^3)', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('pri', True)

# Starting the DearPyGui event loop
dpg.start_dearpygui()

# Cleaning up
dpg.destroy_context()
