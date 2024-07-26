import dearpygui.dearpygui as dpg
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dpg.create_context()


def solve_and_plot_linear_system(sender, app_data, user_data):
    matrix = []
    b_vector = []
    for r in range(row_num):
        row = []
        for c in range(col_num):
            value = dpg.get_value(f"r{r + 1}c{c + 1}")
            row.append(value)
        matrix.append(row)
        b_value = dpg.get_value(f"b{r + 1}")
        b_vector.append(b_value)

    A = np.array(matrix, dtype=float)
    b = np.array(b_vector, dtype=float)

    # Solve Ax = 0
    null_space = linalg.null_space(A)
    if null_space.size > 0:
        x_null = null_space[:, 0]  # Take the first basis vector of the null space
    else:
        x_null = "No non-trivial solution"

    # Solve Ax = b
    try:
        x_particular = linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x_particular = "No unique solution"

    dpg.set_value("result_null", f"Solution for Ax = 0: {x_null}")
    dpg.set_value("result_particular", f"Solution for Ax = b: {x_particular}")

    # Plot solutions
    fig = plt.figure(figsize=(12, 5))

    # Plot Ax = 0
    ax1 = fig.add_subplot(121, projection='3d')
    if isinstance(x_null, np.ndarray):
        t = np.linspace(-10, 10, 100)
        x = x_null[0] * t
        y = x_null[1] * t
        z = x_null[2] * t
        ax1.plot(x, y, z)
    ax1.set_title('Solution for Ax = 0')
    ax1.set_xlabel('x');
    ax1.set_ylabel('y');
    ax1.set_zlabel('z')

    # Plot Ax = b
    ax2 = fig.add_subplot(122, projection='3d')
    if isinstance(x_particular, np.ndarray):
        ax2.scatter(x_particular[0], x_particular[1], x_particular[2])
    ax2.set_title('Solution for Ax = b')
    ax2.set_xlabel('x');
    ax2.set_ylabel('y');
    ax2.set_zlabel('z')

    plt.tight_layout()
    plt.show()


def source_dtype():
    return 'float'


with dpg.window(tag='pri'):
    dpg.add_text('Enter 3x3 augmented matrix [A b] for solution retrieval where Ax = 0 and Ax = b')

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
        for i in range(col_num + 1):  # +1 for b column
            dpg.add_table_column(label=f'{"b" if i == col_num else f"c{i + 1}"}')
        for r in range(row_num):
            with dpg.table_row():
                for c in range(col_num):
                    dpg.add_input_float(tag=f"r{r + 1}c{c + 1}", label=f'r{r + 1}', source=source_dtype())
                dpg.add_input_float(tag=f"b{r + 1}", label=f'b{r + 1}', source=source_dtype())

    dpg.add_button(label="Solve and Plot", callback=solve_and_plot_linear_system)

    dpg.add_text("", tag="result_null")
    dpg.add_text("", tag="result_particular")

dpg.create_viewport(title='Linear algebra app (R^3)', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window('pri', True)
dpg.start_dearpygui()
dpg.destroy_context()

