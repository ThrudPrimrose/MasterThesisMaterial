import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

def draw_grid():
    # Define the number of rows and columns you want
    num_rows = 4
    num_cols = 2

    fig, ax = plt.subplots(figsize=(num_cols,num_rows))

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Draw grid
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot extra lines to form the complete grid
    for x in range(num_cols + 1):
        if x <= 1:
            color = "orange"
        else:
            color = "k"
        if x==0:
            linewidth=2
        else:
            linewidth=1
        ax.plot([x, x], [0, num_rows], color=color, linewidth=linewidth)

    for y in range(num_rows + 1):
        if y==0 or y==num_rows:
            linewidth=1.5
        else:
            linewidth=1
        ax.plot([0, num_cols//2], [y, y], color='orange', linewidth=linewidth)
        ax.plot([num_cols//2, num_cols], [y, y], color='k', linewidth=linewidth)

    rect = patches.Rectangle((0, 0), 1, 4, facecolor='lightblue', alpha=0.5) # (2, 2) is the bottom-left corner of the rectangle and 3x3 is its size
    ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.gca().invert_yaxis()  # Invert y axis so the (0,0) is at the top-left
    plt.savefig("a.pdf")

draw_grid()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_tensor_grid():
    # Define the number of rows and columns you want
    num_rows = 4
    num_cols = 2
    num_z = 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_zlim(0, num_z)

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Draw grid
    #ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

    # Remove axis labels
    #ax.set_xticks([])
    #ax.set_yticks([])

    # Plot extra lines to form the complete grid
    for x in range(num_cols + 1):
        ax.plot([x, x], [0, 0], [0,num_z], color="red", linewidth=1)
    for y in range(num_rows + 1):
        ax.plot([0, 0], [y, y], [0,num_z], color="red", linewidth=1)
        
            
    
    for z in range(num_z + 1):
        for x in [0, num_cols]: #range(num_cols + 1):
            if x <= 1:
                #color = "orange"
                color = "k"
            else:
                color = "k"
            if x==0:
                linewidth=1
            else:
                linewidth=1
            alpha=1
            ax.plot([x, x], [0, num_rows], [z,z], color=color, linewidth=linewidth, alpha=alpha)

    for z in range(num_z + 1):
        for y in [0, num_rows]:
        #for y in range(num_rows + 1):
            if y==0 or y==num_rows:
                linewidth=1
            else:
                linewidth=1
            ax.plot([0, num_cols//2], [y, y], [z,z], color='k', linewidth=linewidth)
            ax.plot([num_cols//2, num_cols], [y, y],[z,z], color='k', linewidth=linewidth)
    # Vertices of the prism
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 4, 0],
        [0, 4, 0],
        [0, 0, 3],
        [1, 0, 3],
        [1, 4, 3],
        [0, 4, 3]
    ]

    # Define the vertices that compose each of the 6 faces of the prism
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[3], vertices[2], vertices[1]],
        [vertices[7], vertices[4], vertices[5], vertices[6]],
        [vertices[7], vertices[3], vertices[0], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    # Create a polygon and add it to the plot
    #polygon = Poly3DCollection(faces, color='lightblue', alpha=0.1)  # with some transparency
    #ax.add_collection3d(polygon)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.gca().invert_yaxis()  # Invert y axis so the (0,0) is at the top-left
    plt.savefig("b.pdf")

draw_tensor_grid()