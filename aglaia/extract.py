"""This script extracts the geometry, energy, forces and atom labels from a data set downloaded from here:
http://quantum-machine.org/datasets/
"""

import os
import ast
import numpy as np

def load_data(directory, n_samples=50):

    # Making a list of the files to look at
    list_of_files = []
    subdirs = [x[0] for x in os.walk(directory)]
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]

    # Extracting the data from each file
    atoms = ""
    traj_coord = []
    ene = []
    forces = []

    for subdir in subdirs:
        for file in files[:n_samples]:
            filename = subdir + file
            with open(filename) as f:
                lines = f.readlines()
                tokens = lines[1].split(';')

                ene.append(float(tokens[0]))

                forces_mat = ast.literal_eval(tokens[1])
                forces_list = []
                for item in forces_mat:
                    for i in range(3):
                        forces_list.append(item[i])
                forces.append(forces_list)

                coord = []
                for line in lines[2:]:
                    tokens = line.split()
                    atoms += tokens[0]
                    coord_float = [float(i) for i in tokens[1:]]
                    for i in range(3):
                        coord.append(coord_float[i])

                traj_coord.append(coord)

    traj_coord = np.asarray(traj_coord)
    ene = np.asarray(ene)
    forces = np.asarray(forces)

    return traj_coord, ene, forces

if __name__ == "__main__":
    coord, ene, forces = load_data("/Users/walfits/Documents/aspirin/")