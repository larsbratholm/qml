import numpy as np
import os
import joblib

# class Data_extraction():
#
#     def __init__(self, XML_input="test.xml"):
#         self.xml_file = XML_input

def out_to_qml(input_filename, kJmol=False, demean=False, xyz=True):
    """
    This function takes a .out molpro file that did a scan and extracts the geometry and the energy and writes it in
     qml format.

    :param kJmol: Transform Hartrees to kJ/mol
    :param demean: Take the mean out
    :xyz: if the xyz files are printed out or not
    :return:
    """

    # This is the input file
    input_file = open(input_filename, 'r')

    # Geometries and energies
    data_x = []
    data_y = []

    for line in input_file:
        geom = []
        if "END OF GEOMETRY OPTIMIZATION." in line:
            for i in range(6):
                line = next(input_file)

            for i in range(7):
                line = line.strip()
                lineSplit = line.split(" ")
                lineSplit = list(filter(None, lineSplit))
                for item in lineSplit:
                    geom.append(item)
                line = next(input_file)
            data_x.append(geom)

        if "!RHF-UCCSD(T)-F12b energy" in line:
            line = line.strip()
            ene_str = line[len("!RHF-UCCSD(T)-F12b energy"):].strip()
            if kJmol:
                ene_kjmol = float(ene_str) * 2625.5
                data_y.append(ene_kjmol)
            else:
                data_y.append(float(ene_str))

    prop_file = open("properties.txt", "w")

    # Mean energy from the VR data set):
    mean_energy_kjmol = -349312.31810359145
    mean_energy_Ha = -349312.31810359145 / 2625.50

    for i in range(len(data_x)):
        file_ind = pad_filename(i, int(len(data_y)/10))

        if xyz:
            f_name = str(file_ind) + ".xyz"
            f = open(f_name, "w")
            write_xyz(f, data_x[i])
            f.close()

        if demean and kJmol:
            prop_file.write(file_ind + ".xyz\t" + str(data_y[i] - mean_energy_kjmol) + "\n")
        elif demean == True and kJmol == False:
            prop_file.write(file_ind + ".xyz\t" + str(data_y[i] - mean_energy_Ha) + "\n")
        else:
            prop_file.write(file_ind + ".xyz\t" + str(data_y[i]) + "\n")

    prop_file.close()

def xml_to_qml(self, kJmol=False, demean=False):
    """
    This function takes as an input the XML file that comes out of the electronic structure calculations and transforms
    it into 2 CSV files. The first one is the 'X part' of the data. It contains a sample per line. Each line has a format:
    atom label (string), coordinate x (float), coordinate y (float), coordinate z (float), ... for each atom in the system.
    The second file contains the 'Y part' of the data. It has a sample per line with the energy of each sample (float).
    :XMLinput: an XML file obtained from grid electronic structure calculations
    """

    # This is the input file
    inputFile = open(self.xml_file, 'r')

    # The information of the molecule is contained in the block <cml:molecule>...<cml:molecule>.
    # The atom xyz coordinates, the labels and the energy have to be retrieved
    # Each configuration corresponds to one line in the CSV files

    self.data_x = []
    self.data_y = []

    for line in inputFile:
        data = []
        if "<cml:molecule>" in line:
            for i in range(3):
                line = next(inputFile)
            while "</cml:atomArray>" not in line:
                indexLab = line.find("elementType=")
                indexX = line.find("x3=")
                indexY = line.find("y3=")
                indexZ = line.find("z3=")
                indexZend = line.find("/>")

                if indexLab >= 0:
                    data.append(line[indexLab + 13])
                    data.append(line[indexX + 4: indexY - 2])
                    data.append(line[indexY + 4: indexZ - 2])
                    data.append(line[indexZ + 4: indexZend - 1])


                line = next(inputFile)

            self.data_x.append(data)


        if '!RHF-UCCSD(T)-F12b energy' in line:
            line = line.strip()
            ene_str = line[len("!RHF-UCCSD(T)-F12b energy"):].strip()
            if kJmol:
                energy = float(ene_str) * 2625.50
            else:
                energy = float(ene_str)
            self.data_y.append(energy)

    # Properties file
    prop_file = open("properties.txt", "w")
    # mean_energy = -347703.86963516066
    mean_energy = -349312.31810359145

    for i in range(len(self.data_x)):
        file_ind = self.pad_filename(i)
        f_name = str(file_ind) + "-pred.xyz"
        f = open(f_name, "w")
        self.write_xyz(f, self.data_x[i])
        f.close()

        if demean:
            prop_file.write(file_ind + ".xyz\t" + str(self.data_y[i] - mean_energy) + "\n")
        else:
            prop_file.write(file_ind + ".xyz\t" + str(self.data_y[i]) + "\n")

    return None

def get_file_number(path_name):
    path_split = path_name.split("/")
    file_name = path_split[-1]
    name_split = file_name.split("_")
    file_number = int(name_split[0])
    return file_number

def pad_filename(filenumber, n_of_zeros=5):
    padding = "0"
    str_file_n = str(filenumber)
    n = n_of_zeros - len(str_file_n)
    file_idx = n*padding + str_file_n
    return file_idx

def molpro_to_qml_format(directory, key, kJmol=False, demean=False, xyz_write=False):

    # Obtaining the list of files to mine
    file_list = list_files(directory, key)
    file_list.sort()

    # Properties
    prop = {}

    # Coordinates
    xyz = {}

    # List of file numbers (some numbers are missing)
    file_idxs = []

    # Properties file
    prop_file = open("properties.txt", "w")

    # Iterating over all the files
    for item in file_list:
        # Extracting the geometry and the energy from a Molpro out file
        geom, ene, partial_ch = extract_molpro(item)

        if len(geom) != 68 or ene == "0" or len(partial_ch) != 34:
            print("The following file couldn't be read properly:" + str(item) + "\n")

        file_number = get_file_number(item)    # Unpadded file number
        file_idxs.append(file_number)

        if kJmol:
            prop[file_number] = float(ene) * 2625.50   # ene is now in kJ/mol
        else:
            prop[file_number] = float(ene)

        xyz[file_number] = geom

    # Calculating the mean energy
    # n_samples = len(file_list)
    # sum_energies = 0
    #
    # for key in prop:
    #     sum_energies += prop[key]
    #
    # mean_energy = sum_energies/n_samples
    # print("The mean energy is: " + str(mean_energy))

    # Writing the energies minus the mean to file (So that they can be written to file in order and the cartesian coordinates to a file
    sorted_idx = np.sort(file_idxs, kind="mergesort")

    # Chose a sample to use as the zero of energy
    ref_energy = 0.0
    for reference_idx in sorted_idx:
        if len(xyz[reference_idx]) != 68 or prop[reference_idx] == 0.0:
            continue
        else:
            ref_energy = prop[reference_idx]
    print("The reference energy is: %s" % (ref_energy))
    # ref_energy = -516891.5043706963

    # Make a folder in which to print the xyz
    if xyz_write:
        dir = os.path.join(os.getcwd(), "geoms/test")
        if not os.path.exists(dir):
            os.makedirs(dir)

    for idx in sorted_idx:

        if len(xyz[idx]) != 68 or prop[idx] == 0.0:
            continue

        file_ind = pad_filename(idx)   # Padded file number
        if demean:
            prop_file.write(file_ind + ".xyz\t" + str(prop[idx] - ref_energy) + "\n")
        else:
            prop_file.write(file_ind + ".xyz\t" + str(prop[idx]) + "\n")

        if xyz_write:
            f_name = str(file_ind) + ".xyz"
            f = open(os.path.join(dir, f_name), "w")
            write_xyz(f, xyz[idx])
            f.close()

def import_molpro(directory, key, kJmol=False):
    # Obtaining the list of files to mine
    file_list = list_files(directory, key)
    file_list.sort()

    # Properties
    prop = {}

    # Coordinates
    xyz = {}

    # List of file numbers (some numbers are missing)
    file_idxs = []

    # Iterating over all the files
    for item in file_list:
        # Extracting the geometry and the energy from a Molpro out file
        geom, ene, partial_ch = self.extract_molpro(item)

        if len(geom) != 28 or ene == "0" or len(partial_ch) != 14:
            print("The following file couldn't be read properly:" + str(item) + "\n")

        file_number = self.get_file_number(item)  # Unpadded file number
        file_idxs.append(file_number)

        if kJmol:
            prop[file_number] = float(ene) * 2625.50  # ene is now in kJ/mol
        else:
            prop[file_number] = float(ene)

        xyz[file_number] = geom
        sorted_idx = np.sort(file_idxs, kind="mergesort")

    return xyz, prop, sorted_idx

def list_files(dir, key):
    """
    This function walks through a directory and makes a list of the files that have a name containing a particular string
    :dir: path to the directory to explore
    :key: string to look for in file names
    :return: list of files containing "key" in their filename
    """

    r = []  # List of files to be joined together
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = next(os.walk(subdir))[2]

        for file in files:
            isTrajectory = file.find(key)
            if isTrajectory >= 0:
                r.append(subdir + "/" + file)
    return r

def extract_molpro(MolproInput):
    """
    This function takes one Molpro .out file and returns the geometry, the energy and the partial charges on the atoms.
    :MolproInput: the molpro .out file (string)
    :return:
    :rawData: List of strings with atom label and atom coordinates - example ['C', '0.1, '0.1', '0.1', ...]
    :ene: Value of the energy (string)
    :partialCh: List of strings with atom label and its partial charge - example ['C', '6.36', 'H', ...]
    """

    # This is the input file
    inputFile = open(MolproInput, 'r')

    # This will contain the data
    rawData = []
    ene = "0"
    partialCh = []

    for line in inputFile:
        # The geometry is found on the line after the keyword "geometry={"
        if "geometry={" in line:
            for i in range(17):
                line = next(inputFile)
                line = line.strip()
                line = line.strip(",")
                lineSplit = line.split(",")
                for j in range(len(lineSplit)):
                    rawData.append(lineSplit[j])
        # The energy is found two lines after the keyword "Final beta  occupancy:"
        elif "!UKS STATE  1.1 Energy" in line:
            line = line.strip()
            ene = line[len("!UKS STATE  1.1 Energy"):].strip()
        elif "Total charge composition:" in line:
            line = next(inputFile)
            line = next(inputFile)
            for i in range(17):
                line = next(inputFile)
                lineSplit = line.rstrip().split(" ")
                lineSplit = list(filter(None, lineSplit))
                partialCh.append(lineSplit[1])
                partialCh.append(lineSplit[-2])

    return rawData, ene, partialCh

def write_xyz(file, geom):
    """
    This takes a file and writes the geometry in a VMD compatible xyz format.
    :param file:
    :param geom:
    :return: None
    """
    n_atoms = len(geom)//4
    file.write(str(n_atoms) + "\n")
    file.write("\n")

    for i in range(0, len(geom), 4):
        for j in range(4):
            file.write(str(geom[i+j]))
            file.write("\t")
        file.write("\n")

def molpro_to_vmd(directory, key):

    # File vmd
    traj = open("traj.xyz", "w")

    file_list = list_files(directory, key)
    file_list.sort()

    # Coordinates
    xyz = {}

    # List of file numbers (some numbers are missing)
    file_idxs = []

    # Iterating over all the files
    for item in file_list:
        # Extracting the geometry and the energy from a Molpro out file
        geom, ene, partial_ch = extract_molpro(item)

        if len(geom) != 28 or ene == "0" or len(partial_ch) != 14:
            print("The following file couldn't be read properly:" + str(item) + "\n")

        file_number = get_file_number(item)  # Unpadded file number
        file_idxs.append(file_number)

        xyz[file_number] = geom

    # Writing the energies minus the mean to file (So that they can be written to file in order and the cartesian coordinates to a file
    sorted_idx = np.sort(file_idxs, kind="mergesort")

    for idx in sorted_idx:
        write_xyz(traj, xyz[idx])

    traj.close()

if __name__ == "__main__":

    molpro_to_qml_format("/Volumes/Transcend/data_sets/OH_squalane_model/training_Molpro/b3lyp", "b3lyp_tzvp_u.out", kJmol=True, demean=True, xyz_write=False)


