#!/usr/bin/env python3

import os, sys, yaml, getopt
from time import time
import numpy as np
from ase.io import read
from ase.optimize import LBFGS, BFGS
from ase.calculators.espresso import Espresso
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms, FixBondLength
from ase.visualize import view
from sella import Sella, Constraints

def sella_opt(yml_file:str):
    """
    yml_file: str, The input is an yaml input file path with parameters for the calculation
    """
    start = time()
    
    with open(yml_file) as f:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    if inputs['cpu'] == True:
        command = '/shared/centos7/openmpi/4.0.5-skylake-gcc10.1/bin/mpirun -np {} /work/westgroup/espresso/qe-7.0/bin/pw.x -input espresso.pwi > espresso.pwo'.format(inputs['cpu_number'])
    elif inputs['gpu'] == True:
        gpu_num = inputs['gpu_number']
        command = f"mpirun -n {gpu_num} /work/westgroup/chao/qe_gpu/7.0-gpu/bin/pw.x -input espresso.pwi -npool 1 -ndiag 1 > espresso.pwo"
        
    espresso_settings = inputs['calculator']
    
    pseudopotentials = inputs['pseudopotentials']
    
    if not inputs['restart']:
        restart_file = inputs['neb_restart']
    
        nimg = inputs['img_num']
    
        images=read(restart_file,index='-%i:'%(nimg)) #Read in the last set of images, not the whole traj
        
        ts_id = inputs['ts_index']
        
        img= images[ts_id]

    else:
        img = Trajectory(inputs['sella_restart_path'])[-1]

    constraint = FixAtoms(mask=[atom.index for atom in img if atom.tag == 0])

    espresso = Espresso(
                    command=command,
                    pseudopotentials=pseudopotentials,
                    tstress=True,
                    tprnfor=True,
                    kpts=inputs['kpoints'],
                    disk_io='none',
                    input_data=espresso_settings,
                    )
    img.set_calculator(espresso)
    img.set_constraint(constraint)
    cons = Constraints(img)
    for atm in [atom for atom in img if atom.tag == 0]:
        cons.fix_translation(atm.index)
    # Set up a Sella Dynamics object
    dyn = Sella(img, constraints=cons,trajectory=inputs['sella_output'], logfile=inputs['sella_log'])
    dyn.run(inputs['fmax'], 1000)

    end = time()
    duration = end - start

    with open(inputs['sella_log'], 'a') as f:
        f.write(f'Completed in {duration} seconds\n')

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('run_sella.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('command should be run_sella.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    sella_opt(inputfile)
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
