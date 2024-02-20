#!/usr/bin/env python3

import os, sys, yaml, getopt
from time import time
import numpy as np
from ase.io import read
from ase.optimize import LBFGS, BFGS
from ase.calculators.espresso import Espresso

from ase import Atoms, Atom
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms, FixBondLength
from ase.build import fcc111,hcp0001,add_adsorbate,molecule,rotate
from ase.build import bulk
from ase.parallel import paropen
from ase.visualize import view

from ase.calculators.socketio import SocketIOCalculator
from ase.neb import NEB
from ase.optimize.fire import FIRE as QuasiNewton

def neb_ci(yml_file:str):
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

    restart_file = inputs['neb_restart']
    
    #nimg = 5
    nimg = inputs['nimg']

    images=read(restart_file,index='-%i:'%(nimg+2)) #Read in the last set of images, not the whole traj

    constraint = FixAtoms(mask=[atom.index for atom in images[0] if atom.tag == 0])

    for i in range(1,nimg+1):
        espresso = Espresso(
                        command=command,
                        pseudopotentials=pseudopotentials,
                        tstress=True,
                        tprnfor=True,
                        kpts=inputs['kpoints'],
                        disk_io='none',
                        input_data=espresso_settings,
                        )
        images[i].set_calculator(espresso)
        images[i].set_constraint(constraint)

    neb=NEB(images, climb=True)

    #Now run the relaxation with BFGS
    dyn = QuasiNewton(neb, trajectory=inputs['neb_output'], logfile=inputs['neb_log'])
    dyn.run(inputs['fmax'])

    end = time()
    duration = end - start

    with open(inputs['neb_log'], 'a') as f:
        f.write(f'Completed in {duration} seconds\n')

def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('run_neb_ci.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('command should be run_neb_ci.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    neb_ci(inputfile)
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
    