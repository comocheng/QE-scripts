import os
import numpy as np
from ase.io import read
from ase.optimize import LBFGS, BFGS
from ase.calculators.espresso import Espresso
from ase.io.trajectory import Trajectory
import sys
from ase.neb import NEB
from time import time
import yaml
import getopt

def ts_search(yml_file:str):
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

    initial=read(inputs['initial_image'])
    final=read(inputs['final_image'])
    
    if inputs['restart'] is True:
        neb_output = inputs['neb_output']
        images = io.read(f'{neb_output}@-7:')
    else:    
        images=[initial]
        for i in range(5):
                espresso = Espresso(
                            command=command,
                            pseudopotentials=pseudopotentials,
                            tstress=True,
                            tprnfor=True,
                            kpts=inputs['kpoints'],
                            ensemble_energies=True,
                            disk_io='none',
                            input_data=espresso_settings,
                           )
                image=initial.copy()
                image.set_calculator(espresso)
                images.append(image)
        images.append(final)

    neb=NEB(images)
    neb.interpolate()

    #Now run the relaxation with BFGS
    dyn = LBFGS(neb, trajectory=inputs['neb_output'], logfile=inputs['neb_log'])
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
        print('run_neb.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('command should be run_neb.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    ts_search(inputfile)
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])