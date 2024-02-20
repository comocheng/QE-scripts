import sys
import os
from shutil import copyfile
from ase.build import bulk, fcc111, add_adsorbate
from ase.io import read, write
from ase import Atoms
from ase.optimize import BFGS, LBFGS, MDMin
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
import ase.io
from ase.io.espresso import read_espresso_out
from ase.io.trajectory import Trajectory
from ase.io.ulm import InvalidULMFileError
from ase.calculators.socketio import SocketIOCalculator
from time import time
import yaml
import getopt

def adsorption(yml_file:str):
    """
    yml_file: str, The input is an yaml input file path with parameters for the calculation
    """
    with open(yml_file) as f:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    
    start = time()
    
    logfile = inputs['log_path']
    
    fmax = inputs['fmax']  # eV/A
    
    adsorbate_file = inputs['adsorbate_path']
    
    slab_file = inputs['slab_path']
    
    if inputs['restart'] == False:
        # read adsorbate trajectory
        traj = Trajectory(adsorbate_file)
        adsorbate = traj[-1]
        adsorbate.set_tags([2] * len(adsorbate))
        
        height = inputs['height']
        
        # # read metal slab trajectory
        adslab = Trajectory(slab_file)[-1]
        
        # freeze bottom layers
        fix_bottom_layers = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag==0)])
        adslab.set_constraint(fix_bottom_layers)
        
        for i in range(inputs['adsorbate_number']):
        # 'ontop', 'bridge', 'fcc', 'hcp'
        # https://wiki.fysik.dtu.dk/ase/ase/build/surface.html#ase.build.add_adsorbate
            if inputs['bond_atom_index'] is None:
                add_adsorbate(adslab, adsorbate, height=height, position=inputs['position'], offset=inputs['offset'][i])
            else:  
                add_adsorbate(adslab, adsorbate, height=height, position=inputs['position'], offset=inputs['offset'][i], mol_index=inputs['bond_atom_index'])
    else:
        adslab = Trajectory(inputs['restart_traj_path'])[-1]

    espresso_settings = inputs['calculator']
    
    pseudopotentials = inputs['pseudopotentials']
    
    cores = inputs['gpu_number']
    
    command = f"mpirun -n {cores} /work/westgroup/chao/qe_gpu/7.0-gpu/bin/pw.x -input espresso.pwi -npool 1 -ndiag 1 > espresso.pwo"

    espresso = Espresso(
        command=command,
        pseudopotentials=pseudopotentials,
        tstress=True,
        tprnfor=True,
        kpts=inputs['kpoints'],
        disk_io='none',
        input_data=espresso_settings,
    )
    
    traj_file = inputs['output_traj_path']
    
    adslab.calc = espresso
    opt = MDMin(adslab, logfile=logfile, trajectory=traj_file)
    opt.run(fmax=0.5)
    opt = LBFGS(adslab, logfile=logfile, trajectory=traj_file)
    opt.run(fmax=fmax)
    
    #calculate the adsorption energy
    total = Trajectory(traj_file)[-1]
    metal= Trajectory(slab_file)[-1]
    ads = Trajectory(adsorbate_file)[-1]
    adsorption_energy = total.get_potential_energy() - ads.get_potential_energy() - metal.get_potential_energy()
    # adsorption_energy = total.get_potential_energy() - ads.get_potential_energy() - len(metal) / 3 * (-14955.550740)
    
    end = time()
    duration = end - start
    
    with open(logfile, 'a') as f:
        f.write(f'Adsorption energy: {adsorption_energy}')
        f.write(f'Completed in {duration} seconds\n')
        
def main(argv):
    inputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('adsorption.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('command should be adsorption.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    adsorption(inputfile)
    print ('Input file is "', inputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
