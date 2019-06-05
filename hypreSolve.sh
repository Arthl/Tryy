# Go to hypreHOME
HYPREHOME=$1
cd $HYPREHOME

NPROC=$2

# Call hypreSolver
mpirun -np $NPROC hypreMatlabSolve
