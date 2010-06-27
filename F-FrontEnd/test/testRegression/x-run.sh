 time for i in *.f*; do echo - $i -; ~/prj/f2f/dev113/trunk/F-FrontEnd/src/F_Front -TD=/tmp $i /dev/null; echo = $i =; done

for j in MPI-BT/ MPI-FT/ OMP-LU/ OMP-MG/ 
do
    echo **-- $j --*
    (cd $j; time for i in *.f*; do echo - $i -; ~/prj/f2f/dev113/trunk/F-FrontEnd/src/F_Front -TD=/tmp $i /dev/null; echo = $i =; done)

    echo **== $j ==*
done
