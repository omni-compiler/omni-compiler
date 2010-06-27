for j in MPI-BT/ MPI-FT/ OMP-LU/ OMP-MG/ bad/
do
    echo **-- $j --**
      (cd $j; time for i in *.f*; do echo - $i -; ~/prj/f2f/dev103/trunk/F-FrontEnd/src/F_Front $i /dev/null; echo = $i =; done)
    echo **== $j ==**
done
