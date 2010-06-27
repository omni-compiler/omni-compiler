c CLASS = A
c  
c  
c  This file is generated automatically by the setparams utility.
c  It sets the number of processors and the class of the NPB
c  in this directory. Do not modify it by hand.
c  
        integer nx, ny, nz, maxdim, niter_default
        integer ntotal, nxp, nyp, ntotalp
        parameter (nx=256, ny=256, nz=128, maxdim=256)
        parameter (niter_default=6)
        parameter (nxp=nx+1, nyp=ny)
        parameter (ntotal=nx*nyp*nz)
        parameter (ntotalp=nxp*nyp*nz)
        logical  convertdouble
        parameter (convertdouble = .false.)
        character compiletime*11
        parameter (compiletime='06 Mar 2009')
        character npbversion*3
        parameter (npbversion='3.3')
        character cs1*3
        parameter (cs1='f77')
        character cs2*6
        parameter (cs2='$(F77)')
        character cs3*6
        parameter (cs3='(none)')
        character cs4*6
        parameter (cs4='(none)')
        character cs5*2
        parameter (cs5='-O')
        character cs6*2
        parameter (cs6='-O')
        character cs7*6
        parameter (cs7='randi8')
