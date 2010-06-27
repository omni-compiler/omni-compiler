!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! This program implements a simple molecular dynamics simulation,
!   using the velocity Verlet time integration scheme. The particles
!   interact with a central pair potential.
! Parallelism is implemented via OpenMP directives.
! THIS PROGRAM USES THE FORTRAN90 RANDOM_NUMBER FUNCTION AND ARRAY 
!   SYNTAX
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      program md
      implicit none

      ! simulation parameters
      integer ndim       ! dimensionality of the physical space
      integer nparts     ! number of particles
      integer nsteps     ! number of time steps in the simulation
!      parameter(ndim=3,nparts=500,nsteps=1000)
      parameter(ndim=3,nparts=500,nsteps=10)
      real*8 mass        ! mass of the particles
      real*8 dt          ! time step
      real*8 box(ndim)   ! dimensions of the simulation box
      parameter(mass=1.0,dt=1.0e-4)

      ! simulation variables
      real*8 position(ndim,nparts)
      real*8 velocity(ndim,nparts)
      real*8 force(ndim,nparts)
      real*8 accel(ndim,nparts)
      real*8 potential, kinetic, E0
      integer i

      box(1:ndim) = 10.

      ! set initial positions, velocities, and accelerations
      call initialize(nparts,ndim,box,position,velocity,accel)

      ! compute the forces and energies
      call compute(nparts,ndim,box,position,velocity,mass,
     .                                      force,potential,kinetic)
      E0 = potential + kinetic
      write(*,*) 'init: ', potential, kinetic, E0

      ! This is the main time stepping loop
      do i=1,nsteps
          call compute(nparts,ndim,box,position,velocity,mass,
     .                                      force,potential,kinetic)
          write(*,*) potential, kinetic,(potential + kinetic - E0)/E0
          call update(nparts,ndim,position,velocity,force,accel,mass,dt)
      enddo

      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute the forces and energies, given positions, masses,
! and velocities
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine compute(np,nd,box,pos,vel,mass,f,pot,kin)
      implicit none

      integer np
      integer nd
      real*8  box(nd)
      real*8  pos(nd,np)
      real*8  vel(nd,np)
      real*8  f(nd,np)
      real*8  mass
      real*8  pot
      real*8  kin

      real*8 dotr8
      external dotr8
      real*8 v, dv, x

      integer i, j, k
      real*8  rij(nd)
      real*8  d
      real*8  PI2
      parameter(PI2=3.14159265d0/2.0d0)

      ! statement function for the pair potential and its derivative
      ! This potential is a harmonic well which smoothly saturates to a
      ! maximum value at PI/2.
      v(x) = sin(min(x,PI2))**2.
      dv(x) = 2.*sin(min(x,PI2))*cos(min(x,PI2))

      pot = 0.0
      kin = 0.0

      ! The computation of forces and energies is fully parallel.
!$omp  parallel do
!$omp& default(shared)
!$omp& private(i,j,k,rij,d)
!$omp& reduction(+ : pot, kin)
      do i=1,np
        ! compute potential energy and forces
        f(1:nd,i) = 0.0
        do j=1,np
             if (i .ne. j) then
               call dist(nd,box,pos(1,i),pos(1,j),rij,d)
               ! attribute half of the potential energy to particle 'j'
               pot = pot + 0.5*v(d)
               do k=1,nd
                 f(k,i) = f(k,i) - rij(k)*dv(d)/d
               enddo
             endif
        enddo
        ! compute kinetic energy
        kin = kin + dotr8(nd,vel(1,i),vel(1,i))
      enddo
!$omp  end parallel do
      kin = kin*0.5*mass
  
      return
      end
       
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Initialize the positions, velocities, and accelerations.
! The Fortran90 random_number function is used to choose positions.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine initialize(np,nd,box,pos,vel,acc)
      implicit none

      integer np
      integer nd
      real*8  box(nd)
      real*8  pos(nd,np)
      real*8  vel(nd,np)
      real*8  acc(nd,np)

      integer i, j
      real*8 x

      do i=1,np
        do j=1,nd
          call random_number(x)
          pos(j,i) = box(j)*x
          vel(j,i) = 0.0
          acc(j,i) = 0.0
        enddo
      enddo

      return
      end

! Compute the displacement vector (and its norm) between two particles.
      subroutine dist(nd,box,r1,r2,dr,d)
      implicit none

      integer nd
      real*8 box(nd)
      real*8 r1(nd)
      real*8 r2(nd)
      real*8 dr(nd)
      real*8 d

      integer i

      d = 0.0
      do i=1,nd
        dr(i) = r1(i) - r2(i)
        d = d + dr(i)**2.
      enddo
      d = sqrt(d)

      return
      end

! Return the dot product between two vectors of type real*8 and length n
      real*8 function dotr8(n,x,y)
      implicit none

      integer n
      real*8 x(n)
      real*8 y(n)

      integer i

      dotr8 = 0.0
      do i = 1,n
        dotr8 = dotr8 + x(i)*y(i)
      enddo

      return
      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Perform the time integration, using a velocity Verlet algorithm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine update(np,nd,pos,vel,f,a,mass,dt)
      implicit none

      integer np
      integer nd
      real*8  pos(nd,np)
      real*8  vel(nd,np)
      real*8  f(nd,np)
      real*8  a(nd,np)
      real*8  mass
      real*8  dt

      integer i, j
      real*8  rmass

      rmass = 1.0/mass

      ! The time integration is fully parallel
!$omp  parallel do
!$omp& default(shared)
!$omp& private(i,j)
      do i = 1,np
        do j = 1,nd
          pos(j,i) = pos(j,i) + vel(j,i)*dt + 0.5*dt*dt*a(j,i)
          vel(j,i) = vel(j,i) + 0.5*dt*(f(j,i)*rmass + a(j,i))
          a(j,i) = f(j,i)*rmass
        enddo
      enddo
!$omp  end parallel do

      return
      end
