module main

   integer, parameter, public :: int_kind  = kind(1) 

   integer (int_kind), parameter, public ::  &  ! model size parameters
      nx_global =  400 ,&! extent of horizontal axis in i direction
      ny_global =  288 ,&! extent of horizontal axis in j direction
      km = 32          ,&! number of vertical levels
      nt =  2            ! total number of tracers


   integer (int_kind), parameter, public :: &
     !max_blocks_tropic =  96   !   in each distribution
      max_blocks_clinic = 421, &! max number of blocks per processor
      max_blocks_tropic = 545   !   in each distribution
   
 end module main
