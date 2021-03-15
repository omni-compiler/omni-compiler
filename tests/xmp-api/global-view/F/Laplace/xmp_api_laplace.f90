program laplace
  use xmp_api
  implicit none
  integer,parameter :: N1=100, N2=200
  real(8),parameter :: PI = 3.141592653589793238463

  integer :: niter=10
!  real(8) :: u(N1,N2),uu(N1,N2)
  real(8), ALLOCATABLE :: u ( : , : )
  real(8), ALLOCATABLE :: uu ( : , : )
  real(8) :: value = 0.0
  integer(8) :: p_desc, t_desc, u_desc, uu_desc
  integer(4) :: i,j,k
  integer(4) :: j_start, j_end, j_step, i_start, i_end, i_step
  integer(8) :: g_i, g_j
  integer(8), dimension(2) :: dim_lb, dim_ub
  integer(4), dimension(2) :: node_dims, local_lb, local_ub
  integer(4) :: status

  call xmp_api_init

! !$xmp nodes p(4,*)
  node_dims(1) = 4
  node_dims(2) = -1
  call xmp_global_nodes(p_desc,2,node_dims,.false.)

!  print *,"init node ..."
!  call xmp_barrier

! !$xmp template t(N1,N2)
  dim_lb(1) = 1
  dim_ub(1) = N1
  dim_lb(2) = 1
  dim_ub(2) = N2
  call xmp_new_template(t_desc,p_desc, 2, dim_lb, dim_ub)
! !$xmp distribute t(block, block) onto p
  call xmp_dist_template_BLOCK(t_desc, 1, 1, status)
  call xmp_dist_template_BLOCK(t_desc, 2, 2, status)

!  print *,"init template .."
!  call xmp_barrier

! !$xmp align u(i,j) with t(i, j)
  dim_lb(1) = 1
  dim_ub(1) = N1
  dim_lb(2) = 1
  dim_ub(2) = N2
  call xmp_new_array(u_desc, t_desc, XMP_DOUBLE, 2, dim_lb, dim_ub)
  call xmp_align_array(u_desc, 1, 1, 0, status)
  call xmp_align_array(u_desc, 2, 2, 0, status)
  call xmp_get_array_local_dim(u_desc, local_lb, local_ub,status)
  allocate(u(local_lb(1):local_ub(1), local_lb(2):local_ub(2)))
  call xmp_allocate_array(u_desc, loc(u), status)
  
!  print *,"init array u  .."
!  call xmp_barrier

! !$xmp align uu(i,j) with t(i, j)
  call xmp_new_array(uu_desc, t_desc, XMP_DOUBLE, 2, dim_lb, dim_ub)
  call xmp_align_array(uu_desc, 1, 1, 0, status)
  call xmp_align_array(uu_desc, 2, 2, 0, status)

! !$xmp shadow uu(1:1,1:1)
  call xmp_set_shadow(uu_desc,1,1,1, status)
  call xmp_set_shadow(uu_desc,2,1,1, status)
  call xmp_get_array_local_dim(uu_desc, local_lb, local_ub,status)
  allocate(uu(local_lb(1):local_ub(1), local_lb(2):local_ub(2)))
  call xmp_allocate_array(uu_desc, loc(uu), status)

!  print *,"init array uu  .."
!  call xmp_barrier

! !$xmp loop (i,j) on t(i,j)
  call xmp_loop_schedule(1,N2,1,t_desc,2,j_start,j_end,j_step, status)
  call xmp_loop_schedule(1,N1,1,t_desc,1,i_start,i_end,i_step, status)
  do j= j_start, j_end, j_step ! 1,N2
     do i= i_start, i_end, i_step ! 1,N1
        u(i,j)=0.0
        uu(i,j)=0.0
     end do
  end do

!  print *,"init clear ..."
!  call xmp_barrier

!  !$xmp loop (i,j) on t(i,j)
  call xmp_loop_schedule(2,N2-1,1,t_desc,2,j_start,j_end,j_step, status)
  call xmp_loop_schedule(2,N1-1,1,t_desc,1,i_start,i_end,i_step, status)
  do j= j_start, j_end, j_step !2,N2-1
     do i= i_start, i_end, i_step !2,N1-1
        call xmp_template_ltog(t_desc,1,i,g_i, status)
        call xmp_template_ltog(t_desc,2,j,g_j, status)
        u(i,j)=sin(dble(g_i-1)/N1*PI)+cos(dble(g_j-1)/N2*PI)
     end do
  end do

!  print *,"init set init value ..."
  call xmp_barrier

  do k=1,niter

     if(xmp_node_num() == 1) then
        print *,'k=',k
     end if

! !$xmp loop (i,j) on t(i,j)
     do j= j_start, j_end, j_step !2,N2-1
        do i= i_start, i_end, i_step !2,N1-1
           uu(i,j)=u(i,j)
        end do
     end do

! !$xmp reflect (uu)
     call xmp_array_reflect(uu_desc, status)

! !$xmp loop (i,j) on t(i,j)
     do j= j_start, j_end, j_step !2,N2-1
        do i= i_start, i_end, i_step !2,N1-1
           u(i,j)=(uu(i-1,j) + uu(i+1,j) + uu(i,j-1) + uu(i,j+1))/4.0
        end do
     end do

     value = 0.0
! !$xmp loop (i,j) on t(i,j) reduction(+:value)
     do j= j_start, j_end, j_step !2,N2-1
        do i= i_start, i_end, i_step !2,N1-1
           value = value + dabs(uu(i,j) -u(i,j))
        end do
     end do
     call xmp_reduction_scalar(XMP_SUM, XMP_DOUBLE, loc(value), status)

!! !$xmp task on p(1,1)
     if(xmp_node_num() == 1) then
        print *, 'Verification =', value
     end if
!! !$xmp end task

  enddo
  call xmp_api_finalize
  
end program laplace
