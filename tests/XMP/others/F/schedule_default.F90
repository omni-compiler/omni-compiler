subroutine test
integer :: i,j
!$OMP PARALLEL DO  PRIVATE(i) SCHEDULE(STATIC,100) DEFAULT(SHARED)
DO i=1,100
j=j+1
PRINT *,"SCHED_DYNAMIC"
end do
end subroutine test
