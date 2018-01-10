module init
  implicit none
  public :: init_u_and_y
contains

  subroutine init_u_and_y(ue_t)
    !$xmp nodes p(2)
    !$xmp template t(32)
    !$xmp distribute t(block) onto p
    implicit none
    complex(8), intent(inout):: ue_t(16,32)
    integer :: itb,iz
    !$xmp align  ue_t(*,k) with t(k)
    !$xmp shadow ue_t(0,1)

    !$xmp loop (iz) on t(iz)
    do iz=1,32
       do itb=1,16
          ue_t(itb,iz)=(0.0d0,0.0d0)
       enddo
    enddo
  end subroutine init_u_and_y
end module init
