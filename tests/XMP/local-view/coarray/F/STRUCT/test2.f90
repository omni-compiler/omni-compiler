  subroutine assign_t1(a,b)
    type t1
       integer n
       real r
    end type t1
    type(t1), intent(out):: a
    real, intent(in) :: b
    a%n = 0
    a%r = b
  end subroutine assign_t1

  program main
    interface assignment(=) 
       subroutine assign_t1(a,b)
         type t1
            integer n
            real r
         end type t1
         type(t1),intent(out):: a
         real,intent(in) :: b
       end subroutine assign_t1
    end interface

    type t1
       integer n
       real r
    end type t1

    type(t1) :: snd
    real :: fst

    snd%n = -99
    snd%r = -99.9
    fst = 3.456

    snd = fst

    write(*,*) snd
    
  end program main


