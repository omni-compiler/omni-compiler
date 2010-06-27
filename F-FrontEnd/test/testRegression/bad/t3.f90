!! main
      program  complex_add_example
      implicit none

      type complex_t
      real(8):: real_p
      real(8):: img_p
      end type complex_t


      type(complex_t):: x

      type(complex_t):: z
      x%real_p = 1.0
      x%img_p = 2.0

      z = complex_add(x, 1.0)

      write(*,*) 'z=(', z%real_p, z%img_p, ')'




!! func
      contains

      function complex_add(x, y) result(w)
      implicit none

      type(complex_t):: x, w
      real(4):: y

      x%real_p = x%real_p + y
      w = x
      end function complex_add


      end program complex_add_example
