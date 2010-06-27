      real function hoehoe(x, y, z)
!        implicit none
        real x, y, z
        hoehoe = x + y + z
        return
        entry hoehoe_entry1(x, y, z)
          hoehoe_entry1 = x + y - z
          return
      end !function
      program main
        real a, b
        a = hoehoe(10.0, 10.0, 10.0)
        b = hoehoe_entry1(10.0, 10.0, 10.0)
        print *, a, b
      end !program
