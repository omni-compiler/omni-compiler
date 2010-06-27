      program test_conjg
          complex      z
          complex(KIND=8) dz
          z = (2.0, 3.0)
          dz = (2.71_8, -3.14_8)
          z= conjg(z)
          dz= conjg(dz)
          print *, z, dz
      end program test_conjg
