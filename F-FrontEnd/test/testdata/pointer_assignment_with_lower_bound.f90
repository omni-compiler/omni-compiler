      PROGRAM main
        REAL,TARGET  :: x(-100:100,-10:10)
        REAL,POINTER :: p(:,:)
        p(1:,1:) => x
      END PROGRAM main
