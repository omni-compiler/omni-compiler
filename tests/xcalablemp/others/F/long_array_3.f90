      PROGRAM main
        INTEGER, DIMENSION(1,1,1,2,1,1,2), CODIMENSION[1,1,1,2,1,1,1,*] :: a
        if(this_image() == 1) then
          a(1, 1, 1, 2, 1, 1, 2)[1, 1, 1, 2, 1, 1, 1, 2] = 10
        END IF
        SYNC ALL
        if(this_image() == 2) then
!         PRINT *, a(1, 1, 1, 2, 1, 1, 2)[1, 1, 1, 2, 1, 1, 1, 2]
          if(a(1, 1, 1, 2, 1, 1, 2)[1, 1, 1, 2, 1, 1, 1, 2].eq.10) then
            PRINT *, 'PASS'
          else
            PRINT *, 'NG'
            CALL EXIT(1)
          end if
        end if
      END PROGRAM main
