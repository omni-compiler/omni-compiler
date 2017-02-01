      MODULE m_allocatable_character
        CHARACTER(len=:), ALLOCATABLE :: name
      END MODULE m_allocatable_character

      PROGRAM main
        USE m_allocatable_character
        name = 'Tommy'
        if(len(name).eq.5) then
          PRINT *, 'PASS'
        else
          PRINT *, 'NG'
          CALL EXIT(1)
        end if
      END PROGRAM main
