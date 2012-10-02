
      PROGRAM DOSTATEMENTLABEL

      INTEGER*4 A(10)
      INTEGER I

      I = 1
      DO 5000
          A(I) = 20*I
          WRITE( *,*) A(I)
          I = I+1
          IF ( I == 11 ) THEN
            GOTO 6000
          END IF
 5000 CONTINUE

 6000 CONTINUE

      END PROGRAM DOSTATEMENTLABEL
