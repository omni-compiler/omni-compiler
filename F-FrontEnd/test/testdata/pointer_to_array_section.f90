      PROGRAM bug130
          REAL,POINTER :: DLA(:)
          REAL DDA(5)
          TARGET DDA
          DLA => DDA(2:3)
      END
