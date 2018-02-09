      INTEGER, POINTER :: a(:,:)
      INTEGER, POINTER :: b(:,:)
      ALLOCATE(a(5,4), SOURCE=RESHAPE((/(i,i=1,20)/),(/5,4/)))
      END
