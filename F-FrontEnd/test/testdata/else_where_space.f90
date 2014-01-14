      logical IL1(10),IL2(10)
      IL1=.true.
      where(IL1)
          IL2=.true.
      else where
          IL2=.false.
      end where
      end
