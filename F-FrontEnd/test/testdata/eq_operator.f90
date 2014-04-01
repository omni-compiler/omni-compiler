      integer a(10,10), b(100)
      logical c
      c = any(a .eq. reshape(b, (/10,10/) ) )
      end
