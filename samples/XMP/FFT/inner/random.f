      subroutine get_random(a, n, key)
      integer n, key
      real*8 a(n)
      integer, allocatable, save :: seeds(:)
      integer, save :: n_seeds

      if (key.ne.0) then        !! first call
         call random_seed(size=n_seeds)
         allocate(seeds(n_seeds))
         call random_seed(get=seeds)
         seeds(1) = mod(150094772735952593_8*key,10461401779_8)
      end if

      call random_seed(put=seeds)
      call random_number(a)

      return 
      end subroutine get_random
