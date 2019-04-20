  integer resc(3)[*], key[*], tmp(3)
  integer me, boss

  me=this_image()
  boss=3

  if (me==boss) then
     resc = 0
  else
     resc = -huge(0)         !! dust
  endif

  sync all

!! start

  if (me/=boss) then
     do i=1,100
        lock(key[boss])
            tmp = resc[boss]
            tmp(1) = tmp(1) + 1
            tmp(2) = tmp(2) + 1
            resc[boss] = tmp
        unlock(key[boss])
     enddo
  else   !! me==boss
     do i=1,100*(num_images()-1)
        lock(key)
            resc(1) = resc(1) - 1
            resc(3) = resc(3) + 1
        unlock(key)
     enddo
  endif

  write(*,*) me, ":", resc

end program

