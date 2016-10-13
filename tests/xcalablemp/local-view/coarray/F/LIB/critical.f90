  integer resc(3)[*], tmp(3)
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
        critical
            tmp = resc[boss]
            tmp(1) = tmp(1) + 1
            tmp(2) = tmp(2) + 1
            resc[boss] = tmp
        end critical
     enddo
  else   !! me==boss
     do i=1,100*(num_images()-1)
        critical
            resc(1) = resc(1) - 1
            resc(3) = resc(3) + 1
        end critical
     enddo
  endif

  write(*,*) me, ":", resc

end program

