program main
  outer:DO
     READ *, val
     new_val = 0
     inner:DO
        new_val = new_val + proc_val(val)
        IF (new_val >= max_val) EXIT inner
        IF (new_val == 0) EXIT outer
     END DO inner
  END DO outer
end program main
