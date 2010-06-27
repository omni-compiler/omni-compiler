      program ft

      logical timers_enabled

      call timer_start(T_total)
      if (timers_enabled) call timer_start(T_setup)
      end
