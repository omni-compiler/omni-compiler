      program main
        logical ch_type, num_type, log_type
        character :: ch = 'c'
        integer :: num = 1
        logical :: log = .true.

        select case (ch)
        case ('c')
          ch_type = .true.
        case ('h')
          ch_type = .true.
        case default
          ch_type = .false.
        end select

        select case (num)
        case (0)
          num_type = .true.
        case (1:9)
          num_type = .true.
        case default
          num_type = .false.
        end select

        select case (log)
        case (.true.)
          log_type = .true.
        case (.false.)
          log_type = .false.
        case default
          log_type = .false.
        end select
      end
