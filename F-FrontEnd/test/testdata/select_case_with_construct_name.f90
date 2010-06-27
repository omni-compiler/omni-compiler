      program main
        logical ch_type, num_type, log_type
        character :: ch = 'c'
        integer :: num = 1
        logical :: log = .true.

        label1: select case (ch)
        case ('c') label1
          ch_type = .true.
        case ('h') label1
          ch_type = .true.
        case default label1
          ch_type = .false.
        end select label1

        label2: select case (num)
        case (0) label2
          num_type = .true.
        case (1:9) label2
          num_type = .true.
        case default label2
          num_type = .false.
        end select label2

        label3: select case (log)
        case (.true.) label3
          log_type = .true.
        case (.false.) label3
          log_type = .false.
        case default label3
          log_type = .false.
        end select label3
      end
