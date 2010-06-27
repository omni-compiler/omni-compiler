      integer function func1 (x)
        integer x
        func1 = x
        pause 12345
      end

      real function func2 (x)
        real x
        func2 = x
        pause 'ABORT FUNC2'
      end
