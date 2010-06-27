      module multi_mod1
        integer a
      end module
      
      module multi_mod2
        integer b

      end module

      program main
        use multi_mod1
        use multi_mod2
        print *,a,b
      end
