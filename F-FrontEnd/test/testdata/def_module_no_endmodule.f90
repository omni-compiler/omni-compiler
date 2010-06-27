      module no_endmodule_m1
        integer a
      end module
      
      module no_endmodule_m2
        integer b
      end

      program main
        use no_endmodule_m1
        use no_endmodule_m2
        print *,a,b
      end

