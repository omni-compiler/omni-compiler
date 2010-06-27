        program main
            logical,dimension(3)::a
            integer,dimension(3)::b
            where(a)
                b = 1
            end where
        end
