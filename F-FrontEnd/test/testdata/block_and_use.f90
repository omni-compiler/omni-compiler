      MODULE block_and_use
        INTEGER :: a
      END MODULE block_and_use

      PROGRAM MAIN
        use block_and_use
        BLOCK
          use block_and_use
        END BLOCK
      END PROGRAM MAIN

