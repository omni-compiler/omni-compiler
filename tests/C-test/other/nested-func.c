#include <stdio.h>

int main() {
    void func () {
        #pragma omp parallel
        printf("%d\n", 1);
    }

    func ();

    return 0;
}
