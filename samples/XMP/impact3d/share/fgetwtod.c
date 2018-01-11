#include <sys/time.h>
#include <stdio.h>

double fgetwtod_ ()
{
     static double wtod;
     static long time0;
     static int init=0;
     static struct timeval tod;
     static struct timezone tz;
     gettimeofday( &tod, &tz );
     if ( init == 0 ) {
        init = 1;
        time0 = tod.tv_sec;
     }
     wtod = ( tod.tv_sec - time0 ) + tod.tv_usec * 1.0e-6;
     return( wtod );
}
