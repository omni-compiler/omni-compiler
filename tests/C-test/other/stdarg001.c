static char rcsid[] = "";
/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#include <stdarg.h>
#include <stdio.h>



int
test_stdarg (char *fmt, ...)
{
  va_list	ap;
  void		*ptr;
  char		c;
  short		s;
  long		l;
  long long	ll;
  int		i;
  float		f;
  double	d;

  int		errors = 0;



  va_start (ap, fmt);
  while (*fmt) {
    switch (*fmt++) {
    case 'p':
      ptr = va_arg (ap, void *);
      break;

    case 'c':
      c = (char) va_arg (ap, int);
      if (c != *((char *)ptr)) {
	errors += 1;
      }
      break;

    case 's':
      s = (short) va_arg (ap, int);
      if (s != *((short *)ptr)) {
	errors += 1;
      }
      break;

    case 'l':
      l = va_arg (ap, long);
      if (l != *((long *)ptr)) {
	errors += 1;
      }
      break;

    case 'L':
      ll = va_arg (ap, long long);
      if (ll != *((long long *)ptr)) {
	errors += 1;
      }
      break;

    case 'i':
      i = va_arg (ap, int);
      if (i != *((int *)ptr)) {
	errors += 1;
      }
      break;

    case 'f':
      f = (float) va_arg (ap, double);
      if (f != *((float *)ptr)) {
	errors += 1;
      }
      break;

    case 'd':
      d = va_arg (ap, double);
      if (d != *((double *)ptr)) {
	errors += 1;
      }
      break;

    default:
      errors += 1;
      break;
    }
  }

  va_end (ap);

  return errors;
}



int
test_stdarg_unsigned (char *fmt, ...)
{
  va_list	ap;
  void		*ptr;
  unsigned char		c;
  unsigned short	s;
  unsigned long		l;
  unsigned long long	ll;
  unsigned int		i;

  int		errors = 0;



  va_start (ap, fmt);
  while (*fmt) {
    switch (*fmt++) {
    case 'p':
      ptr = va_arg (ap, void *);
      break;

    case 'c':
      c = (unsigned char) va_arg (ap, unsigned int);
      if (c != *((unsigned char *)ptr)) {
	errors += 1;
      }
      break;

    case 's':
      s = (unsigned short) va_arg (ap, unsigned int);
      if (s != *((unsigned short *)ptr)) {
	errors += 1;
      }
      break;

    case 'l':
      l = va_arg (ap, unsigned long);
      if (l != *((unsigned long *)ptr)) {
	errors += 1;
      }
      break;

    case 'L':
      ll = va_arg (ap, unsigned long long);
      if (ll != *((unsigned long long *)ptr)) {
	errors += 1;
      }
      break;

    case 'i':
      i = va_arg (ap, unsigned int);
      if (i != *((int *)ptr)) {
	errors += 1;
      }
      break;

    default:
      errors += 1;
      break;
    }
  }

  va_end (ap);

  return errors;
}



int
main ()
{
  int	errors = 0;

  char		c  = 12;
  short		s  = 123;
  long		l  = 1234;
  int		i  = 12345;
  long long	ll = 123456;
  float		f  = 123456.7;
  double	d  = 123456.78;

  char		mc  = -12;
  short		ms  = -123;
  long		ml  = -1234;
  int		mi  = -12345;
  long long	mll = -123456;
  float		mf  = -123456.7;
  double	md  = -123456.78;


  errors += test_stdarg ("pcpsplpipLpfpd",
			 &c, c, &s, s, &l, l, &i, i, &ll, ll, &f, f, &d, d);
  errors += test_stdarg ("pcpsplpipLpfpd",
			 &mc, mc, &ms, ms, &ml, ml, &mi, mi, &mll, mll,
			 &mf, mf, &md, md);

  errors += test_stdarg ("pdpfpLpiplpspc",
			 &d, d, &f, f, &ll, ll, &i, i, &l, l, &s, s, &c, c);
  errors += test_stdarg ("pdpfpLpiplpspc",
			 &md, md, &mf, mf,
			 &mll, mll, &mi, mi, &ml, ml, &ms, ms, &mc, mc);

  errors += test_stdarg_unsigned ("pcpsplpipL",
				  &c, c, &s, s, &l, l, &i, i, &ll, ll);
  errors += test_stdarg_unsigned ("pcpsplpipL",
				  &mc, mc, &ms, ms, &ml, ml, &mi, mi, &mll, mll);

  errors += test_stdarg_unsigned ("pLpiplpspc",
				  &ll, ll, &i, i, &l, l, &s, s, &c, c);
  errors += test_stdarg_unsigned ("pLpiplpspc",
				  &mll, mll, &mi, mi, &ml, ml, &ms, ms, &mc, mc);

  if (errors == 0) {
    printf ("stdarg 001 : SUCCESS\n");
    return 0;
  } else {
    printf ("stdarg 001 : FAILED\n");
    return 1;
  }
}
