/*
*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
/*
#include <gmp.h>
*/
#ifdef _ZLIB_
#include <zlib.h>
#endif

#include "import_module.h"
#include "hwint.h"
#include "safe-ctype.h"
#include "gfortran.h"
#if 0
#include "tsystem.h"
#endif
#include "insn-modes.h"

#define MODULE_EXTENSION ".mod"

/* Don't put any single quote (') in MOD_VERSION,
   if yout want it to be recognized.  */
#define MOD_VERSION "9"

#ifndef EXIT_SUCCESS
# define EXIT_SUCCESS 0
#endif
#ifndef EXIT_FAILURE
# define EXIT_FAILURE 1
#endif

/* If we don't have an overriding definition, set SUCCESS_EXIT_CODE and
   FATAL_EXIT_CODE to EXIT_SUCCESS and EXIT_FAILURE respectively,
   or 0 and 1 if those macros are not defined.  */
#ifndef SUCCESS_EXIT_CODE
# ifdef EXIT_SUCCESS
#  define SUCCESS_EXIT_CODE EXIT_SUCCESS
# else
#  define SUCCESS_EXIT_CODE 0
# endif
#endif

#ifndef FATAL_EXIT_CODE
# ifdef EXIT_FAILURE
#  define FATAL_EXIT_CODE EXIT_FAILURE
# else
#  define FATAL_EXIT_CODE 1
# endif
#endif

#define ICE_EXIT_CODE 4

#define PTRDIFF_TYPE            "int"
#define INT_TYPE_SIZE         32

extern int  mod_version;
extern char *modincludeDirv;

/* Structure that describes a position within a module file.  */

typedef struct
{
  int column, line;
  fpos_t pos;
#ifdef _ZLIB_
  long   posgz;
#endif
}
module_locus;

typedef enum
{
  P_UNKNOWN = 0, P_OTHER, P_NAMESPACE, P_COMPONENT, P_SYMBOL
}
pointer_t;

/* The fixup structure lists pointers to pointers that have to
   be updated when a pointer value becomes known.  */

typedef struct fixup_t
{
  void **pointer;
  struct fixup_t *next;
}
fixup_t;

/* Structure for holding extra info needed for pointers being read.  */

enum gfc_rsym_state
{
  UNUSED,
  NEEDED,
  USED
};

enum gfc_wsym_state
{
  UNREFERENCED = 0,
  NEEDS_WRITE,
  WRITTEN
};

typedef struct pointer_info
{
  BBT_HEADER (pointer_info);
  int integer;
  pointer_t type;

  /* The first component of each member of the union is the pointer
     being stored.  */

  fixup_t *fixup;

  union
  {
    void *pointer;      /* Member for doing pointer searches.  */

    struct
    {
      gfc_symbol *sym;
      char *true_name, *module, *binding_label;
      fixup_t *stfixup;
      gfc_symtree *symtree;
      enum gfc_rsym_state state;
      int ns, referenced, renamed;
      module_locus where;
    }
    rsym;

    struct
    {
      gfc_symbol *sym;
      enum gfc_wsym_state state;
    }
    wsym;
  }
  u;

}
pointer_info;

#define gfc_get_pointer_info() XCNEW (pointer_info)

/* Structure for holding module and include file search path.  */
typedef struct gfc_directorylist
{
  char *path;
  bool use_for_modules;
  struct gfc_directorylist *next;
}
gfc_directorylist;

typedef struct true_name
{
  BBT_HEADER (true_name);
  const char *name;
  gfc_symbol *sym;
}
true_name;

/* Holds switches parsed by gfc_cpp_handle_option (), but whose
   handling is deferred to gfc_cpp_init ().  */
typedef struct
{
    enum opt_code code;
    const char *arg;
}
gfc_cpp_deferred_opt_t;

struct gfc_cpp_option_data
{
  /* Argument of -cpp, implied by SPEC;
     if NULL, preprocessing disabled.  */
  const char *temporary_filename;

  const char *output_filename;          /* -o <arg>  */
  int preprocess_only;                  /* -E  */
  int discard_comments;                 /* -C  */
  int discard_comments_in_macro_exp;    /* -CC  */
  int print_include_names;              /* -H  */
  int no_line_commands;                 /* -P  */
  char dump_macros;                     /* -d[DMNU]  */
  int dump_includes;                    /* -dI  */
  int working_directory;                /* -fworking-directory  */
  int no_predefined;                    /* -undef */
  int standard_include_paths;           /* -nostdinc */
  int verbose;                          /* -v */
  int deps;                             /* -M */
  int deps_skip_system;                 /* -MM */
  const char *deps_filename;            /* -M[M]D */
  const char *deps_filename_user;       /* -MF <arg> */
  int deps_missing_are_generated;       /* -MG */
  int deps_phony;                       /* -MP */

  const char *multilib;                 /* -imultilib <dir>  */
  const char *prefix;                   /* -iprefix <dir>  */
  const char *sysroot;                  /* -isysroot <dir>  */

  /* Options whose handling needs to be deferred until the
     appropriate cpp-objects are created:
      -A predicate=answer
      -D <macro>[=<val>]
      -U <macro>  */
  gfc_cpp_deferred_opt_t *deferred_opt;
  int deferred_opt_count;
}
gfc_cpp_option;

typedef struct gfc_treap
{
  BBT_HEADER (gfc_treap);
}
gfc_bbt;

#if 0
/* Structure describing the result of decoding an option.  */

struct cl_decoded_option
{
  /* The index of this option, or an OPT_SPECIAL_* value for
     non-options and unknown options.  */
  size_t opt_index;

  /* Any warning to give for use of this option, or NULL if none.  */
  const char *warn_message;

  /* The string argument, or NULL if none.  For OPT_SPECIAL_* cases,
     the option or non-option command-line argument.  */
  const char *arg;

  /* The original text of option plus arguments, with separate argv
     elements concatenated into one string with spaces separating
     them.  This is for such uses as diagnostics and
     -frecord-gcc-switches.  */
  const char *orig_option_with_args_text;

  /* The canonical form of the option and its argument, for when it is
     necessary to reconstruct argv elements (in particular, for
     processing specs and passing options to subprocesses from the
     driver).  */
  const char *canonical_option[4];

  /* The number of elements in the canonical form of the option and
     arguments; always at least 1.  */
  size_t canonical_option_num_elements;

  /* For a boolean option, 1 for the true case and 0 for the "no-"
     case.  For an unsigned integer option, the value of the
     argument.  1 in all other cases.  */
  int value;

  /* Any flags describing errors detected in this option.  */
  int errors;
};
#endif


/* Local variables */

static true_name *true_name_root;

/* The FILE for the module we're reading or writing.  */
static FILE *module_fp;

static unsigned char gzType = 0;

/* MD5 context structure.  */
/*
static struct md5_ctx ctx;
*/

/* The name of the module we're reading (USE'ing) or writing.  */
static const char *module_name;
/*
static gfc_use_list *module_list;
*/
static char* module_content;

static long module_pos;
static int module_line, module_column, only_flag;
static int prev_module_line, prev_module_column, prev_character;

static enum
{ IO_INPUT, IO_OUTPUT }
iomode;

static gfc_use_rename *gfc_rename_list;
static pointer_info *pi_root;
static int symbol_number;       /* Counter for assigning symbol numbers */

/* Tells mio_expr_ref to make symbols for unused equivalence members.  */
static bool in_load_equiv;

int errno;

locus gfc_current_locus;

gfc_namespace *gfc_current_ns;

gfc_option_t gfc_option;

int gfc_default_integer_kind = 4;
int gfc_default_real_kind    = 4;

/*****************************************************************/

/* Module reading and writing.  */

typedef enum
{
  ATOM_NAME, ATOM_LPAREN, ATOM_RPAREN, ATOM_INTEGER, ATOM_STRING
}
atom_type;

static atom_type last_atom;

/* The name buffer must be at least as long as a symbol name.  Right
   now it's not clear how we're going to store numeric constants--
   probably as a hexadecimal string, since this will allow the exact
   number to be preserved (this can't be done by a decimal
   representation).  Worry about that later.  TODO!  */

#define MAX_ATOM_SIZE 100

static int atom_int;
static char *atom_string, atom_name[MAX_ATOM_SIZE];

/* List of include file search directories.  */
static gfc_directorylist *include_dirs, *intrinsic_modules_dirs;

static int terminal_width, buffer_flag, errors, warnings;

static unsigned int g77_newargc;
static struct cl_decoded_option *g77_new_decoded_options;

/*****************************************************************/

/***********************/
/***** xstrerror.c *****/
/***********************/
#define ERRSTR_FMT "undocumented error #%d"

static char xstrerror_buf[sizeof ERRSTR_FMT + 20];

static void bad_module (const char *) ATTRIBUTE_NORETURN;

static atom_type parse_atom (void);

/* Like strerror, but result is never a null pointer.  */

char *
xstrerror (int errnum)
{
  char *errstr;
#ifdef VMS
  char *(*vmslib_strerror) (int,...);

  /* Override any possibly-conflicting declaration from system header.  */
  vmslib_strerror = (char *(*) (int,...)) strerror;
  /* Second argument matters iff first is EVMSERR, but it's simpler to
     pass it unconditionally.  `vaxc$errno' is declared in <errno.h>
     and maintained by the run-time library in parallel to `errno'.
     We assume that `errnum' corresponds to the last value assigned to
     errno by the run-time library, hence vaxc$errno will be relevant.  */
  errstr = (*vmslib_strerror) (errnum, vaxc$errno);
#else
  errstr = strerror (errnum);
#endif

  /* If `errnum' is out of range, result might be NULL.  We'll fix that.  */
  if (!errstr)
    {
      sprintf (xstrerror_buf, ERRSTR_FMT, errnum);
      errstr = xstrerror_buf;
    }
  return errstr;
}

/*******************/
/***** error.c *****/
/*******************/

static int suppress_errors = 0;

static int warnings_not_errors = 0;

static int inhibit_warnings    = 0;

static int warnings_are_errors = 0;

static gfc_error_buf error_buffer, warning_buffer, *cur_error_buffer;

/* Add a single character to the error buffer or output depending on
   buffer_flag.  */

static void
error_char (char c)
{
  if (buffer_flag)
    {
      if (cur_error_buffer->index >= cur_error_buffer->allocated)
        {
          cur_error_buffer->allocated = cur_error_buffer->allocated
                                      ? cur_error_buffer->allocated * 2 : 1000;
          cur_error_buffer->message = XRESIZEVEC (char, cur_error_buffer->message,
                                                  cur_error_buffer->allocated);
        }
      cur_error_buffer->message[cur_error_buffer->index++] = c;
    }
  else
    {
      if (c != 0)
        {
          /* We build up complete lines before handing things
             over to the library in order to speed up error printing.  */
          static char *line;
          static size_t allocated = 0, index = 0;

          if (index + 1 >= allocated)
            {
              allocated = allocated ? allocated * 2 : 1000;
              line = XRESIZEVEC (char, line, allocated);
            }
          line[index++] = c;
          if (c == '\n')
            {
              line[index] = '\0';
              fputs (line, stderr);
              index = 0;
            }
        }
    }
}


/* Copy a string to wherever it needs to go.  */

static void
error_string (const char *p)
{
  while (*p)
    error_char (*p++);
}


/* Print a formatted integer to the error buffer or output.  */

#define IBUF_LEN 60

static void
error_uinteger (unsigned long int i)
{
  char *p, int_buf[IBUF_LEN];

  p = int_buf + IBUF_LEN - 1;
  *p-- = '\0';

  if (i == 0)
    *p-- = '0';

  while (i > 0)
    {
      *p-- = i % 10 + '0';
      i = i / 10;
    }

  error_string (p + 1);
}

static void
error_integer (long int i)
{
  unsigned long int u;

  if (i < 0)
    {
      u = (unsigned long int) -i;
      error_char ('-');
    }
  else
    u = i;

  error_uinteger (u);
}

#ifdef _RESOLUTION__
static void
print_wide_char_into_buffer (gfc_char_t c, char *buf)
{
  static const char xdigit[16] = { '0', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };

  if (gfc_wide_is_printable (c))
    {
      buf[1] = '\0';
      buf[0] = (unsigned char) c;
    }
  else if (c < ((gfc_char_t) 1 << 8))
    {
      buf[4] = '\0';
      buf[3] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[2] = xdigit[c & 0x0F];

      buf[1] = 'x';
      buf[0] = '\\';
    }
  else if (c < ((gfc_char_t) 1 << 16))
    {
      buf[6] = '\0';
      buf[5] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[4] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[3] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[2] = xdigit[c & 0x0F];

      buf[1] = 'u';
      buf[0] = '\\';
    }
  else
    {
      buf[10] = '\0';
      buf[9] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[8] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[7] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[6] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[5] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[4] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[3] = xdigit[c & 0x0F];
      c = c >> 4;
      buf[2] = xdigit[c & 0x0F];

      buf[1] = 'U';
      buf[0] = '\\';
    }
}

static char wide_char_print_buffer[11];

const char *
gfc_print_wide_char (gfc_char_t c)
{
  print_wide_char_into_buffer (c, wide_char_print_buffer);
  return wide_char_print_buffer;
}
#endif


static void error_printf (const char *, ...) ATTRIBUTE_GCC_GFC(1,2);

#ifdef _RESOLUTION__
static void
show_locus (locus *loc, int c1, int c2)
{
  gfc_linebuf *lb;
  gfc_file *f;
  gfc_char_t c, *p;
  int i, offset, cmax;

  /* TODO: Either limit the total length and number of included files
     displayed or add buffering of arbitrary number of characters in
     error messages.  */

  /* Write out the error header line, giving the source file and error
     location (in GNU standard "[file]:[line].[column]:" format),
     followed by an "included by" stack and a blank line.  This header
     format is matched by a testsuite parser defined in
     lib/gfortran-dg.exp.  */

  lb = loc->lb;
  f = lb->file;

  error_string (f->filename);
  error_char (':');

  error_integer (LOCATION_LINE (lb->location));

  if ((c1 > 0) || (c2 > 0))
    error_char ('.');

  if (c1 > 0)
    error_integer (c1);

  if ((c1 > 0) && (c2 > 0))
    error_char ('-');

  if (c2 > 0)
    error_integer (c2);

  error_char (':');
  error_char ('\n');

  for (;;)
    {
      i = f->inclusion_line;

      f = f->up;
      if (f == NULL) break;

      error_printf ("    Included at %s:%d:", f->filename, i);
    }

  error_char ('\n');

  /* Calculate an appropriate horizontal offset of the source line in
     order to get the error locus within the visible portion of the
     line.  Note that if the margin of 5 here is changed, the
     corresponding margin of 10 in show_loci should be changed.  */

  offset = 0;

  /* If the two loci would appear in the same column, we shift
     '2' one column to the right, so as to print '12' rather than
     just '1'.  We do this here so it will be accounted for in the
     margin calculations.  */

  if (c1 == c2)
    c2 += 1;

  cmax = (c1 < c2) ? c2 : c1;
  if (cmax > terminal_width - 5)
    offset = cmax - terminal_width + 5;

  /* Show the line itself, taking care not to print more than what can
     show up on the terminal.  Tabs are converted to spaces, and
     nonprintable characters are converted to a "\xNN" sequence.  */

  /* TODO: Although setting i to the terminal width is clever, it fails
     to work correctly when nonprintable characters exist.  A better
     solution should be found.  */

  p = &(lb->line[offset]);
  i = gfc_wide_strlen (p);
  if (i > terminal_width)
    i = terminal_width - 1;

  for (; i > 0; i--)
    {
      static char buffer[11];

      c = *p++;
      if (c == '\t')
        c = ' ';

      print_wide_char_into_buffer (c, buffer);
      error_string (buffer);
    }

  error_char ('\n');

  /* Show the '1' and/or '2' corresponding to the column of the error
     locus.  Note that a value of -1 for c1 or c2 will simply cause
     the relevant number not to be printed.  */

  c1 -= offset;
  c2 -= offset;

  for (i = 0; i <= cmax; i++)
    {
      if (i == c1)
        error_char ('1');
      else if (i == c2)
        error_char ('2');
      else
        error_char (' ');
    }

  error_char ('\n');

}

/* As part of printing an error, we show the source lines that caused
   the problem.  We show at least one, and possibly two loci; the two
   loci may or may not be on the same source line.  */

static void
show_loci (locus *l1, locus *l2)
{
  int m, c1, c2;

  if (l1 == NULL || l1->lb == NULL)
    {
      error_printf ("<During initialization>\n");
      return;
    }

  /* While calculating parameters for printing the loci, we consider possible
     reasons for printing one per line.  If appropriate, print the loci
     individually; otherwise we print them both on the same line.  */

  c1 = l1->nextc - l1->lb->line;
  if (l2 == NULL)
    {
      show_locus (l1, c1, -1);
      return;
    }

  c2 = l2->nextc - l2->lb->line;

  if (c1 < c2)
    m = c2 - c1;
  else
    m = c1 - c2;

  /* Note that the margin value of 10 here needs to be less than the
     margin of 5 used in the calculation of offset in show_locus.  */

  if (l1->lb != l2->lb || m > terminal_width - 10)
    {
      show_locus (l1, c1, -1);
      show_locus (l2, -1, c2);
      return;
    }

  show_locus (l1, c1, c2);

  return;
}
#endif

/* Workhorse for the error printing subroutines.  This subroutine is
   inspired by g77's error handling and is similar to printf() with
   the following %-codes:

   %c Character, %d or %i Integer, %s String, %% Percent
   %L  Takes locus argument
   %C  Current locus (no argument)

   If a locus pointer is given, the actual source line is printed out
   and the column is indicated.  Since we want the error message at
   the bottom of any source file information, we must scan the
   argument list twice -- once to determine whether the loci are
   present and record this for printing, and once to print the error
   message after and loci have been printed.  A maximum of two locus
   arguments are permitted.

   This function is also called (recursively) by show_locus in the
   case of included files; however, as show_locus does not resupply
   any loci, the recursion is at most one level deep.  */

#define MAX_ARGS 10

static void ATTRIBUTE_GCC_GFC(2,0)
error_print (const char *type, const char *format0, va_list argp)
{
  enum { TYPE_CURRENTLOC, TYPE_LOCUS, TYPE_INTEGER, TYPE_UINTEGER,
         TYPE_LONGINT, TYPE_ULONGINT, TYPE_CHAR, TYPE_STRING,
         NOTYPE };
  struct
  {
    int type;
    int pos;
    union
    {
      int intval;
      unsigned int uintval;
      long int longintval;
      unsigned long int ulongintval;
      char charval;
      const char * stringval;
    } u;
  } arg[MAX_ARGS], spec[MAX_ARGS];
  /* spec is the array of specifiers, in the same order as they
     appear in the format string.  arg is the array of arguments,
     in the same order as they appear in the va_list.  */

  char c;
  int i, n, have_l1, pos, maxpos;
  locus *l1, *l2, *loc;
  const char *format;

  loc = l1 = l2 = NULL;

  have_l1 = 0;
  pos = -1;
  maxpos = -1;

  n = 0;
  format = format0;

  for (i = 0; i < MAX_ARGS; i++)
    {
      arg[i].type = NOTYPE;
      spec[i].pos = -1;
    }

  /* First parse the format string for position specifiers.  */
  while (*format)
    {
      c = *format++;
      if (c != '%')
        continue;

      if (*format == '%')
        {
          format++;
          continue;
        }

      if (ISDIGIT (*format))
        {
          /* This is a position specifier.  For example, the number
             12 in the format string "%12$d", which specifies the third
             argument of the va_list, formatted in %d format.
             For details, see "man 3 printf".  */
          pos = atoi(format) - 1;
          gcc_assert (pos >= 0);
          while (ISDIGIT(*format))
            format++;
          gcc_assert (*format++ == '$');
        }
      else
        pos++;

      c = *format++;

      if (pos > maxpos)
        maxpos = pos;

      switch (c)
        {
          case 'C':
            arg[pos].type = TYPE_CURRENTLOC;
            break;

          case 'L':
            arg[pos].type = TYPE_LOCUS;
            break;

          case 'd':
          case 'i':
            arg[pos].type = TYPE_INTEGER;
            break;

          case 'u':
            arg[pos].type = TYPE_UINTEGER;
            break;

          case 'l':
            c = *format++;
            if (c == 'u')
              arg[pos].type = TYPE_ULONGINT;
            else if (c == 'i' || c == 'd')
              arg[pos].type = TYPE_LONGINT;
            else
              gcc_unreachable ();
            break;

          case 'c':
            arg[pos].type = TYPE_CHAR;
            break;

          case 's':
            arg[pos].type = TYPE_STRING;
            break;

          default:
            gcc_unreachable ();
        }

      spec[n++].pos = pos;
    }

  /* Then convert the values for each %-style argument.  */
  for (pos = 0; pos <= maxpos; pos++)
    {
      gcc_assert (arg[pos].type != NOTYPE);
      switch (arg[pos].type)
        {
          case TYPE_CURRENTLOC:
            loc = &gfc_current_locus;
            /* Fall through.  */

          case TYPE_LOCUS:
            if (arg[pos].type == TYPE_LOCUS)
              loc = va_arg (argp, locus *);

            if (have_l1)
              {
                l2 = loc;
                arg[pos].u.stringval = "(2)";
              }
            else
              {
                l1 = loc;
                have_l1 = 1;
                arg[pos].u.stringval = "(1)";
              }
            break;

          case TYPE_INTEGER:
            arg[pos].u.intval = va_arg (argp, int);
            break;

          case TYPE_UINTEGER:
            arg[pos].u.uintval = va_arg (argp, unsigned int);
            break;

          case TYPE_LONGINT:
            arg[pos].u.longintval = va_arg (argp, long int);
            break;

          case TYPE_ULONGINT:
            arg[pos].u.ulongintval = va_arg (argp, unsigned long int);
            break;

          case TYPE_CHAR:
            arg[pos].u.charval = (char) va_arg (argp, int);
            break;
          case TYPE_STRING:
            arg[pos].u.stringval = (const char *) va_arg (argp, char *);
            break;

          default:
            gcc_unreachable ();
        }
    }

  for (n = 0; spec[n].pos >= 0; n++)
    spec[n].u = arg[spec[n].pos].u;

  /* Show the current loci if we have to.  */
#ifdef _RESOLUTION__
  if (have_l1)
    show_loci (l1, l2);
#endif

  if (*type)
    {
      error_string (type);
      error_char (' ');
    }

  have_l1 = 0;
  format = format0;
  n = 0;

  for (; *format; format++)
    {
      if (*format != '%')
        {
          error_char (*format);
          continue;
        }

      format++;
      if (ISDIGIT (*format))
        {
          /* This is a position specifier.  See comment above.  */
          while (ISDIGIT (*format))
            format++;

          /* Skip over the dollar sign.  */
          format++;
        }

      switch (*format)
        {
        case '%':
          error_char ('%');
          break;

        case 'c':
          error_char (spec[n++].u.charval);
          break;

        case 's':
        case 'C':               /* Current locus */
        case 'L':               /* Specified locus */
          error_string (spec[n++].u.stringval);
          break;

        case 'd':
        case 'i':
          error_integer (spec[n++].u.intval);
          break;

        case 'u':
          error_uinteger (spec[n++].u.uintval);
          break;

        case 'l':
          format++;
          if (*format == 'u')
            error_uinteger (spec[n++].u.ulongintval);
          else
            error_integer (spec[n++].u.longintval);
          break;

        }
    }

  error_char ('\n');
}


/* Wrapper for error_print().  */

static void
error_printf (const char *gmsgid, ...)
{
  va_list argp;

  va_start (argp, gmsgid);
  error_print ("", _(gmsgid), argp);
  va_end (argp);
}


/* Increment the number of errors, and check whether too many have
   been printed.  */

static void
gfc_increment_error_count (void)
{
  errors++;
  if ((gfc_option.max_errors != 0) && (errors >= gfc_option.max_errors))
    gfc_fatal_error ("Error count reached limit of %d.", gfc_option.max_errors);
}


/* Possibly issue a warning/error about use of a nonstandard (or deleted)
   feature.  An error/warning will be issued if the currently selected
   standard does not contain the requested bits.  Return FAILURE if
   an error is generated.  */

gfc_try
gfc_notify_std (int std, const char *gmsgid, ...)
{
  va_list argp;
  bool warning;

  warning = ((gfc_option.warn_std & std) != 0) && !inhibit_warnings;
  if ((gfc_option.allow_std & std) != 0 && !warning)
    return SUCCESS;

  if (suppress_errors)
    return warning ? SUCCESS : FAILURE;

  cur_error_buffer = warning ? &warning_buffer : &error_buffer;
  cur_error_buffer->flag = 1;
  cur_error_buffer->index = 0;

  va_start (argp, gmsgid);
  if (warning)
    error_print (_("Warning:"), _(gmsgid), argp);
  else
    error_print (_("Error:"), _(gmsgid), argp);
  va_end (argp);

  error_char ('\0');

  if (buffer_flag == 0)
    {
      if (warning && !warnings_are_errors)
        warnings++;
      else
        gfc_increment_error_count();
    }

  return (warning && !warnings_are_errors) ? SUCCESS : FAILURE;
}


/* Issue an error.  */

void
gfc_error (const char *gmsgid, ...)
{
  va_list argp;

  if (warnings_not_errors)
    goto warning;

  if (suppress_errors)
    return;

  error_buffer.flag = 1;
  error_buffer.index = 0;
  cur_error_buffer = &error_buffer;

  va_start (argp, gmsgid);
  error_print (_("Error:"), _(gmsgid), argp);
  va_end (argp);

  error_char ('\0');

  if (buffer_flag == 0)
    gfc_increment_error_count();

  return;

warning:

  if (inhibit_warnings)
    return;

  warning_buffer.flag = 1;
  warning_buffer.index = 0;
  cur_error_buffer = &warning_buffer;

  va_start (argp, gmsgid);
  error_print (_("Warning:"), _(gmsgid), argp);
  va_end (argp);

  error_char ('\0');

  if (buffer_flag == 0)
  {
    warnings++;
    if (warnings_are_errors)
      gfc_increment_error_count();
  }
}


/* Fatal error, never returns.  */

void
gfc_fatal_error (const char *gmsgid, ...)
{
  va_list argp;

  buffer_flag = 0;

printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
fflush(stdout);

  va_start (argp, gmsgid);
  error_print (_("Fatal Error:"), _(gmsgid), argp);
  va_end (argp);

  exit (FATAL_EXIT_CODE);
}


/* This shouldn't happen... but sometimes does.  */

void
gfc_internal_error (const char *format, ...)
{
  va_list argp;

  buffer_flag = 0;

  va_start (argp, format);
#ifdef _RESOLUTION__
  show_loci (&gfc_current_locus, NULL);
#endif
  error_printf ("Internal Error at (1):");

  error_print ("", format, argp);
  va_end (argp);

  exit (ICE_EXIT_CODE);
}

/*****************/
/***** cpp.c *****/
/*****************/

void
gfc_cpp_init_options (unsigned int decoded_options_count,
                      struct cl_decoded_option *decoded_options ATTRIBUTE_UNUSED)
{
  /* Do not create any objects from libcpp here. If no
     preprocessing is requested, this would be wasted
     time and effort.

     See gfc_cpp_post_options() instead.  */

  gfc_cpp_option.temporary_filename = NULL;
  gfc_cpp_option.output_filename = NULL;
  gfc_cpp_option.preprocess_only = 0;
  gfc_cpp_option.discard_comments = 1;
  gfc_cpp_option.discard_comments_in_macro_exp = 1;
  gfc_cpp_option.print_include_names = 0;
  gfc_cpp_option.no_line_commands = 0;
  gfc_cpp_option.dump_macros = '\0';
  gfc_cpp_option.dump_includes = 0;
  gfc_cpp_option.working_directory = -1;
  gfc_cpp_option.no_predefined = 0;
  gfc_cpp_option.standard_include_paths = 1;
  gfc_cpp_option.verbose = 0;
  gfc_cpp_option.deps = 0;
  gfc_cpp_option.deps_skip_system = 0;
  gfc_cpp_option.deps_phony = 0;
  gfc_cpp_option.deps_missing_are_generated = 0;
  gfc_cpp_option.deps_filename = NULL;
  gfc_cpp_option.deps_filename_user = NULL;

  gfc_cpp_option.multilib = NULL;
  gfc_cpp_option.prefix = NULL;
  gfc_cpp_option.sysroot = NULL;
#ifdef _RESOLUTION_
  gfc_cpp_option.deferred_opt = XNEWVEC (gfc_cpp_deferred_opt_t,
                                         decoded_options_count);
#endif
  gfc_cpp_option.deferred_opt_count = 0;
}

/********************/
/***** option.c *****/
/********************/

const char *gfc_source_file;
gfc_option_t gfc_option;


/* Set flags that control warnings and errors for different
   Fortran standards to their default values.  Keep in sync with
   libgfortran/runtime/compile_options.c (init_compile_options).  */

static void
set_default_std_flags (void)
{
  gfc_option.allow_std = GFC_STD_F95_OBS | GFC_STD_F95_DEL
    | GFC_STD_F2003 | GFC_STD_F2008 | GFC_STD_F95 | GFC_STD_F77
    | GFC_STD_F2008_OBS | GFC_STD_F2008_TS | GFC_STD_GNU | GFC_STD_LEGACY;
  gfc_option.warn_std = GFC_STD_F95_DEL | GFC_STD_LEGACY;
}

/* Get ready for options handling. Keep in sync with
   libgfortran/runtime/compile_options.c (init_compile_options). */

void
gfc_init_options (unsigned int decoded_options_count,
                  struct cl_decoded_option *decoded_options)
{
  gfc_source_file = NULL;
  gfc_option.module_dir = NULL;
  gfc_option.source_form = FORM_UNKNOWN;
  gfc_option.fixed_line_length = 72;
  gfc_option.free_line_length = 132;
  gfc_option.max_continue_fixed = 255;
  gfc_option.max_continue_free = 255;
  gfc_option.max_identifier_length = GFC_MAX_SYMBOL_LEN;
  gfc_option.max_subrecord_length = 0;
  gfc_option.flag_max_array_constructor = 65535;
  gfc_option.convert = GFC_CONVERT_NATIVE;
  gfc_option.record_marker = 0;
  gfc_option.dump_fortran_original = 0;
  gfc_option.dump_fortran_optimized = 0;

  gfc_option.warn_aliasing = 0;
  gfc_option.warn_ampersand = 0;
  gfc_option.warn_character_truncation = 0;
  gfc_option.warn_array_temp = 0;
  gfc_option.gfc_warn_conversion = 0;
  gfc_option.warn_conversion_extra = 0;
  gfc_option.warn_function_elimination = 0;
  gfc_option.warn_implicit_interface = 0;
  gfc_option.warn_line_truncation = 0;
  gfc_option.warn_surprising = 0;
  gfc_option.warn_tabs = 1;
  gfc_option.warn_underflow = 1;
  gfc_option.warn_intrinsic_shadow = 0;
  gfc_option.warn_intrinsics_std = 0;
  gfc_option.warn_align_commons = 1;
  gfc_option.warn_real_q_constant = 0;
  gfc_option.warn_unused_dummy_argument = 0;
  gfc_option.max_errors = 25;

  gfc_option.flag_all_intrinsics = 0;
  gfc_option.flag_default_double = 0;
  gfc_option.flag_default_integer = 0;
  gfc_option.flag_default_real = 0;
  gfc_option.flag_integer4_kind = 0;
  gfc_option.flag_real4_kind = 0;
  gfc_option.flag_real8_kind = 0;
  gfc_option.flag_dollar_ok = 0;
  gfc_option.flag_underscoring = 1;
  gfc_option.flag_whole_file = 1;
  gfc_option.flag_f2c = 0;
  gfc_option.flag_second_underscore = -1;
  gfc_option.flag_implicit_none = 0;

  /* Default value of flag_max_stack_var_size is set in gfc_post_options.  */
  gfc_option.flag_max_stack_var_size = -2;
  gfc_option.flag_stack_arrays = -1;

  gfc_option.flag_range_check = 1;
  gfc_option.flag_pack_derived = 0;
  gfc_option.flag_repack_arrays = 0;
  gfc_option.flag_preprocessed = 0;
  gfc_option.flag_automatic = 1;
  gfc_option.flag_backslash = 0;
  gfc_option.flag_module_private = 0;
  gfc_option.flag_backtrace = 1;
  gfc_option.flag_allow_leading_underscore = 0;
  gfc_option.flag_external_blas = 0;
  gfc_option.blas_matmul_limit = 30;
  gfc_option.flag_cray_pointer = 0;
  gfc_option.flag_d_lines = -1;
  gfc_option.gfc_flag_openmp = 0;
  gfc_option.flag_sign_zero = 1;
  gfc_option.flag_recursive = 0;
  gfc_option.flag_init_integer = GFC_INIT_INTEGER_OFF;
  gfc_option.flag_init_integer_value = 0;
  gfc_option.flag_init_real = GFC_INIT_REAL_OFF;
  gfc_option.flag_init_logical = GFC_INIT_LOGICAL_OFF;
  gfc_option.flag_init_character = GFC_INIT_CHARACTER_OFF;
  gfc_option.flag_init_character_value = (char)0;
  gfc_option.flag_align_commons = 1;
  gfc_option.flag_protect_parens = -1;
  gfc_option.flag_realloc_lhs = -1;
  gfc_option.flag_aggressive_function_elimination = 0;
  gfc_option.flag_frontend_optimize = -1;

  gfc_option.fpe = 0;
  gfc_option.rtcheck = 0;
  gfc_option.coarray = GFC_FCOARRAY_NONE;

  set_default_std_flags ();

  /* Initialize cpp-related options.  */
  gfc_cpp_init_options (decoded_options_count, decoded_options);
}

/************************/
/***** intrinsic.c *****/
/************************/

static gfc_intrinsic_sym *functions, *subroutines, *conversion, *next_sym;
static gfc_intrinsic_sym *char_conversions;
static gfc_intrinsic_arg *next_arg;

static int nfunc, nsub, nargs, nconv, ncharconv;

/* Locate an intrinsic symbol given a base pointer, number of elements
   in the table and a pointer to a name.  Returns the NULL pointer if
   a name is not found.  */

static gfc_intrinsic_sym *
find_sym (gfc_intrinsic_sym *start, int n, const char *name)
{
  /* name may be a user-supplied string, so we must first make sure
     that we're comparing against a pointer into the global string
     table.  */
  const char *p = gfc_get_string (name);

  while (n > 0)
    {
      if (p == start->name)
        return start;

      start++;
      n--;
    }

  return NULL;
}

/* Given a name, find a function in the intrinsic function table.
   Returns NULL if not found.  */

gfc_intrinsic_sym *
gfc_find_function (const char *name)
{
  gfc_intrinsic_sym *sym;

  sym = find_sym (functions, nfunc, name);
  if (!sym || sym->from_module)
    sym = find_sym (conversion, nconv, name);

  return (!sym || sym->from_module) ? NULL : sym;
}


/******************/
/***** decl.c *****/
/******************/

/* Free a gfc_data_variable structure and everything beneath it.  */

static void
free_variable (gfc_data_variable *p)
{
  gfc_data_variable *q;

  for (; p; p = q)
    {
      q = p->next;
      gfc_free_expr (p->expr);
      gfc_free_iterator (&p->iter, 0);
      free_variable (p->list);
      free (p);
    }
}


/* Free a gfc_data_value structure and everything beneath it.  */

static void
free_value (gfc_data_value *p)
{
  gfc_data_value *q;

  for (; p; p = q)
    {
      q = p->next;
      mpz_clear (p->repeat);
      gfc_free_expr (p->expr);
      free (p);
    }
}

/* Free a list of gfc_data structures.  */

void
gfc_free_data (gfc_data *p)
{
  gfc_data *q;

  for (; p; p = q)
    {
      q = p->next;
      free_variable (p->var);
      free_value (p->value);
      free (p);
    }
}

/***********************/
/***** interface.c *****/
/***********************/


/* Free a singly linked list of gfc_interface structures.  */

void
gfc_free_interface (gfc_interface *intr)
{
  gfc_interface *next;

  for (; intr; intr = next)
    {
      next = intr->next;
      free (intr);
    }
}

/* Gets rid of a formal argument list.  We do not free symbols.
   Symbols are freed when a namespace is freed.  */

void
gfc_free_formal_arglist (gfc_formal_arglist *p)
{
  gfc_formal_arglist *q;

  for (; p; p = q)
    {
      q = p->next;
      free (p);
    }
}


/*******************/
/***** array.c *****/
/*******************/

/* Copy an array reference structure.  */

gfc_array_ref *
gfc_copy_array_ref (gfc_array_ref *src)
{
  gfc_array_ref *dest;
  int i;

  if (src == NULL)
    return NULL;

  dest = gfc_get_array_ref ();

  *dest = *src;

  for (i = 0; i < GFC_MAX_DIMENSIONS; i++)
    {
      dest->start[i] = gfc_copy_expr (src->start[i]);
      dest->end[i] = gfc_copy_expr (src->end[i]);
      dest->stride[i] = gfc_copy_expr (src->stride[i]);
    }

  dest->offset = gfc_copy_expr (src->offset);

  return dest;
}


/* Free all of the expressions associated with array bounds
   specifications.  */

void
gfc_free_array_spec (gfc_array_spec *as)
{
  int i;

  if (as == NULL)
    return;

  for (i = 0; i < as->rank + as->corank; i++)
    {
      gfc_free_expr (as->lower[i]);
      gfc_free_expr (as->upper[i]);
    }

  free (as);
}


/* Copy an iterator structure.  */

gfc_iterator *
gfc_copy_iterator (gfc_iterator *src)
{
  gfc_iterator *dest;

  if (src == NULL)
    return NULL;

  dest = gfc_get_iterator ();

  dest->var = gfc_copy_expr (src->var);
  dest->start = gfc_copy_expr (src->start);
  dest->end = gfc_copy_expr (src->end);
  dest->step = gfc_copy_expr (src->step);

  return dest;
}


/************************/
/***** splay-tree.c *****/
/************************/

#ifndef _WIN64
#if 0
  typedef unsigned long int libi_uhostptr_t;
  typedef long int libi_shostptr_t;
#endif
#else
#ifdef __GNUC__
  __extension__
#endif
  typedef unsigned long long libi_uhostptr_t;
#ifdef __GNUC__
  __extension__
#endif
  typedef long long libi_shostptr_t;
#endif

#if 0
typedef libi_uhostptr_t splay_tree_value;
#endif

extern splay_tree splay_tree_new_with_allocator (splay_tree_compare_fn,
                                                 splay_tree_delete_key_fn,
                                                 splay_tree_delete_value_fn,
                                                 splay_tree_allocate_fn,
                                                 splay_tree_deallocate_fn,
                                                 void *);
extern splay_tree splay_tree_new_typed_alloc (splay_tree_compare_fn,
                                              splay_tree_delete_key_fn,
                                              splay_tree_delete_value_fn,
                                              splay_tree_allocate_fn,
                                              splay_tree_allocate_fn,
                                              splay_tree_deallocate_fn,
                                              void *);




/* Deallocate NODE (a member of SP), and all its sub-trees.  */

static void
splay_tree_delete_helper (splay_tree sp, splay_tree_node node)
{
  splay_tree_node pending = 0;
  splay_tree_node active = 0;

  if (!node)
    return;

#define KDEL(x)  if (sp->delete_key) (*sp->delete_key)(x);
#define VDEL(x)  if (sp->delete_value) (*sp->delete_value)(x);

  KDEL (node->key);
  VDEL (node->value);

  /* We use the "key" field to hold the "next" pointer.  */
  node->key = (splay_tree_key)pending;
  pending = (splay_tree_node)node;

  /* Now, keep processing the pending list until there aren't any
     more.  This is a little more complicated than just recursing, but
     it doesn't toast the stack for large trees.  */

  while (pending)
    {
      active = pending;
      pending = 0;
      while (active)
        {
          splay_tree_node temp;

          /* active points to a node which has its key and value
             deallocated, we just need to process left and right.  */

          if (active->left)
            {
              KDEL (active->left->key);
              VDEL (active->left->value);
              active->left->key = (splay_tree_key)pending;
              pending = (splay_tree_node)(active->left);
            }
          if (active->right)
            {
              KDEL (active->right->key);
              VDEL (active->right->value);
              active->right->key = (splay_tree_key)pending;
              pending = (splay_tree_node)(active->right);
            }

          temp = active;
          active = (splay_tree_node)(temp->key);
          (*sp->deallocate) ((char*) temp, sp->allocate_data);
        }
    }
#undef KDEL
#undef VDEL
}


/* Rotate the edge joining the left child N with its parent P.  PP is the
   grandparents' pointer to P.  */

static inline void
rotate_left_splay (splay_tree_node *pp, splay_tree_node p, splay_tree_node n)
{
  splay_tree_node tmp;
  tmp = n->right;
  n->right = p;
  p->left = tmp;
  *pp = n;
}

/* Rotate the edge joining the right child N with its parent P.  PP is the
   grandparents' pointer to P.  */

static inline void
rotate_right_splay (splay_tree_node *pp, splay_tree_node p, splay_tree_node n)
{
  splay_tree_node tmp;
  tmp = n->left;
  n->left = p;
  p->right = tmp;
  *pp = n;
}


/* Bottom up splay of key.  */

static void
splay_tree_splay (splay_tree sp, splay_tree_key key)
{
  if (sp->root == 0)
    return;

  do {
    int cmp1, cmp2;
    splay_tree_node n, c;

    n = sp->root;
    cmp1 = (*sp->comp) (key, n->key);

    /* Found.  */
    if (cmp1 == 0)
      return;

    /* Left or right?  If no child, then we're done.  */
    if (cmp1 < 0)
      c = n->left;
    else
      c = n->right;
    if (!c)
      return;

    /* Next one left or right?  If found or no child, we're done
       after one rotation.  */
    cmp2 = (*sp->comp) (key, c->key);
    if (cmp2 == 0
        || (cmp2 < 0 && !c->left)
        || (cmp2 > 0 && !c->right))
      {
        if (cmp1 < 0)
          rotate_left_splay (&sp->root, n, c);
        else
          rotate_right_splay (&sp->root, n, c);
        return;
      }

    /* Now we have the four cases of double-rotation.  */
    if (cmp1 < 0 && cmp2 < 0)
      {
        rotate_left_splay (&n->left, c, c->left);
        rotate_left_splay (&sp->root, n, n->left);
      }
    else if (cmp1 > 0 && cmp2 > 0)
      {
        rotate_right_splay (&n->right, c, c->right);
        rotate_right_splay (&sp->root, n, n->right);
      }
    else if (cmp1 < 0 && cmp2 > 0)
      {
        rotate_right_splay (&n->left, c, c->right);
        rotate_left_splay (&sp->root, n, n->left);
      }
    else if (cmp1 > 0 && cmp2 < 0)
      {
        rotate_left_splay (&n->right, c, c->left);
        rotate_right_splay (&sp->root, n, n->right);
      }
  } while (1);
}


/* Return the node in SP with the greatest key.  */

splay_tree_node
splay_tree_max (splay_tree sp)
{
  splay_tree_node n = sp->root;

  if (!n)
    return NULL;

  while (n->right)
    n = n->right;

  return n;
}



/* An allocator and deallocator based on xmalloc.  */
static void *
splay_tree_xmalloc_allocate (int size, void *data ATTRIBUTE_UNUSED)
{
  return (void *) xmalloc (size);
}

static void
splay_tree_xmalloc_deallocate (void *object, void *data ATTRIBUTE_UNUSED)
{
  free (object);
}

/* Allocate a new splay tree, using COMPARE_FN to compare nodes,
   DELETE_KEY_FN to deallocate keys, and DELETE_VALUE_FN to deallocate
   values.  Use xmalloc to allocate the splay tree structure, and any
   nodes added.  */

splay_tree
splay_tree_new (splay_tree_compare_fn compare_fn,
                splay_tree_delete_key_fn delete_key_fn,
                splay_tree_delete_value_fn delete_value_fn)
{
  return (splay_tree_new_with_allocator
          (compare_fn, delete_key_fn, delete_value_fn,
           splay_tree_xmalloc_allocate, splay_tree_xmalloc_deallocate, 0));
}


/* Allocate a new splay tree, using COMPARE_FN to compare nodes,
   DELETE_KEY_FN to deallocate keys, and DELETE_VALUE_FN to deallocate
   values.  */

splay_tree
splay_tree_new_with_allocator (splay_tree_compare_fn compare_fn,
                               splay_tree_delete_key_fn delete_key_fn,
                               splay_tree_delete_value_fn delete_value_fn,
                               splay_tree_allocate_fn allocate_fn,
                               splay_tree_deallocate_fn deallocate_fn,
                               void *allocate_data)
{
  return
    splay_tree_new_typed_alloc (compare_fn, delete_key_fn, delete_value_fn,
                                allocate_fn, allocate_fn, deallocate_fn,
                                allocate_data);
}


splay_tree
splay_tree_new_typed_alloc (splay_tree_compare_fn compare_fn,
                            splay_tree_delete_key_fn delete_key_fn,
                            splay_tree_delete_value_fn delete_value_fn,
                            splay_tree_allocate_fn tree_allocate_fn,
                            splay_tree_allocate_fn node_allocate_fn,
                            splay_tree_deallocate_fn deallocate_fn,
                            void * allocate_data)
{
  splay_tree sp = (splay_tree) (*tree_allocate_fn)
    (sizeof (struct splay_tree_s), allocate_data);

  sp->root = 0;
  sp->comp = compare_fn;
  sp->delete_key = delete_key_fn;
  sp->delete_value = delete_value_fn;
  sp->allocate = node_allocate_fn;
  sp->deallocate = deallocate_fn;
  sp->allocate_data = allocate_data;

  return sp;
}


/* Deallocate SP.  */

void
splay_tree_delete (splay_tree sp)
{
  splay_tree_delete_helper (sp, sp->root);
  (*sp->deallocate) ((char*) sp, sp->allocate_data);
}


/* Insert a new node (associating KEY with DATA) into SP.  If a
   previous node with the indicated KEY exists, its data is replaced
   with the new value.  Returns the new node.  */

splay_tree_node
splay_tree_insert (splay_tree sp, splay_tree_key key, splay_tree_value value)
{
  int comparison = 0;

  splay_tree_splay (sp, key);

  if (sp->root)
    comparison = (*sp->comp)(sp->root->key, key);

  if (sp->root && comparison == 0)
    {
      /* If the root of the tree already has the indicated KEY, just
         replace the value with VALUE.  */
      if (sp->delete_value)
        (*sp->delete_value)(sp->root->value);
      sp->root->value = value;
    }
  else
    {
      /* Create a new node, and insert it at the root.  */
      splay_tree_node node;

      node = ((splay_tree_node)
              (*sp->allocate) (sizeof (struct splay_tree_node_s),
                               sp->allocate_data));
      node->key = key;
      node->value = value;

      if (!sp->root)
        node->left = node->right = 0;
      else if (comparison < 0)
        {
          node->left = sp->root;
          node->right = node->left->right;
          node->left->right = 0;
        }
      else
        {
          node->right = sp->root;
          node->left = node->right->left;
          node->right->left = 0;
        }

      sp->root = node;
    }

  return sp->root;
}


/* Call FN, passing it the DATA, for every node below NODE, all of
   which are from SP, following an in-order traversal.  If FN every
   returns a non-zero value, the iteration ceases immediately, and the
   value is returned.  Otherwise, this function returns 0.  */

static int
splay_tree_foreach_helper (splay_tree_node node,
                           splay_tree_foreach_fn fn, void *data)
{
  int val;
  splay_tree_node *stack;
  int stack_ptr, stack_size;
}

/* Call FN, passing it the DATA, for every node in SP, following an
   in-order traversal.  If FN every returns a non-zero value, the
   iteration ceases immediately, and the value is returned.
   Otherwise, this function returns 0.  */

int
splay_tree_foreach (splay_tree sp, splay_tree_foreach_fn fn, void *data)
{
  return splay_tree_foreach_helper (sp->root, fn, data);
}

/* Splay-tree comparison function, treating the keys as ints.  */

int
splay_tree_compare_ints (splay_tree_key k1, splay_tree_key k2)
{
  if ((int) k1 < (int) k2)
    return -1;
  else if ((int) k1 > (int) k2)
    return 1;
  else
    return 0;
}

/*************************/
/***** spalay-tree.c *****/
/*************************/

/* Return the node in SP with the smallest key.  */

splay_tree_node
splay_tree_min (splay_tree sp)
{
  splay_tree_node n = sp->root;

  if (!n)
    return NULL;

  while (n->left)
    n = n->left;

  return n;
}

/* Return the immediate successor KEY, or NULL if there is no
   successor.  KEY need not be present in the tree.  */

splay_tree_node
splay_tree_successor (splay_tree sp, splay_tree_key key)
{
  int comparison;
  splay_tree_node node;

  /* If the tree is empty, there is certainly no successor.  */
  if (!sp->root)
    return NULL;

  /* Splay the tree around KEY.  That will leave either the KEY
     itself, its predecessor, or its successor at the root.  */
  splay_tree_splay (sp, key);
  comparison = (*sp->comp)(sp->root->key, key);

  /* If the successor is at the root, just return it.  */
  if (comparison > 0)
    return sp->root;

  /* Otherwise, find the leftmost element of the right subtree.  */
  node = sp->root->right;
  if (node)
    while (node->left)
      node = node->left;

  return node;
}


/*************************/
/***** constructor.c *****/
/*************************/

static gfc_constructor *
node_copy (splay_tree_node node, void *base)
{
  gfc_constructor *c, *src = (gfc_constructor*)node->value;

  c = XCNEW (gfc_constructor);
  c->base = (gfc_constructor_base)base;
  c->expr = gfc_copy_expr (src->expr);
  c->iterator = gfc_copy_iterator (src->iterator);
  c->where = src->where;
  c->n.component = src->n.component;

  mpz_init_set (c->offset, src->offset);
  mpz_init_set (c->repeat, src->repeat);

  return c;
}


/* Given an constructor structure, place the expression node at position.
   Returns the constructor node inserted.  */
gfc_constructor *gfc_constructor_insert (gfc_constructor_base *base,
                                         gfc_constructor *c, int n);


static void
node_free (splay_tree_value value)
{
  gfc_constructor *c = (gfc_constructor*)value;

  if (c->expr)
    gfc_free_expr (c->expr);

  if (c->iterator)
    gfc_free_iterator (c->iterator, 1);

  mpz_clear (c->offset);
  mpz_clear (c->repeat);

  free (c);
}

static int
node_copy_and_insert (splay_tree_node node, void *base)
{
  int n = mpz_get_si (((gfc_constructor*)node->value)->offset);
  gfc_constructor_insert ((gfc_constructor_base*)base,
                          node_copy (node, base), n);
  return 0;
}


gfc_constructor *
gfc_constructor_get (void)
{
  gfc_constructor *c = XCNEW (gfc_constructor);
  c->base = NULL;
  c->expr = NULL;
  c->iterator = NULL;

  mpz_init_set_si (c->offset, 0);
  mpz_init_set_si (c->repeat, 1);

  return c;
}

gfc_constructor_base gfc_constructor_get_base (void)
{
  return splay_tree_new (splay_tree_compare_ints, NULL, node_free);
}

void
gfc_constructor_free (gfc_constructor_base base)
{
  if (base)
    splay_tree_delete (base);
}


gfc_constructor_base
gfc_constructor_copy (gfc_constructor_base base)
{
  gfc_constructor_base new_base;

  if (!base)
    return NULL;

  new_base = gfc_constructor_get_base ();
  splay_tree_foreach (base, node_copy_and_insert, &new_base);

  return new_base;
}


gfc_constructor *
gfc_constructor_append (gfc_constructor_base *base, gfc_constructor *c)
{
  int offset = 0;
  if (*base)
    offset = (int)(splay_tree_max (*base)->key) + 1;

  return gfc_constructor_insert (base, c, offset);
}


gfc_constructor *
gfc_constructor_append_expr (gfc_constructor_base *base,
                             gfc_expr *e, locus *where)
{
  gfc_constructor *c = gfc_constructor_get ();
  c->expr = e;
  if (where)
    c->where = *where;

  return gfc_constructor_append (base, c);
}


gfc_constructor *
gfc_constructor_insert (gfc_constructor_base *base, gfc_constructor *c, int n)
{
  splay_tree_node node;

  if (*base == NULL)
    *base = splay_tree_new (splay_tree_compare_ints, NULL, node_free);

  c->base = *base;
  mpz_set_si (c->offset, n);

  node = splay_tree_insert (*base, (splay_tree_key) n, (splay_tree_value) c);
  gcc_assert (node);

  return (gfc_constructor*)node->value;
}


gfc_constructor *
gfc_constructor_first (gfc_constructor_base base)
{
  if (base)
    {
      splay_tree_node node = splay_tree_min (base);
      return node ? (gfc_constructor*) node->value : NULL;
    }
  else
    return NULL;
}


gfc_constructor *
gfc_constructor_next (gfc_constructor *ctor)
{
  if (ctor)
    {
      splay_tree_node node = splay_tree_successor (ctor->base,
                                                   mpz_get_si (ctor->offset));
      return node ? (gfc_constructor*) node->value : NULL;
    }
  else
    return NULL;
}


/**********************/
/***** iresolve.c *****/
/**********************/

/* Given printf-like arguments, return a stable version of the result string.

   We already have a working, optimized string hashing table in the form of
   the identifier table.  Reusing this table is likely not to be wasted,
   since if the function name makes it to the gimple output of the frontend,
   we'll have to create the identifier anyway.  */

const char *
gfc_get_string (const char *format, ...)
{
  char temp_name[128];
  va_list ap;
  tree ident;

  memset(&temp_name,0x00,128);

  va_start (ap, format);
  vsnprintf (temp_name, sizeof (temp_name), format, ap);
  va_end (ap);
  temp_name[sizeof (temp_name) - 1] = 0;

#ifdef _RESOLUTION_
  ident = get_identifier (temp_name);
  return IDENTIFIER_POINTER (ident);
#else
  const char *rval;
  char       *r_name;
  int         len;
  len = strlen(temp_name);
  r_name = (char *)malloc(sizeof(temp_name));
  strcpy(r_name,temp_name);
  return r_name;
#endif
}


/****************/
/**** misc.c ****/
/****************/

/* Initialize a typespec to unknown.  */

void
gfc_clear_ts (gfc_typespec *ts)
{
  ts->type = BT_UNKNOWN;
  ts->u.derived = NULL;
  ts->kind = 0;
  ts->u.cl = NULL;
  ts->interface = NULL;
  /* flag that says if the type is C interoperable */
  ts->is_c_interop = 0;
  /* says what f90 type the C kind interops with */
  ts->f90_type = BT_UNKNOWN;
  /* flag that says whether it's from iso_c_binding or not */
  ts->is_iso_c = 0;
  ts->deferred = false;
}

/* Open a file for reading.  */

FILE *
gfc_open_file (const char *name)
{
  if (!*name)
    return stdin;

  return fopen (name, "r");
}


/* Given an mstring array and a string, returns the value of the tag
   field.  Returns the final tag if no matches to the string are found.  */

int
gfc_string2code (const mstring *m, const char *string)
{
  for (; m->string != NULL; m++)
    if (strcmp (m->string, string) == 0)
      return m->tag;

  return m->tag;
}


/*******************/
/**** scanner.c ****/
/*******************/

size_t
gfc_wide_strlen (const gfc_char_t *str)
{
  size_t i;

  for (i = 0; str[i]; i++)
    ;

  return i;
}


/* Functions dealing with our wide characters (gfc_char_t) and
   sequences of such characters.  */

int
gfc_wide_fits_in_byte (gfc_char_t c)
{
  return (c <= UCHAR_MAX);
}


int
gfc_wide_is_printable (gfc_char_t c)
{
  return (gfc_wide_fits_in_byte (c) && ISPRINT ((unsigned char) c));
}

gfc_char_t *
gfc_char_to_widechar (const char *s)
{
  size_t len, i;
  gfc_char_t *res;

  if (s == NULL)
    return NULL;

  len = strlen (s);
  res = gfc_get_wide_string (len + 1);

  for (i = 0; i < len; i++)
    res[i] = (unsigned char) s[i];

  res[len] = '\0';
  return res;
}

static FILE *
open_included_file (const char *name, gfc_directorylist *list,
                    bool module, bool system)
{
  char *fullname;
  gfc_directorylist *p;
  FILE *f;

  for (p = list; p; p = p->next)
    {
      if (module && !p->use_for_modules)
        continue;

      fullname = (char *) alloca(strlen (p->path) + strlen (name) + 1);
      strcpy (fullname, p->path);
      strcat (fullname, name);

      f = gfc_open_file (fullname);
      if (f != NULL)
        {
#ifdef _RESOLUTION_
          if (gfc_cpp_makedep ())
            gfc_cpp_add_dep (fullname, system);
#endif
          return f;
        }
    }

  return NULL;
}


/* Opens file for reading, searching through the include directories
   given if necessary.  If the include_cwd argument is true, we try
   to open the file in the current directory first.  */

FILE *
gfc_open_included_file (const char *name, bool include_cwd, bool module)
{
  FILE *f = NULL;

/*if (IS_ABSOLUTE_PATH (name) || include_cwd)*/
  if (name)
    {
      f = gfc_open_file (name);
#ifdef _RESOLUTION_
      if (f && gfc_cpp_makedep ())
        gfc_cpp_add_dep (name, false);
#endif
    }

  if (!f)
    f = open_included_file (name, include_dirs, module, false);

  return f;
}

FILE *
gfc_open_intrinsic_module (const char *name)
{
  FILE *f = NULL;

/*if (IS_ABSOLUTE_PATH (name))*/
  if (name)
    {
      f = gfc_open_file (name);
/*
      if (f && gfc_cpp_makedep ())
        gfc_cpp_add_dep (name, true);
*/
    }

  if (!f)
    f = open_included_file (name, intrinsic_modules_dirs, true, true);

  return f;
}

#ifdef _ZLIB_
static gzFile
gzopen_included_file_1 (const char *name, gfc_directorylist *list,
                     bool module, bool system)
{
  char *fullname;
  gfc_directorylist *p;
  gzFile f;

  for (p = list; p; p = p->next)
    {
      if (module && !p->use_for_modules)
       continue;

      fullname = (char *) alloca(strlen (p->path) + strlen (name) + 1);
      strcpy (fullname, p->path);
      strcat (fullname, name);

      f = gzopen (fullname, "r");
      if (f != NULL)
       {
#ifdef _RESOLUTION_
         if (gfc_cpp_makedep ())
           gfc_cpp_add_dep (fullname, system);
#endif
         return f;
       }
    }

  return NULL;
}


static gzFile
gzopen_included_file (const char *name, bool include_cwd, bool module)
{
  gzFile f = NULL;

/*if (IS_ABSOLUTE_PATH (name) || include_cwd)*/
  if (name)
    {
      f = gzopen (name, "r");
#ifdef _RESOLUTION_
      if (f && gfc_cpp_makedep ())
       gfc_cpp_add_dep (name, false);
#endif
    }

  if (!f)
    f = gzopen_included_file_1 (name, include_dirs, module, false);

  return f;
}

static gzFile
gzopen_intrinsic_module (const char* name)
{
  gzFile f = NULL;

/*if (IS_ABSOLUTE_PATH (name))*/
  if (name)
    {
      f = gzopen (name, "r");
#ifdef _RESOLUTION_
      if (f && gfc_cpp_makedep ())
        gfc_cpp_add_dep (name, true);
#endif
    }

  if (!f)
    f = gzopen_included_file_1 (name, intrinsic_modules_dirs, true, true);

  return f;
}
#endif /* _ZLIB_ */

/**********************/
/***** symbol.c *****/
/**********************/

/* Strings for all symbol attributes.  We use these for dumping the
   parse tree, in error messages, and also when reading and writing
   modules.  */

const mstring flavors[] =
{
  minit ("UNKNOWN-FL", FL_UNKNOWN), minit ("PROGRAM", FL_PROGRAM),
  minit ("BLOCK-DATA", FL_BLOCK_DATA), minit ("MODULE", FL_MODULE),
  minit ("VARIABLE", FL_VARIABLE), minit ("PARAMETER", FL_PARAMETER),
  minit ("LABEL", FL_LABEL), minit ("PROCEDURE", FL_PROCEDURE),
  minit ("DERIVED", FL_DERIVED), minit ("NAMELIST", FL_NAMELIST),
  minit (NULL, -1)
};

const mstring procedures[] =
{
    minit ("UNKNOWN-PROC", PROC_UNKNOWN),
    minit ("MODULE-PROC", PROC_MODULE),
    minit ("INTERNAL-PROC", PROC_INTERNAL),
    minit ("DUMMY-PROC", PROC_DUMMY),
    minit ("INTRINSIC-PROC", PROC_INTRINSIC),
    minit ("EXTERNAL-PROC", PROC_EXTERNAL),
    minit ("STATEMENT-PROC", PROC_ST_FUNCTION),
    minit (NULL, -1)
};

const mstring intents[] =
{
    minit ("UNKNOWN-INTENT", INTENT_UNKNOWN),
    minit ("IN", INTENT_IN),
    minit ("OUT", INTENT_OUT),
    minit ("INOUT", INTENT_INOUT),
    minit (NULL, -1)
};

const mstring access_types[] =
{
    minit ("UNKNOWN-ACCESS", ACCESS_UNKNOWN),
    minit ("PUBLIC", ACCESS_PUBLIC),
    minit ("PRIVATE", ACCESS_PRIVATE),
    minit (NULL, -1)
};


const mstring ifsrc_types[] =
{
    minit ("UNKNOWN", IFSRC_UNKNOWN),
    minit ("DECL", IFSRC_DECL),
    minit ("BODY", IFSRC_IFBODY)
};

const mstring save_status[] =
{
    minit ("UNKNOWN", SAVE_NONE),
    minit ("EXPLICIT-SAVE", SAVE_EXPLICIT),
    minit ("IMPLICIT-SAVE", SAVE_IMPLICIT),
};


static gfc_symbol *changed_syms = NULL;


/* List of tentative typebound-procedures.  */

typedef struct tentative_tbp
{
  gfc_typebound_proc *proc;
  struct tentative_tbp *next;
}
tentative_tbp;

static tentative_tbp *tentative_tbp_list = NULL;


/* Recursive function to switch derived types of all symbol in a
   namespace.  */

static void
switch_types (gfc_symtree *st, gfc_symbol *from, gfc_symbol *to)
{
  gfc_symbol *sym;

  if (st == NULL)
    return;

  sym = st->n.sym;
  if (sym->ts.type == BT_DERIVED && sym->ts.u.derived == from)
    sym->ts.u.derived = to;

  switch_types (st->left, from, to);
  switch_types (st->right, from, to);
}


/* This subroutine is called when a derived type is used in order to
   make the final determination about which version to use.  The
   standard requires that a type be defined before it is 'used', but
   such types can appear in IMPLICIT statements before the actual
   definition.  'Using' in this context means declaring a variable to
   be that type or using the type constructor.

   If a type is used and the components haven't been defined, then we
   have to have a derived type in a parent unit.  We find the node in
   the other namespace and point the symtree node in this namespace to
   that node.  Further reference to this name point to the correct
   node.  If we can't find the node in a parent namespace, then we have
   an error.

   This subroutine takes a pointer to a symbol node and returns a
   pointer to the translated node or NULL for an error.  Usually there
   is no translation and we return the node we were passed.  */

gfc_symbol *
gfc_use_derived (gfc_symbol *sym)
{
  gfc_symbol *s;
  gfc_typespec *t;
  gfc_symtree *st;
  int i;

  if (!sym)
    return NULL;

  if (sym->attr.generic)
    sym = gfc_find_dt_in_generic (sym);

  if (sym->components != NULL || sym->attr.zero_comp)
    return sym;               /* Already defined.  */

  if (sym->ns->parent == NULL)
    goto bad;

  if (gfc_find_symbol (sym->name, sym->ns->parent, 1, &s))
    {
      gfc_error ("Symbol '%s' at %C is ambiguous", sym->name);
      return NULL;
    }

  if (s == NULL || s->attr.flavor != FL_DERIVED)
    goto bad;

  /* Get rid of symbol sym, translating all references to s.  */
  for (i = 0; i < GFC_LETTERS; i++)
    {
      t = &sym->ns->default_type[i];
      if (t->u.derived == sym)
        t->u.derived = s;
    }

  st = gfc_find_symtree (sym->ns->sym_root, sym->name);
  st->n.sym = s;

  s->refs++;

  /* Unlink from list of modified symbols.  */
  gfc_commit_symbol (sym);

  switch_types (sym->ns->sym_root, sym, s);

  /* TODO: Also have to replace sym -> s in other lists like
     namelists, common lists and interface lists.  */
  gfc_free_symbol (sym);

  return s;

bad:
  gfc_error ("Derived type '%s' at %C is being used before it is defined",
             sym->name);
  return NULL;
}


/* Given a derived type node and a component name, try to locate the
   component structure.  Returns the NULL pointer if the component is
   not found or the components are private.  If noaccess is set, no access
   checks are done.  */

gfc_component *
gfc_find_component (gfc_symbol *sym, const char *name,
                    bool noaccess, bool silent)
{
  gfc_component *p;

  if (name == NULL || sym == NULL)
    return NULL;

  sym = gfc_use_derived (sym);

  if (sym == NULL)
    return NULL;

  for (p = sym->components; p; p = p->next)
    if (strcmp (p->name, name) == 0)
      break;

  if (p && sym->attr.use_assoc && !noaccess)
    {
      bool is_parent_comp = sym->attr.extension && (p == sym->components);
      if (p->attr.access == ACCESS_PRIVATE ||
          (p->attr.access != ACCESS_PUBLIC
           && sym->component_access == ACCESS_PRIVATE
           && !is_parent_comp))
        {
          if (!silent)
            gfc_error ("Component '%s' at %C is a PRIVATE component of '%s'",
                       name, sym->name);
          return NULL;
        }
    }


  if (p == NULL
        && sym->attr.extension
        && sym->components->ts.type == BT_DERIVED)
    {
      p = gfc_find_component (sym->components->ts.u.derived, name,
                              noaccess, silent);
      /* Do not overwrite the error.  */
      if (p == NULL)
        return p;
    }

  if (p == NULL && !silent)
    gfc_error ("'%s' at %C is not a member of the '%s' structure",
               name, sym->name);

  return p;
}


/* Given a symbol, free all of the component structures and everything
   they point to.  */

static void
free_components (gfc_component *p)
{
  gfc_component *q;

  for (; p; p = q)
    {
      q = p->next;

      gfc_free_array_spec (p->as);
      gfc_free_expr (p->initializer);

      gfc_free_formal_arglist (p->formal);
      gfc_free_namespace (p->formal_ns);

      free (p);
    }
}


/* Free a whole tree of gfc_st_label structures.  */

static void
free_st_labels (gfc_st_label *label)
{

  if (label == NULL)
    return;

  free_st_labels (label->left);
  free_st_labels (label->right);

  if (label->format != NULL)
    gfc_free_expr (label->format);
  free (label);
}


/* Clears all attributes.  */

void
gfc_clear_attr (symbol_attribute *attr)
{
  memset (attr, 0, sizeof (symbol_attribute));
}


gfc_namespace *
gfc_get_namespace (gfc_namespace *parent, int parent_types)
{
  gfc_namespace *ns;
  gfc_typespec *ts;
  int in;
  int i;

  ns = XCNEW (gfc_namespace);
  ns->sym_root = NULL;
  ns->uop_root = NULL;
  ns->tb_sym_root = NULL;
  ns->finalizers = NULL;
  ns->default_access = ACCESS_UNKNOWN;
  ns->parent = parent;

  for (in = GFC_INTRINSIC_BEGIN; in != GFC_INTRINSIC_END; in++)
    {
      ns->operator_access[in] = ACCESS_UNKNOWN;
      ns->tb_op[in] = NULL;
    }

  /* Initialize default implicit types.  */
  for (i = 'a'; i <= 'z'; i++)
    {
      ns->set_flag[i - 'a'] = 0;
      ts = &ns->default_type[i - 'a'];

      if (parent_types && ns->parent != NULL)
        {
          /* Copy parent settings.  */
          *ts = ns->parent->default_type[i - 'a'];
          continue;
        }

      if (gfc_option.flag_implicit_none != 0)
        {
          gfc_clear_ts (ts);
          continue;
        }

      if ('i' <= i && i <= 'n')
        {
          ts->type = BT_INTEGER;
          ts->kind = gfc_default_integer_kind;
        }
      else
        {
          ts->type = BT_REAL;
          ts->kind = gfc_default_real_kind;
        }
    }

  ns->refs = 1;

  return ns;
}


/* Comparison function for symtree nodes.  */

static int
compare_symtree (void *_st1, void *_st2)
{
  gfc_symtree *st1, *st2;

  st1 = (gfc_symtree *) _st1;
  st2 = (gfc_symtree *) _st2;

  return strcmp (st1->name, st2->name);
}


/* Allocate a new symtree node and associate it with the new symbol.  */

gfc_symtree *
gfc_new_symtree (gfc_symtree **root, const char *name)
{
  gfc_symtree *st;

  st = XCNEW (gfc_symtree);
  st->name = gfc_get_string (name);

  gfc_insert_bbt (root, st, compare_symtree);
  return st;
}


/* Given a root symtree node and a name, try to find the symbol within
   the namespace.  Returns NULL if the symbol is not found.  */

gfc_symtree *
gfc_find_symtree (gfc_symtree *st, const char *name)
{
  int c;

  while (st != NULL)
    {
      c = strcmp (name, st->name);
      if (c == 0)
        return st;

      st = (c < 0) ? st->left : st->right;
    }

  return NULL;
}


/* Return a symtree node with a name that is guaranteed to be unique
   within the namespace and corresponds to an illegal fortran name.  */

gfc_symtree *
gfc_get_unique_symtree (gfc_namespace *ns)
{
  char name[GFC_MAX_SYMBOL_LEN + 1];
  static int serial = 0;

  sprintf (name, "@%d", serial++);
  return gfc_new_symtree (&ns->sym_root, name);
}


/* Given a name find a user operator node, creating it if it doesn't
   exist.  These are much simpler than symbols because they can't be
   ambiguous with one another.  */

gfc_user_op *
gfc_get_uop (const char *name)
{
  gfc_user_op *uop;
  gfc_symtree *st;

  st = gfc_find_symtree (gfc_current_ns->uop_root, name);
  if (st != NULL)
    return st->n.uop;

  st = gfc_new_symtree (&gfc_current_ns->uop_root, name);

  uop = st->n.uop = XCNEW (gfc_user_op);
  uop->name = gfc_get_string (name);
  uop->access = ACCESS_UNKNOWN;
  uop->ns = gfc_current_ns;

  return uop;
}


/* Given a name find the user operator node.  Returns NULL if it does
   not exist.  */

gfc_user_op *
gfc_find_uop (const char *name, gfc_namespace *ns)
{
  gfc_symtree *st;

  if (ns == NULL)
    ns = gfc_current_ns;

  st = gfc_find_symtree (ns->uop_root, name);
  return (st == NULL) ? NULL : st->n.uop;
}

/* Remove a gfc_symbol structure and everything it points to.  */

void
gfc_free_symbol (gfc_symbol *sym)
{

  if (sym == NULL)
    return;

  gfc_free_array_spec (sym->as);

  free_components (sym->components);

  gfc_free_expr (sym->value);

  gfc_free_namelist (sym->namelist);

  gfc_free_namespace (sym->formal_ns);

  if (!sym->attr.generic_copy)
    gfc_free_interface (sym->generic);

  gfc_free_formal_arglist (sym->formal);

  gfc_free_namespace (sym->f2k_derived);

  free (sym);
}


/* Decrease the reference counter and free memory when we reach zero.  */

void
gfc_release_symbol (gfc_symbol *sym)
{
  if (sym == NULL)
    return;

  if (sym->formal_ns != NULL && sym->refs == 2)
    {
      /* As formal_ns contains a reference to sym, delete formal_ns just
         before the deletion of sym.  */
      gfc_namespace *ns = sym->formal_ns;
      sym->formal_ns = NULL;
      gfc_free_namespace (ns);
    }

  sym->refs--;
  if (sym->refs > 0)
    return;

  gcc_assert (sym->refs == 0);
  gfc_free_symbol (sym);
}


/* Allocate and initialize a new symbol node.  */

gfc_symbol *
gfc_new_symbol (const char *name, gfc_namespace *ns)
{
  gfc_symbol *p;

  p = XCNEW (gfc_symbol);

  gfc_clear_ts (&p->ts);
  gfc_clear_attr (&p->attr);
  p->ns = ns;

  p->declared_at = gfc_current_locus;

  if (strlen (name) > GFC_MAX_SYMBOL_LEN)
    gfc_internal_error ("new_symbol(): Symbol name too long");

  p->name = gfc_get_string (name);

  /* Make sure flags for symbol being C bound are clear initially.  */
  p->attr.is_bind_c = 0;
  p->attr.is_iso_c = 0;

  /* Clear the ptrs we may need.  */
  p->common_block = NULL;
  p->f2k_derived = NULL;
  p->assoc = NULL;

  return p;
}


/* Generate an error if a symbol is ambiguous.  */

static void
ambiguous_symbol (const char *name, gfc_symtree *st)
{

  if (st->n.sym->module)
    gfc_error ("Name '%s' at %C is an ambiguous reference to '%s' "
               "from module '%s'", name, st->n.sym->name, st->n.sym->module);
  else
    gfc_error ("Name '%s' at %C is an ambiguous reference to '%s' "
               "from current program unit", name, st->n.sym->name);
}


/* If we're in a SELECT TYPE block, check if the variable 'st' matches any
   selector on the stack. If yes, replace it by the corresponding temporary.  */

static void
select_type_insert_tmp (gfc_symtree **st)
{
  gfc_select_type_stack *stack = select_type_stack;
  for (; stack; stack = stack->prev)
    if ((*st)->n.sym == stack->selector && stack->tmp)
      *st = stack->tmp;
}


/* Search for a symtree starting in the current namespace, resorting to
   any parent namespaces if requested by a nonzero parent_flag.
   Returns nonzero if the name is ambiguous.  */

int
gfc_find_sym_tree (const char *name, gfc_namespace *ns, int parent_flag,
                   gfc_symtree **result)
{
  gfc_symtree *st;

  if (ns == NULL)
    ns = gfc_current_ns;

  do
    {
      st = gfc_find_symtree (ns->sym_root, name);
      if (st != NULL)
        {
          select_type_insert_tmp (&st);

          *result = st;
          /* Ambiguous generic interfaces are permitted, as long
             as the specific interfaces are different.  */
          if (st->ambiguous && !st->n.sym->attr.generic)
            {
              ambiguous_symbol (name, st);
              return 1;
            }

          return 0;
        }

      if (!parent_flag)
        break;

      ns = ns->parent;
    }
  while (ns != NULL);

  *result = NULL;
  return 0;
}

/* Same, but returns the symbol instead.  */

int
gfc_find_symbol (const char *name, gfc_namespace *ns, int parent_flag,
                 gfc_symbol **result)
{
  gfc_symtree *st;
  int i;

  i = gfc_find_sym_tree (name, ns, parent_flag, &st);

  if (st == NULL)
    *result = NULL;
  else
    *result = st->n.sym;

  return i;
}


/* Save symbol with the information necessary to back it out.  */

static void
save_symbol_data (gfc_symbol *sym)
{

  if (sym->gfc_new || sym->old_symbol != NULL)
    return;

  sym->old_symbol = XCNEW (gfc_symbol);
  *(sym->old_symbol) = *sym;

  sym->tlink = changed_syms;
  changed_syms = sym;
}


/* Given a name, find a symbol, or create it if it does not exist yet
   in the current namespace.  If the symbol is found we make sure that
   it's OK.

   The integer return code indicates
     0   All OK
     1   The symbol name was ambiguous
     2   The name meant to be established was already host associated.

   So if the return value is nonzero, then an error was issued.  */

int
gfc_get_sym_tree (const char *name, gfc_namespace *ns, gfc_symtree **result,
                  bool allow_subroutine)
{
  gfc_symtree *st;
  gfc_symbol *p;

  /* This doesn't usually happen during resolution.  */
  if (ns == NULL)
    ns = gfc_current_ns;

  /* Try to find the symbol in ns.  */
  st = gfc_find_symtree (ns->sym_root, name);

  if (st == NULL)
    {
      /* If not there, create a new symbol.  */
      p = gfc_new_symbol (name, ns);

      /* Add to the list of tentative symbols.  */
      p->old_symbol = NULL;
      p->tlink = changed_syms;
      p->mark = 1;
      p->gfc_new = 1;
      changed_syms = p;

      st = gfc_new_symtree (&ns->sym_root, name);
      st->n.sym = p;
      p->refs++;

    }
  else
    {
      /* Make sure the existing symbol is OK.  Ambiguous
         generic interfaces are permitted, as long as the
         specific interfaces are different.  */
      if (st->ambiguous && !st->n.sym->attr.generic)
        {
          ambiguous_symbol (name, st);
          return 1;
        }

      p = st->n.sym;
      if (p->ns != ns && (!p->attr.function || ns->proc_name != p)
          && !(allow_subroutine && p->attr.subroutine)
          && !(ns->proc_name && ns->proc_name->attr.if_source == IFSRC_IFBODY
          && (ns->has_import_set || p->attr.imported)))
        {
          /* Symbol is from another namespace.  */
          gfc_error ("Symbol '%s' at %C has already been host associated",
                     name);
          return 2;
        }

      p->mark = 1;

      /* Copy in case this symbol is changed.  */
      save_symbol_data (p);
    }

  *result = st;
  return 0;
}


int
gfc_get_symbol (const char *name, gfc_namespace *ns, gfc_symbol **result)
{
  gfc_symtree *st;
  int i;

  i = gfc_get_sym_tree (name, ns, &st, false);
  if (i != 0)
    return i;

  if (st)
    *result = st->n.sym;
  else
    *result = NULL;
  return i;
}



/* Free sym->old_symbol. sym->old_symbol is mostly a shallow copy of sym; the
   components of old_symbol that might need deallocation are the "allocatables"
   that are restored in gfc_undo_symbols(), with two exceptions: namelist and
   namelist_tail.  In case these differ between old_symbol and sym, it's just
   because sym->namelist has gotten a few more items.  */

static void
free_old_symbol (gfc_symbol *sym)
{

  if (sym->old_symbol == NULL)
    return;

  if (sym->old_symbol->as != sym->as)
    gfc_free_array_spec (sym->old_symbol->as);

  if (sym->old_symbol->value != sym->value)
    gfc_free_expr (sym->old_symbol->value);

  if (sym->old_symbol->formal != sym->formal)
    gfc_free_formal_arglist (sym->old_symbol->formal);

  free (sym->old_symbol);
  sym->old_symbol = NULL;
}



/* Makes the changes made in one symbol permanent -- gets rid of undo
   information.  */

void
gfc_commit_symbol (gfc_symbol *sym)
{
  gfc_symbol *p;

  if (changed_syms == sym)
    changed_syms = sym->tlink;
  else
    {
      for (p = changed_syms; p; p = p->tlink)
        if (p->tlink == sym)
          {
            p->tlink = sym->tlink;
            break;
          }
    }

  sym->tlink = NULL;
  sym->mark = 0;
  sym->gfc_new = 0;

  free_old_symbol (sym);
}


/* Recursively free trees containing type-bound procedures.  */

static void
free_tb_tree (gfc_symtree *t)
{
  if (t == NULL)
    return;

  free_tb_tree (t->left);
  free_tb_tree (t->right);

  /* TODO: Free type-bound procedure structs themselves; probably needs some
     sort of ref-counting mechanism.  */

  free (t);
}


/* Recursive function that deletes an entire tree and all the common
   head structures it points to.  */

static void
free_common_tree (gfc_symtree * common_tree)
{
  if (common_tree == NULL)
    return;

  free_common_tree (common_tree->left);
  free_common_tree (common_tree->right);

  free (common_tree);
}

/* Recursive function that deletes an entire tree and all the user
   operator nodes that it contains.  */

static void
free_uop_tree (gfc_symtree *uop_tree)
{
  if (uop_tree == NULL)
    return;

  free_uop_tree (uop_tree->left);
  free_uop_tree (uop_tree->right);

  gfc_free_interface (uop_tree->n.uop->op);
  free (uop_tree->n.uop);
  free (uop_tree);
}


/* Recursive function that deletes an entire tree and all the symbols
   that it contains.  */

static void
free_sym_tree (gfc_symtree *sym_tree)
{
  if (sym_tree == NULL)
    return;

  free_sym_tree (sym_tree->left);
  free_sym_tree (sym_tree->right);

  gfc_release_symbol (sym_tree->n.sym);
  free (sym_tree);
}


/* Free the gfc_equiv_info's.  */

static void
gfc_free_equiv_infos (gfc_equiv_info *s)
{
  if (s == NULL)
    return;
  gfc_free_equiv_infos (s->next);
  free (s);
}


/* Free the gfc_equiv_lists.  */

static void
gfc_free_equiv_lists (gfc_equiv_list *l)
{
  if (l == NULL)
    return;
  gfc_free_equiv_lists (l->next);
  gfc_free_equiv_infos (l->equiv);
  free (l);
}


/* Free a finalizer procedure list.  */

void
gfc_free_finalizer (gfc_finalizer* el)
{
  if (el)
    {
      gfc_release_symbol (el->proc_sym);
      free (el);
    }
}

static void
gfc_free_finalizer_list (gfc_finalizer* list)
{
  while (list)
    {
      gfc_finalizer* current = list;
      list = list->next;
      gfc_free_finalizer (current);
    }
}


/* Create a new gfc_charlen structure and add it to a namespace.
   If 'old_cl' is given, the newly created charlen will be a copy of it.  */

gfc_charlen*
gfc_new_charlen (gfc_namespace *ns, gfc_charlen *old_cl)
{
  gfc_charlen *cl;
  cl = gfc_get_charlen ();

  /* Copy old_cl.  */
  if (old_cl)
    {
      /* Put into namespace, but don't allow reject_statement
         to free it if old_cl is given.  */
      gfc_charlen **prev = &ns->cl_list;
      cl->next = ns->old_cl_list;
      while (*prev != ns->old_cl_list)
        prev = &(*prev)->next;
      *prev = cl;
      ns->old_cl_list = cl;
      cl->length = gfc_copy_expr (old_cl->length);
      cl->length_from_typespec = old_cl->length_from_typespec;
      cl->backend_decl = old_cl->backend_decl;
      cl->passed_length = old_cl->passed_length;
      cl->resolved = old_cl->resolved;
    }
  else
    {
      /* Put into namespace.  */
      cl->next = ns->cl_list;
      ns->cl_list = cl;
    }

  return cl;
}



/* Free the charlen list from cl to end (end is not freed).
   Free the whole list if end is NULL.  */

void
gfc_free_charlen (gfc_charlen *cl, gfc_charlen *end)
{
  gfc_charlen *cl2;

  for (; cl != end; cl = cl2)
    {
      gcc_assert (cl);

      cl2 = cl->next;
      gfc_free_expr (cl->length);
      free (cl);
    }
}


/* Free entry list structs.  */

static void
free_entry_list (gfc_entry_list *el)
{
  gfc_entry_list *next;

  if (el == NULL)
    return;

  next = el->next;
  free (el);
  free_entry_list (next);
}

/* Free a namespace structure and everything below it.  Interface
   lists associated with intrinsic operators are not freed.  These are
   taken care of when a specific name is freed.  */

void
gfc_free_namespace (gfc_namespace *ns)
{
  gfc_namespace *p, *q;
  int i;

  if (ns == NULL)
    return;

  ns->refs--;
  if (ns->refs > 0)
    return;
  gcc_assert (ns->refs == 0);

  gfc_free_statements (ns->code);

  free_sym_tree (ns->sym_root);
  free_uop_tree (ns->uop_root);
  free_common_tree (ns->common_root);
  free_tb_tree (ns->tb_sym_root);
  free_tb_tree (ns->tb_uop_root);
  gfc_free_finalizer_list (ns->finalizers);
  gfc_free_charlen (ns->cl_list, NULL);
  free_st_labels (ns->st_labels);

  free_entry_list (ns->entries);
  gfc_free_equiv (ns->equiv);
  gfc_free_equiv_lists (ns->equiv_lists);
  gfc_free_use_stmts (ns->use_stmts);

  for (i = GFC_INTRINSIC_BEGIN; i != GFC_INTRINSIC_END; i++)
    gfc_free_interface (ns->op[i]);

  gfc_free_data (ns->data);
  p = ns->contained;
  free (ns);

  /* Recursively free any contained namespaces.  */
  while (p != NULL)
    {
      q = p;
      p = p->sibling;
      gfc_free_namespace (q);
    }
}


void
gfc_symbol_init_2 (void)
{

  gfc_current_ns = gfc_get_namespace (NULL, 0);
}


/* Construct a typebound-procedure structure.  Those are stored in a tentative
   list and marked `error' until symbols are committed.  */

gfc_typebound_proc*
gfc_get_typebound_proc (gfc_typebound_proc *tb0)
{
  gfc_typebound_proc *result;
  tentative_tbp *list_node;

  result = XCNEW (gfc_typebound_proc);
  if (tb0)
    *result = *tb0;
  result->error = 1;

  list_node = XCNEW (tentative_tbp);
  list_node->next = tentative_tbp_list;
  list_node->proc = result;
  tentative_tbp_list = list_node;

  return result;
}


gfc_symbol *
gfc_find_dt_in_generic (gfc_symbol *sym)
{
  gfc_interface *intr = NULL;

  if (!sym || sym->attr.flavor == FL_DERIVED)
    return sym;

  if (sym->attr.generic)
    for (intr = (sym ? sym->generic : NULL); intr; intr = intr->next)
      if (intr->sym->attr.flavor == FL_DERIVED)
        break;
  return intr ? intr->sym : NULL;
}


/*******************/
/***** match.c *****/
/*******************/

/* Stack of SELECT TYPE statements.  */
gfc_select_type_stack *select_type_stack = NULL;

/* For debugging and diagnostic purposes.  Return the textual representation
   of the intrinsic operator OP.  */
const char *
gfc_op2string (gfc_intrinsic_op op)
{
  switch (op)
    {
    case INTRINSIC_UPLUS:
    case INTRINSIC_PLUS:
      return "+";

    case INTRINSIC_UMINUS:
    case INTRINSIC_MINUS:
      return "-";

    case INTRINSIC_POWER:
      return "**";
    case INTRINSIC_CONCAT:
      return "//";
    case INTRINSIC_TIMES:
      return "*";
    case INTRINSIC_DIVIDE:
      return "/";

    case INTRINSIC_AND:
      return ".and.";
    case INTRINSIC_OR:
      return ".or.";
    case INTRINSIC_EQV:
      return ".eqv.";
    case INTRINSIC_NEQV:
      return ".neqv.";

    case INTRINSIC_EQ_OS:
      return ".eq.";
    case INTRINSIC_EQ:
      return "==";
    case INTRINSIC_NE_OS:
      return ".ne.";
    case INTRINSIC_NE:
      return "/=";
    case INTRINSIC_GE_OS:
      return ".ge.";
    case INTRINSIC_GE:
      return ">=";
    case INTRINSIC_LE_OS:
      return ".le.";
    case INTRINSIC_LE:
      return "<=";
    case INTRINSIC_LT_OS:
      return ".lt.";
    case INTRINSIC_LT:
      return "<";
    case INTRINSIC_GT_OS:
      return ".gt.";
    case INTRINSIC_GT:
      return ">";
    case INTRINSIC_NOT:
      return ".not.";

    case INTRINSIC_ASSIGN:
      return "=";

    case INTRINSIC_PARENTHESES:
      return "parens";

    default:
      break;
    }

  gfc_internal_error ("gfc_op2string(): Bad code");
  /* Not reached.  */
}


/* Free a gfc_iterator structure.  */

void
gfc_free_iterator (gfc_iterator *iter, int flag)
{

  if (iter == NULL)
    return;

  gfc_free_expr (iter->var);
  gfc_free_expr (iter->start);
  gfc_free_expr (iter->end);
  gfc_free_expr (iter->step);

  if (flag)
    free (iter);
}


/******************** FORALL subroutines ********************/

/* Free a list of FORALL iterators.  */

void
gfc_free_forall_iterator (gfc_forall_iterator *iter)
{
  gfc_forall_iterator *next;

  while (iter)
    {
      next = iter->next;
      gfc_free_expr (iter->var);
      gfc_free_expr (iter->start);
      gfc_free_expr (iter->end);
      gfc_free_expr (iter->stride);
      free (iter);
      iter = next;
    }
}


/* Given a name, return a pointer to the common head structure,
   creating it if it does not exist. If FROM_MODULE is nonzero, we
   mangle the name so that it doesn't interfere with commons defined
   in the using namespace.
   TODO: Add to global symbol tree.  */

gfc_common_head *
gfc_get_common (const char *name, int from_module)
{
  gfc_symtree *st;
  static int serial = 0;
  char mangled_name[GFC_MAX_SYMBOL_LEN + 1];

  if (from_module)
    {
      /* A use associated common block is only needed to correctly layout
         the variables it contains.  */
      snprintf (mangled_name, GFC_MAX_SYMBOL_LEN, "_%d_%s", serial++, name);
      st = gfc_new_symtree (&gfc_current_ns->common_root, mangled_name);
    }
  else
    {
      st = gfc_find_symtree (gfc_current_ns->common_root, name);

      if (st == NULL)
        st = gfc_new_symtree (&gfc_current_ns->common_root, name);
    }

  if (st->n.common == NULL)
    {
      st->n.common = gfc_get_common_head ();
      st->n.common->where = gfc_current_locus;
      strcpy (st->n.common->name, name);
    }

  return st->n.common;
}


/* Frees a list of gfc_alloc structures.  */

void
gfc_free_alloc_list (gfc_alloc *p)
{
  gfc_alloc *q;

  for (; p; p = q)
    {
      q = p->next;
      gfc_free_expr (p->expr);
      free (p);
    }
}


/* Free a namelist structure.  */

void
gfc_free_namelist (gfc_namelist *name)
{
  gfc_namelist *n;

  for (; name; name = n)
    {
      n = name->next;
      free (name);
    }
}


/* Free equivalence sets and lists.  Recursively is the easiest way to
   do this.  */

void
gfc_free_equiv_until (gfc_equiv *eq, gfc_equiv *stop)
{
  if (eq == stop)
    return;

  gfc_free_equiv (eq->eq);
  gfc_free_equiv_until (eq->next, stop);
  gfc_free_expr (eq->expr);
  free (eq);
}


void
gfc_free_equiv (gfc_equiv *eq)
{
  gfc_free_equiv_until (eq, NULL);
}


/***************** SELECT CASE subroutines ******************/

/* Free a single case structure.  */

static void
free_case (gfc_case *p)
{
  if (p->low == p->high)
    p->high = NULL;
  gfc_free_expr (p->low);
  gfc_free_expr (p->high);
  free (p);
}


/* Free a list of case structures.  */

void
gfc_free_case_list (gfc_case *p)
{
  gfc_case *q;

  for (; p; p = q)
    {
      q = p->next;
      free_case (p);
    }
}



/****************/
/***** io.c *****/
/****************/


/* Free the gfc_open structure and all the expressions it contains.  */

void
gfc_free_open (gfc_open *open)
{
  if (open == NULL)
    return;

  gfc_free_expr (open->unit);
  gfc_free_expr (open->iomsg);
  gfc_free_expr (open->iostat);
  gfc_free_expr (open->file);
  gfc_free_expr (open->status);
  gfc_free_expr (open->access);
  gfc_free_expr (open->form);
  gfc_free_expr (open->recl);
  gfc_free_expr (open->blank);
  gfc_free_expr (open->position);
  gfc_free_expr (open->action);
  gfc_free_expr (open->delim);
  gfc_free_expr (open->pad);
  gfc_free_expr (open->decimal);
  gfc_free_expr (open->encoding);
  gfc_free_expr (open->round);
  gfc_free_expr (open->sign);
  gfc_free_expr (open->convert);
  gfc_free_expr (open->asynchronous);
  gfc_free_expr (open->newunit);
  free (open);
}


/* Free a gfc_close structure an all its expressions.  */

void
gfc_free_close (gfc_close *close)
{
  if (close == NULL)
    return;

  gfc_free_expr (close->unit);
  gfc_free_expr (close->iomsg);
  gfc_free_expr (close->iostat);
  gfc_free_expr (close->status);
  free (close);
}


/* Free a gfc_filepos structure.  */

void
gfc_free_filepos (gfc_filepos *fp)
{
  gfc_free_expr (fp->unit);
  gfc_free_expr (fp->iomsg);
  gfc_free_expr (fp->iostat);
  free (fp);
}


/* Free a data transfer structure and everything below it.  */

void
gfc_free_dt (gfc_dt *dt)
{
  if (dt == NULL)
    return;

  gfc_free_expr (dt->io_unit);
  gfc_free_expr (dt->format_expr);
  gfc_free_expr (dt->rec);
  gfc_free_expr (dt->advance);
  gfc_free_expr (dt->iomsg);
  gfc_free_expr (dt->iostat);
  gfc_free_expr (dt->size);
  gfc_free_expr (dt->pad);
  gfc_free_expr (dt->delim);
  gfc_free_expr (dt->sign);
  gfc_free_expr (dt->round);
  gfc_free_expr (dt->blank);
  gfc_free_expr (dt->decimal);
  gfc_free_expr (dt->pos);
  gfc_free_expr (dt->dt_io_kind);
  /* dt->extra_comma is a link to dt_io_kind if it is set.  */
  free (dt);
}


/* Free a gfc_inquire structure.  */

void
gfc_free_inquire (gfc_inquire *inquire)
{

  if (inquire == NULL)
    return;

  gfc_free_expr (inquire->unit);
  gfc_free_expr (inquire->file);
  gfc_free_expr (inquire->iomsg);
  gfc_free_expr (inquire->iostat);
  gfc_free_expr (inquire->exist);
  gfc_free_expr (inquire->opened);
  gfc_free_expr (inquire->number);
  gfc_free_expr (inquire->named);
  gfc_free_expr (inquire->name);
  gfc_free_expr (inquire->access);
  gfc_free_expr (inquire->sequential);
  gfc_free_expr (inquire->direct);
  gfc_free_expr (inquire->form);
  gfc_free_expr (inquire->formatted);
  gfc_free_expr (inquire->unformatted);
  gfc_free_expr (inquire->recl);
  gfc_free_expr (inquire->nextrec);
  gfc_free_expr (inquire->blank);
  gfc_free_expr (inquire->position);
  gfc_free_expr (inquire->action);
  gfc_free_expr (inquire->read);
  gfc_free_expr (inquire->write);
  gfc_free_expr (inquire->readwrite);
  gfc_free_expr (inquire->delim);
  gfc_free_expr (inquire->encoding);
  gfc_free_expr (inquire->pad);
  gfc_free_expr (inquire->iolength);
  gfc_free_expr (inquire->convert);
  gfc_free_expr (inquire->strm_pos);
  gfc_free_expr (inquire->asynchronous);
  gfc_free_expr (inquire->decimal);
  gfc_free_expr (inquire->pending);
  gfc_free_expr (inquire->id);
  gfc_free_expr (inquire->sign);
  gfc_free_expr (inquire->size);
  gfc_free_expr (inquire->round);
  free (inquire);
}


void
gfc_free_wait (gfc_wait *wait)
{
  if (wait == NULL)
    return;

  gfc_free_expr (wait->unit);
  gfc_free_expr (wait->iostat);
  gfc_free_expr (wait->iomsg);
  gfc_free_expr (wait->id);
}



/********************/
/***** openmp.c *****/
/********************/

/* Free an omp_clauses structure.  */

void
gfc_free_omp_clauses (gfc_omp_clauses *c)
{
  int i;
  if (c == NULL)
    return;

  gfc_free_expr (c->if_expr);
  gfc_free_expr (c->final_expr);
  gfc_free_expr (c->num_threads);
  gfc_free_expr (c->chunk_size);
  for (i = 0; i < OMP_LIST_NUM; i++)
    gfc_free_namelist (c->lists[i]);
  free (c);
}


/*************************/
/***** tree.h *****/
/*************************/


/* An enumeration of the standard C integer types.  These must be
   ordered so that shorter types appear before longer ones, and so
   that signed types appear before unsigned ones, for the correct
   functioning of interpret_integer() in c-lex.c.  */
enum integer_type_kind
{
  itk_char,
  itk_signed_char,
  itk_unsigned_char,
  itk_short,
  itk_unsigned_short,
  itk_int,
  itk_unsigned_int,
  itk_long,
  itk_unsigned_long,
  itk_long_long,
  itk_unsigned_long_long,
  itk_int128,
  itk_unsigned_int128,
  itk_none
};

typedef enum integer_type_kind integer_type_kind;

/* The standard C integer types.  Use integer_type_kind to index into
   this array.  */
extern GTY(()) tree integer_types[itk_none];

#define char_type_node                  integer_types[itk_char]
#define signed_char_type_node           integer_types[itk_signed_char]
#define unsigned_char_type_node         integer_types[itk_unsigned_char]
#define short_integer_type_node         integer_types[itk_short]
#define short_unsigned_type_node        integer_types[itk_unsigned_short]
#define integer_type_node               integer_types[itk_int]
#define unsigned_type_node              integer_types[itk_unsigned_int]
#define long_integer_type_node          integer_types[itk_long]
#define long_unsigned_type_node         integer_types[itk_unsigned_long]
#define long_long_integer_type_node     integer_types[itk_long_long]
#define long_long_unsigned_type_node    integer_types[itk_unsigned_long_long]
#define int128_integer_type_node        integer_types[itk_int128]
#define int128_unsigned_type_node       integer_types[itk_unsigned_int128]

#define NULL_TREE (tree) NULL

/*************************/
/***** trans-types.c *****/
/*************************/


/* Arrays for all integral and real kinds.  We'll fill this in at runtime
   after the target has a chance to process command-line options.  */

#define MAX_INT_KINDS 5
gfc_integer_info gfc_integer_kinds[MAX_INT_KINDS + 1];
gfc_logical_info gfc_logical_kinds[MAX_INT_KINDS + 1];
/*
static GTY(()) tree gfc_integer_types[MAX_INT_KINDS + 1];
static GTY(()) tree gfc_logical_types[MAX_INT_KINDS + 1];
*/

#define MAX_REAL_KINDS 5
gfc_real_info gfc_real_kinds[MAX_REAL_KINDS + 1];
/*
static GTY(()) tree gfc_real_types[MAX_REAL_KINDS + 1];
static GTY(()) tree gfc_complex_types[MAX_REAL_KINDS + 1];
*/

#define MAX_CHARACTER_KINDS 2
gfc_character_info gfc_character_kinds[MAX_CHARACTER_KINDS + 1];
/*
static GTY(()) tree gfc_character_types[MAX_CHARACTER_KINDS + 1];
static GTY(()) tree gfc_pcharacter_types[MAX_CHARACTER_KINDS + 1];
*/

/* The integer kind to use for array indices.  This will be set to the
   proper value based on target information from the backend.  */

int gfc_index_integer_kind;

/* The default kinds of the various types.  */

int gfc_default_integer_kind;
int gfc_max_integer_kind;
int gfc_default_real_kind;
int gfc_default_double_kind;
int gfc_default_character_kind;
int gfc_default_logical_kind;
int gfc_default_complex_kind;
int gfc_c_int_kind;
int gfc_atomic_int_kind;
int gfc_atomic_logical_kind;

/* The kind size used for record offsets. If the target system supports
   kind=8, this will be set to 8, otherwise it is set to 4.  */
int gfc_intio_kind;

/* Query the target to determine which machine modes are available for
   computation.  Choose KIND numbers for them.  */

/* The integer kind used to store character lengths.  */
int gfc_charlen_int_kind;

/* The size of the numeric storage unit and character storage unit.  */
int gfc_numeric_storage_size;
int gfc_character_storage_size;

static int
get_int_kind_from_width (int size)
{
  int i;

  for (i = 0; gfc_integer_kinds[i].kind != 0; i++)
    if (gfc_integer_kinds[i].bit_size == size)
      return gfc_integer_kinds[i].kind;

  return -2;
}

void
gfc_init_kinds (void)
{
  unsigned int mode;
  int i_index, r_index, kind, bitsize;
  bool saw_i4 = false, saw_i8 = false;
  bool saw_r4 = false, saw_r8 = false, saw_r10 = false, saw_r16 = false;


      i_index = 0;
      bitsize = 8;
      kind = bitsize / 8;
      gfc_integer_kinds[i_index].kind = kind;
      gfc_integer_kinds[i_index].radix = 2;
      gfc_integer_kinds[i_index].digits = bitsize - 1;
      gfc_integer_kinds[i_index].bit_size = bitsize;
      gfc_logical_kinds[i_index].kind = kind;
      gfc_logical_kinds[i_index].bit_size = bitsize;

      i_index = 1;
      bitsize = 16;
      kind = bitsize / 8;
      gfc_integer_kinds[i_index].kind = kind;
      gfc_integer_kinds[i_index].radix = 2;
      gfc_integer_kinds[i_index].digits = bitsize - 1;
      gfc_integer_kinds[i_index].bit_size = bitsize;
      gfc_logical_kinds[i_index].kind = kind;
      gfc_logical_kinds[i_index].bit_size = bitsize;

      i_index = 2;
      bitsize = 32;
      kind = bitsize / 8;
      gfc_integer_kinds[i_index].kind = kind;
      gfc_integer_kinds[i_index].radix = 2;
      gfc_integer_kinds[i_index].digits = bitsize - 1;
      gfc_integer_kinds[i_index].bit_size = bitsize;
      gfc_logical_kinds[i_index].kind = kind;
      gfc_logical_kinds[i_index].bit_size = bitsize;

      i_index = 3;
      bitsize = 64;
      kind = bitsize / 8;
      gfc_integer_kinds[i_index].kind = kind;
      gfc_integer_kinds[i_index].radix = 2;
      gfc_integer_kinds[i_index].digits = bitsize - 1;
      gfc_integer_kinds[i_index].bit_size = bitsize;
      gfc_logical_kinds[i_index].kind = kind;
      gfc_logical_kinds[i_index].bit_size = bitsize;

  /* Set the kind used to match GFC_INT_IO in libgfortran.  This is
     used for large file access.  */

    gfc_intio_kind = 8;

  /* Set the maximum integer kind.  Used with at least BOZ constants.  */
  gfc_max_integer_kind = gfc_integer_kinds[i_index].kind;

      r_index = 0;
      kind = 4;
      gfc_real_kinds[r_index].kind = kind;
      gfc_real_kinds[r_index].radix = 2;
      gfc_real_kinds[r_index].digits = 24;
      gfc_real_kinds[r_index].min_exponent = -125;
      gfc_real_kinds[r_index].max_exponent = 128;
      gfc_real_kinds[r_index].mode_precision = 64;

      r_index = 1;
      kind = 8;
      gfc_real_kinds[r_index].kind = kind;
      gfc_real_kinds[r_index].radix = 2;
      gfc_real_kinds[r_index].digits = 53;
      gfc_real_kinds[r_index].min_exponent = -1021;
      gfc_real_kinds[r_index].max_exponent = 1024;
      gfc_real_kinds[r_index].mode_precision = 64;

      r_index = 2;
      kind = 10;
      gfc_real_kinds[r_index].kind = kind;
      gfc_real_kinds[r_index].radix = 2;
      gfc_real_kinds[r_index].digits = 64;
      gfc_real_kinds[r_index].min_exponent = -16381;
      gfc_real_kinds[r_index].max_exponent = 16384;
      gfc_real_kinds[r_index].mode_precision = 80;

      r_index = 3;
      kind = 16;
      gfc_real_kinds[r_index].kind = kind;
      gfc_real_kinds[r_index].radix = 2;
      gfc_real_kinds[r_index].digits = 113;
      gfc_real_kinds[r_index].min_exponent = -16381;
      gfc_real_kinds[r_index].max_exponent = 16384;
      gfc_real_kinds[r_index].mode_precision = 128;

  gfc_numeric_storage_size = 4 * 8;

  gfc_default_integer_kind = 4;
  gfc_default_real_kind = 4;
  gfc_default_double_kind = 8;
  gfc_default_logical_kind = gfc_default_integer_kind;
  gfc_default_complex_kind = gfc_default_real_kind;

  /* We only have two character kinds: ASCII and UCS-4.
     ASCII corresponds to a 8-bit integer type, if one is available.
     UCS-4 corresponds to a 32-bit integer type, if one is available. */
  i_index = 0;
  if ((kind = get_int_kind_from_width (8)) > 0)
    {
      gfc_character_kinds[i_index].kind = kind;
      gfc_character_kinds[i_index].bit_size = 8;
      gfc_character_kinds[i_index].name = "ascii";
      i_index++;
    }
  if ((kind = get_int_kind_from_width (32)) > 0)
    {
      gfc_character_kinds[i_index].kind = kind;
      gfc_character_kinds[i_index].bit_size = 32;
      gfc_character_kinds[i_index].name = "iso_10646";
      i_index++;
    }

  /* Choose the smallest integer kind for our default character.  */
  gfc_default_character_kind = gfc_character_kinds[0].kind;
  gfc_character_storage_size = gfc_default_character_kind * 8;

  gfc_index_integer_kind = 8;

  /* Pick a kind the same size as the C "int" type.  */
  gfc_c_int_kind = INT_TYPE_SIZE / 8;

  /* Choose atomic kinds to match C's int.  */
  gfc_atomic_int_kind = gfc_c_int_kind;
  gfc_atomic_logical_kind = gfc_c_int_kind;
}


/* Make sure that a valid kind is present.  Returns an index into the
   associated kinds array, -1 if the kind is not present.  */

static int
validate_integer (int kind)
{
  int i;

  for (i = 0; gfc_integer_kinds[i].kind != 0; i++)
    if (gfc_integer_kinds[i].kind == kind)
      return i;

  return -1;
}

static int
validate_real (int kind)
{
  int i;

  for (i = 0; gfc_real_kinds[i].kind != 0; i++)
    if (gfc_real_kinds[i].kind == kind)
      return i;

  return -1;
}

static int
validate_logical (int kind)
{
  int i;

  for (i = 0; gfc_logical_kinds[i].kind; i++)
    if (gfc_logical_kinds[i].kind == kind)
      return i;

  return -1;
}


static int
validate_character (int kind)
{
  int i;

  for (i = 0; gfc_character_kinds[i].kind; i++)
    if (gfc_character_kinds[i].kind == kind)
      return i;

  return -1;
}

/* Validate a kind given a basic type.  The return value is the same
   for the child functions, with -1 indicating nonexistence of the
   type.  If MAY_FAIL is false, then -1 is never returned, and we ICE.  */

int
gfc_validate_kind (bt type, int kind, bool may_fail)
{
  int rc;

  switch (type)
    {
    case BT_REAL:               /* Fall through */
    case BT_COMPLEX:
      rc = validate_real (kind);
      break;
    case BT_INTEGER:
      rc = validate_integer (kind);
      break;
    case BT_LOGICAL:
      rc = validate_logical (kind);
      break;
    case BT_CHARACTER:
      rc = validate_character (kind);
      break;

    default:
      gfc_internal_error ("gfc_validate_kind(): Got bad type");
    }

  if (rc < 0 && !may_fail)
    gfc_internal_error ("gfc_validate_kind(): Got bad kind");

  return rc;
}


/******************/
/***** expr.c *****/
/******************/


/* Set the model number precision by the requested KIND.  */

void
gfc_set_model_kind (int kind)
{
  int index = gfc_validate_kind (BT_REAL, kind, false);
  int base2prec;

  base2prec = gfc_real_kinds[index].digits;
  if (gfc_real_kinds[index].radix != 2)
    base2prec *= gfc_real_kinds[index].radix / 2;
  mpfr_set_default_prec (base2prec);
}


/* The following set of functions provide access to gfc_expr* of
   various types - actual all but EXPR_FUNCTION and EXPR_VARIABLE.

   There are two functions available elsewhere that provide
   slightly different flavours of variables.  Namely:
     expr.c (gfc_get_variable_expr)
     symbol.c (gfc_lval_expr_from_sym)
   TODO: Merge these functions, if possible.  */

/* Get a new expression node.  */

gfc_expr *
gfc_get_expr (void)
{
  gfc_expr *e;

  e = XCNEW (gfc_expr);
  gfc_clear_ts (&e->ts);
  e->shape = NULL;
  e->ref = NULL;
  e->symtree = NULL;
  return e;
}

/* Copy a shape array.  */

mpz_t *
gfc_copy_shape (mpz_t *shape, int rank)
{
  mpz_t *new_shape;
  int n;

  if (shape == NULL)
    return NULL;

  new_shape = gfc_get_shape (rank);

  for (n = 0; n < rank; n++)
    mpz_init_set (new_shape[n], shape[n]);

  return new_shape;
}


static void
mio_gmp_real (mpfr_t *real)
{
  mp_exp_t exponent;
  char *p;

/*
  if (iomode == IO_INPUT)
*/
    {
      if (parse_atom () != ATOM_STRING)
        bad_module ("Expected real string");

      mpfr_init (*real);
      mpfr_set_str (*real, atom_string, 16, GFC_RND_MODE);
      free (atom_string);
    }
/*
  else
    {
      p = mpfr_get_str (NULL, &exponent, 16, 0, *real, GFC_RND_MODE);

      if (mpfr_nan_p (*real) || mpfr_inf_p (*real))
        {
          write_atom (ATOM_STRING, p);
          free (p);
          return;
        }

      atom_string = XCNEWVEC (char, strlen (p) + 20);

      sprintf (atom_string, "0.%s@%ld", p, exponent);
*/

      /* Fix negative numbers.  */
/*
      if (atom_string[2] == '-')
        {
          atom_string[0] = '-';
          atom_string[1] = '0';
          atom_string[2] = '.';
        }

      write_atom (ATOM_STRING, atom_string);

      free (atom_string);
      free (p);
    }
*/
}


/* Given an expression pointer, return a copy of the expression.  This
   subroutine is recursive.  */

gfc_expr *
gfc_copy_expr (gfc_expr *p)
{
  gfc_expr *q;
  gfc_char_t *s;
  char *c;

  if (p == NULL)
    return NULL;

  q = gfc_get_expr ();
  *q = *p;

  switch (q->expr_type)
    {
    case EXPR_SUBSTRING:
      s = gfc_get_wide_string (p->value.character.length + 1);
      q->value.character.string = s;
      memcpy (s, p->value.character.string,
              (p->value.character.length + 1) * sizeof (gfc_char_t));
      break;

    case EXPR_CONSTANT:
      /* Copy target representation, if it exists.  */
      if (p->representation.string)
        {
          c = XCNEWVEC (char, p->representation.length + 1);
          q->representation.string = c;
          memcpy (c, p->representation.string, (p->representation.length + 1));
        }

      /* Copy the values of any pointer components of p->value.  */
      switch (q->ts.type)
        {
        case BT_INTEGER:
          mpz_init_set (q->value.integer, p->value.integer);
          break;

        case BT_REAL:
          gfc_set_model_kind (q->ts.kind);
          mpfr_init (q->value.real);
          mpfr_set (q->value.real, p->value.real, GFC_RND_MODE);
          break;

        case BT_COMPLEX:
          gfc_set_model_kind (q->ts.kind);
#ifdef _MPCLIB_
          mpc_init2 (q->value.complex, mpfr_get_default_prec());
          mpc_set (q->value.complex, p->value.complex, GFC_MPC_RND_MODE);
#else
          mio_gmp_real (&q->value.complex.r);
          mio_gmp_real (&q->value.complex.i);
#endif
          break;

        case BT_CHARACTER:
          if (p->representation.string)
            q->value.character.string
              = gfc_char_to_widechar (q->representation.string);
          else
            {
              s = gfc_get_wide_string (p->value.character.length + 1);
              q->value.character.string = s;

              /* This is the case for the C_NULL_CHAR named constant.  */
              if (p->value.character.length == 0
                  && (p->ts.is_c_interop || p->ts.is_iso_c))
                {
                  *s = '\0';
                  /* Need to set the length to 1 to make sure the NUL
                     terminator is copied.  */
                  q->value.character.length = 1;
                }
              else
                memcpy (s, p->value.character.string,
                        (p->value.character.length + 1) * sizeof (gfc_char_t));
            }
          break;

        case BT_HOLLERITH:
        case BT_LOGICAL:
        case BT_DERIVED:
        case BT_CLASS:
          break;                /* Already done.  */

        case BT_PROCEDURE:
        case BT_VOID:
           /* Should never be reached.  */
        case BT_UNKNOWN:
          gfc_internal_error ("gfc_copy_expr(): Bad expr node");
          /* Not reached.  */
        }

      break;

    case EXPR_OP:
      switch (q->value.op.op)
        {
        case INTRINSIC_NOT:
        case INTRINSIC_PARENTHESES:
        case INTRINSIC_UPLUS:
        case INTRINSIC_UMINUS:
          q->value.op.op1 = gfc_copy_expr (p->value.op.op1);
          break;

        default:                /* Binary operators.  */
          q->value.op.op1 = gfc_copy_expr (p->value.op.op1);
          q->value.op.op2 = gfc_copy_expr (p->value.op.op2);
          break;
        }

      break;

    case EXPR_FUNCTION:
      q->value.function.actual =
        gfc_copy_actual_arglist (p->value.function.actual);
      break;

    case EXPR_COMPCALL:
    case EXPR_PPC:
      q->value.compcall.actual =
        gfc_copy_actual_arglist (p->value.compcall.actual);
      q->value.compcall.tbp = p->value.compcall.tbp;
      break;

    case EXPR_STRUCTURE:
    case EXPR_ARRAY:
      q->value.constructor = gfc_constructor_copy (p->value.constructor);
      break;

    case EXPR_VARIABLE:
    case EXPR_NULL:
      break;
    }

  q->shape = gfc_copy_shape (p->shape, p->rank);

  q->ref = gfc_copy_ref (p->ref);

  return q;
}

void
gfc_clear_shape (mpz_t *shape, int rank)
{
  int i;

  for (i = 0; i < rank; i++)
    mpz_clear (shape[i]);
}


void
gfc_free_shape (mpz_t **shape, int rank)
{
  if (*shape == NULL)
    return;

  gfc_clear_shape (*shape, rank);
  free (*shape);
  *shape = NULL;
}


/* Workhorse function for gfc_free_expr() that frees everything
   beneath an expression node, but not the node itself.  This is
   useful when we want to simplify a node and replace it with
   something else or the expression node belongs to another structure.  */

static void
free_expr0 (gfc_expr *e)
{
  switch (e->expr_type)
    {
    case EXPR_CONSTANT:
      /* Free any parts of the value that need freeing.  */
      switch (e->ts.type)
        {
        case BT_INTEGER:
          mpz_clear (e->value.integer);
          break;

        case BT_REAL:
          mpfr_clear (e->value.real);
          break;

        case BT_CHARACTER:
          free (e->value.character.string);
          break;

        case BT_COMPLEX:
#ifdef _MPCLIB_
          mpc_clear (e->value.complex);
#endif
          break;

        default:
          break;
        }

      /* Free the representation.  */
      free (e->representation.string);

      break;

    case EXPR_OP:
      if (e->value.op.op1 != NULL)
        gfc_free_expr (e->value.op.op1);
      if (e->value.op.op2 != NULL)
        gfc_free_expr (e->value.op.op2);
      break;

    case EXPR_FUNCTION:
      gfc_free_actual_arglist (e->value.function.actual);
      break;

    case EXPR_COMPCALL:
    case EXPR_PPC:
      gfc_free_actual_arglist (e->value.compcall.actual);
      break;

    case EXPR_VARIABLE:
      break;

    case EXPR_ARRAY:
    case EXPR_STRUCTURE:
      gfc_constructor_free (e->value.constructor);
      break;

    case EXPR_SUBSTRING:
      free (e->value.character.string);
      break;

    case EXPR_NULL:
      break;

    default:
      gfc_internal_error ("free_expr0(): Bad expr type");
    }

  /* Free a shape array.  */
  gfc_free_shape (&e->shape, e->rank);

  gfc_free_ref_list (e->ref);

  memset (e, '\0', sizeof (gfc_expr));
}

/* Free an expression node and everything beneath it.  */

void
gfc_free_expr (gfc_expr *e)
{
  if (e == NULL)
    return;
  free_expr0 (e);
  free (e);
}

/* Free an argument list and everything below it.  */

void
gfc_free_actual_arglist (gfc_actual_arglist *a1)
{
  gfc_actual_arglist *a2;

  while (a1)
    {
      a2 = a1->next;
      gfc_free_expr (a1->expr);
      free (a1);
      a1 = a2;
    }
}

/* Copy an arglist structure and all of the arguments.  */

gfc_actual_arglist *
gfc_copy_actual_arglist (gfc_actual_arglist *p)
{
  gfc_actual_arglist *head, *tail, *new_arg;

  head = tail = NULL;

  for (; p; p = p->next)
    {
      new_arg = gfc_get_actual_arglist ();
      *new_arg = *p;

      new_arg->expr = gfc_copy_expr (p->expr);
      new_arg->next = NULL;

      if (head == NULL)
        head = new_arg;
      else
        tail->next = new_arg;

      tail = new_arg;
    }

  return head;
}


/* Free a list of reference structures.  */

void
gfc_free_ref_list (gfc_ref *p)
{
  gfc_ref *q;
  int i;

  for (; p; p = q)
    {
      q = p->next;

      switch (p->type)
        {
        case REF_ARRAY:
          for (i = 0; i < GFC_MAX_DIMENSIONS; i++)
            {
              gfc_free_expr (p->u.ar.start[i]);
              gfc_free_expr (p->u.ar.end[i]);
              gfc_free_expr (p->u.ar.stride[i]);
            }

          break;

        case REF_SUBSTRING:
          gfc_free_expr (p->u.ss.start);
          gfc_free_expr (p->u.ss.end);
          break;

        case REF_COMPONENT:
          break;
        }

      free (p);
    }
}


/* Recursively copy a list of reference structures.  */

gfc_ref *
gfc_copy_ref (gfc_ref *src)
{
  gfc_array_ref *ar;
  gfc_ref *dest;

  if (src == NULL)
    return NULL;

  dest = gfc_get_ref ();
  dest->type = src->type;

  switch (src->type)
    {
    case REF_ARRAY:
      ar = gfc_copy_array_ref (&src->u.ar);
      dest->u.ar = *ar;
      free (ar);
      break;

    case REF_COMPONENT:
      dest->u.c = src->u.c;
      break;

    case REF_SUBSTRING:
      dest->u.ss = src->u.ss;
      dest->u.ss.start = gfc_copy_expr (src->u.ss.start);
      dest->u.ss.end = gfc_copy_expr (src->u.ss.end);
      break;
    }

  dest->next = gfc_copy_ref (src->next);

  return dest;
}

/****************/
/***** st.c *****/
/****************/

/* Free a single code structure, but not the actual structure itself.  */

void
gfc_free_statement (gfc_code *p)
{
  if (p->expr1)
    gfc_free_expr (p->expr1);
  if (p->expr2)
    gfc_free_expr (p->expr2);

  switch (p->op)
    {
    case EXEC_NOP:
    case EXEC_END_BLOCK:
    case EXEC_END_NESTED_BLOCK:
    case EXEC_ASSIGN:
    case EXEC_INIT_ASSIGN:
    case EXEC_GOTO:
    case EXEC_CYCLE:
    case EXEC_RETURN:
    case EXEC_END_PROCEDURE:
    case EXEC_IF:
    case EXEC_PAUSE:
    case EXEC_STOP:
    case EXEC_ERROR_STOP:
    case EXEC_EXIT:
    case EXEC_WHERE:
    case EXEC_IOLENGTH:
    case EXEC_POINTER_ASSIGN:
    case EXEC_DO_WHILE:
    case EXEC_CONTINUE:
    case EXEC_TRANSFER:
    case EXEC_LABEL_ASSIGN:
    case EXEC_ENTRY:
    case EXEC_ARITHMETIC_IF:
    case EXEC_CRITICAL:
    case EXEC_SYNC_ALL:
    case EXEC_SYNC_IMAGES:
    case EXEC_SYNC_MEMORY:
    case EXEC_LOCK:
    case EXEC_UNLOCK:
      break;

    case EXEC_BLOCK:
      gfc_free_namespace (p->ext.block.ns);
      gfc_free_association_list (p->ext.block.assoc);
      break;

    case EXEC_COMPCALL:
    case EXEC_CALL_PPC:
    case EXEC_CALL:
    case EXEC_ASSIGN_CALL:
      gfc_free_actual_arglist (p->ext.actual);
      break;

    case EXEC_SELECT:
    case EXEC_SELECT_TYPE:
      if (p->ext.block.case_list)
        gfc_free_case_list (p->ext.block.case_list);
      break;

    case EXEC_DO:
      gfc_free_iterator (p->ext.iterator, 1);
      break;

    case EXEC_ALLOCATE:
    case EXEC_DEALLOCATE:
      gfc_free_alloc_list (p->ext.alloc.list);
      break;

    case EXEC_OPEN:
      gfc_free_open (p->ext.open);
      break;

    case EXEC_CLOSE:
      gfc_free_close (p->ext.close);
      break;

    case EXEC_BACKSPACE:
    case EXEC_ENDFILE:
    case EXEC_REWIND:
    case EXEC_FLUSH:
      gfc_free_filepos (p->ext.filepos);
      break;

    case EXEC_INQUIRE:
      gfc_free_inquire (p->ext.inquire);
      break;

    case EXEC_WAIT:
      gfc_free_wait (p->ext.wait);
      break;

    case EXEC_READ:
    case EXEC_WRITE:
      gfc_free_dt (p->ext.dt);
      break;

    case EXEC_DT_END:
      /* The ext.dt member is a duplicate pointer and doesn't need to
         be freed.  */
      break;

    case EXEC_DO_CONCURRENT:
    case EXEC_FORALL:
      gfc_free_forall_iterator (p->ext.forall_iterator);
      break;

    case EXEC_OMP_DO:
    case EXEC_OMP_END_SINGLE:
    case EXEC_OMP_PARALLEL:
    case EXEC_OMP_PARALLEL_DO:
    case EXEC_OMP_PARALLEL_SECTIONS:
    case EXEC_OMP_SECTIONS:
    case EXEC_OMP_SINGLE:
    case EXEC_OMP_TASK:
    case EXEC_OMP_WORKSHARE:
    case EXEC_OMP_PARALLEL_WORKSHARE:
      gfc_free_omp_clauses (p->ext.omp_clauses);
      break;

    case EXEC_OMP_CRITICAL:
      free (CONST_CAST (char *, p->ext.omp_name));
      break;

    case EXEC_OMP_FLUSH:
      gfc_free_namelist (p->ext.omp_namelist);
      break;

    case EXEC_OMP_ATOMIC:
    case EXEC_OMP_BARRIER:
    case EXEC_OMP_MASTER:
    case EXEC_OMP_ORDERED:
    case EXEC_OMP_END_NOWAIT:
    case EXEC_OMP_TASKWAIT:
    case EXEC_OMP_TASKYIELD:
      break;

    default:
      gfc_internal_error ("gfc_free_statement(): Bad statement");
    }
}


/* Free a code statement and all other code structures linked to it.  */

void
gfc_free_statements (gfc_code *p)
{
  gfc_code *q;

  for (; p; p = q)
    {
      q = p->next;

      if (p->block)
        gfc_free_statements (p->block);
      gfc_free_statement (p);
      free (p);
    }
}


/* Free an association list (of an ASSOCIATE statement).  */

void
gfc_free_association_list (gfc_association_list* assoc)
{
  if (!assoc)
    return;

  gfc_free_association_list (assoc->next);
  free (assoc);
}


/*******************/
/***** class.c *****/
/*******************/


/* Get a typebound-procedure symtree or create and insert it if not yet
   present.  This is like a very simplified version of gfc_get_sym_tree for
   tbp-symtrees rather than regular ones.  */

gfc_symtree*
gfc_get_tbp_symtree (gfc_symtree **root, const char *name)
{
  gfc_symtree *result;

  result = gfc_find_symtree (*root, name);
  if (!result)
    {
      result = gfc_new_symtree (root, name);
      gcc_assert (result);
      result->n.tb = NULL;
    }

  return result;
}


/*****************/
/***** bbt.c *****/
/*****************/

/* Simple linear congruential pseudorandom number generator.  The
   period of this generator is 44071, which is plenty for our
   purposes.  */

static int
pseudo_random (void)
{
  static int x0 = 5341;

  x0 = (22611 * x0 + 10) % 44071;
  return x0;
}


/* Rotate the treap left.  */

static gfc_bbt *
rotate_left (gfc_bbt *t)
{
  gfc_bbt *temp;

  temp = t->right;
  t->right = t->right->left;
  temp->left = t;

  return temp;
}


/* Rotate the treap right.  */

static gfc_bbt *
rotate_right (gfc_bbt *t)
{
  gfc_bbt *temp;

  temp = t->left;
  t->left = t->left->right;
  temp->right = t;

  return temp;
}

/* Recursive insertion function.  Returns the updated treap, or
   aborts if we find a duplicate key.  */

static gfc_bbt *
insert (gfc_bbt *new_bbt, gfc_bbt *t, compare_fn compare)
{
  int c;

  if (t == NULL)
    return new_bbt;

  c = (*compare) (new_bbt, t);

  if (c < 0)
    {
      t->left = insert (new_bbt, t->left, compare);
      if (t->priority < t->left->priority)
        t = rotate_right (t);
    }
  else if (c > 0)
    {
      t->right = insert (new_bbt, t->right, compare);
      if (t->priority < t->right->priority)
        t = rotate_left (t);
    }
  else /* if (c == 0)  */
    gfc_internal_error("insert_bbt(): Duplicate key found!");

  return t;
}


/* Given root pointer, a new node and a comparison function, insert
   the new node into the treap.  It is an error to insert a key that
   already exists.  */

void
gfc_insert_bbt (void *root, void *new_node, compare_fn compare)
{
  gfc_bbt **r, *n;

  r = (gfc_bbt **) root;
  n = (gfc_bbt *) new_node;
  n->priority = pseudo_random ();
  *r = insert (n, *r, compare);
}



/*******************/
/*****         *****/
/*******************/

/* Report problems with a module.  Error reporting is not very
   elaborate, since this sorts of errors shouldn't really happen.
   This subroutine never returns.  */

static void
bad_module (const char *msgid)
{
  if (gzType)
    {
      XDELETEVEC (module_content);
      module_content = NULL;
    }
  else
    {
      fclose (module_fp);
    }
  

  switch (iomode)
    {
    case IO_INPUT:
      gfc_fatal_error ("Reading module %s at line %d column %d: %s",
                       module_name, module_line, module_column, msgid);
      break;
/*
    case IO_OUTPUT:
      gfc_fatal_error ("Writing module %s at line %d column %d: %s",
                       module_name, module_line, module_column, msgid);
      break;
*/
    default:
      gfc_fatal_error ("Module %s at line %d column %d: %s",
                       module_name, module_line, module_column, msgid);
      break;
    }
}



/* Set the module's input pointer.  */

static void
set_module_locus (module_locus *m)
{
  module_column = m->column;
  module_line = m->line;

#ifdef _ZLIB_
  if (gzType)
    {
      module_pos = m->posgz;
    }
  else
#endif
    {
      fsetpos (module_fp, &m->pos);
    }
}


/* Get the module's input pointer so that we can restore it later.  */

static void
get_module_locus (module_locus *m)
{
  m->column = module_column;
  m->line = module_line;
#ifdef _ZLIB_
  if (gzType)
    {
      m->posgz = module_pos;
    }
  else
#endif
    {
      fgetpos (module_fp, &m->pos);
    }
}


/* Get the next character in the module, updating our reckoning of
   where we are.  */

static int
module_char (void)
{
  int c;

  if (gzType)
    {
      c = module_content[module_pos++];
    }
  else
    {
      c = getc (module_fp);
    }

  if (c == EOF)
    bad_module ("Unexpected EOF");

  prev_module_line = module_line;
  prev_module_column = module_column;
  if (!gzType)
    {
      prev_character = c;
    }

  if (c == '\n')
    {
      module_line++;
      module_column = 0;
    }

  module_column++;
  return c;
}


/* Unget a character while remembering the line and column.  Works for
   a single character only.  */

static void
module_unget_char (void)
{
  module_line = prev_module_line;
  module_column = prev_module_column;
  if (gzType)
    {
      module_pos--;
    }
  else
    {
      ungetc (prev_character, module_fp);
    }
}


/* Parse a string constant.  The delimiter is guaranteed to be a
   single quote.  */

static void
parse_string (void)
{
  int c;
  size_t cursz = 30;
  size_t len = 0;

  atom_string = XNEWVEC (char, cursz);

  for ( ; ; )
    {
      c = module_char ();

      if (c == '\'')
        {
          int c2 = module_char ();
          if (c2 != '\'')
            {
              module_unget_char ();
              break;
            }
        }

      if (len >= cursz)
        {
          cursz *= 2;
          atom_string = XRESIZEVEC (char, atom_string, cursz);
        }
      atom_string[len] = c;
      len++;
    }

  atom_string = XRESIZEVEC (char, atom_string, len + 1);
  atom_string[len] = '\0';      /* C-style string for debug purposes.  */
}


/* Parse a small integer.  */

static void
parse_integer (int c)
{
  atom_int = c - '0';

  for (;;)
    {
      c = module_char ();
      if (!ISDIGIT (c))
        {
          module_unget_char ();
          break;
        }

      atom_int = 10 * atom_int + c - '0';
      if (atom_int > 99999999)
        bad_module ("Integer overflow");
    }

}


/* Parse a name.  */

static void
parse_name (int c)
{
  char *p;
  int len;

  p = atom_name;

  *p++ = c;
  len = 1;

  for (;;)
    {
      c = module_char ();
      if (!ISALNUM (c) && c != '_' && c != '-')
        {
          module_unget_char ();
          break;
        }

      *p++ = c;
      if (++len > GFC_MAX_SYMBOL_LEN)
        bad_module ("Name too long");
    }

  *p = '\0';

}


/* Read the next atom in the module's input stream.  */

static atom_type
parse_atom (void)
{
  int c;

  do
    {
      c = module_char ();
    }
  while (c == ' ' || c == '\r' || c == '\n');

  switch (c)
    {
    case '(':
      return ATOM_LPAREN;

    case ')':
      return ATOM_RPAREN;

    case '\'':
      parse_string ();
      return ATOM_STRING;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      parse_integer (c);
      return ATOM_INTEGER;

    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
    case 'n':
    case 'o':
    case 'p':
    case 'q':
    case 'r':
    case 's':
    case 't':
    case 'u':
    case 'v':
    case 'w':
    case 'x':
    case 'y':
    case 'z':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
    case 'T':
    case 'U':
    case 'V':
    case 'W':
    case 'X':
    case 'Y':
    case 'Z':
      parse_name (c);
      return ATOM_NAME;

    default:
      bad_module ("Bad name");
    }

  /* Not reached.  */
}

/* Peek at the next atom on the input.  */

static atom_type
peek_atom (void)
{
  int c;

  do
    {
      c = module_char ();
    }
  while (c == ' ' || c == '\r' || c == '\n');

  switch (c)
    {
    case '(':
      module_unget_char ();
      return ATOM_LPAREN;

    case ')':
      module_unget_char ();
      return ATOM_RPAREN;

    case '\'':
      module_unget_char ();
      return ATOM_STRING;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      module_unget_char ();
      return ATOM_INTEGER;

    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
    case 'n':
    case 'o':
    case 'p':
    case 'q':
    case 'r':
    case 's':
    case 't':
    case 'u':
    case 'v':
    case 'w':
    case 'x':
    case 'y':
    case 'z':
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
    case 'G':
    case 'H':
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case 'Q':
    case 'R':
    case 'S':
    case 'T':
    case 'U':
    case 'V':
    case 'W':
    case 'X':
    case 'Y':
    case 'Z':
      module_unget_char ();
      return ATOM_NAME;

    default:
      bad_module ("Bad name");
    }
}

/* Read the next atom from the input, requiring that it be a
   particular kind.  */

static void
require_atom (atom_type type)
{
  atom_type t;
  const char *p;
  int column, line;

  column = module_column;
  line = module_line;

  t = parse_atom ();
  if (t != type)
    {
      switch (type)
        {
        case ATOM_NAME:
          p = _("Expected name");
          break;
        case ATOM_LPAREN:
          p = _("Expected left parenthesis");
          break;
        case ATOM_RPAREN:
          p = _("Expected right parenthesis");
          break;
        case ATOM_INTEGER:
          p = _("Expected integer");
          break;
        case ATOM_STRING:
          p = _("Expected string");
          break;
        default:
          gfc_internal_error ("require_atom(): bad atom type required");
        }

      module_column = column;
      module_line = line;
      bad_module (p);
    }
}


/* Given a pointer to an mstring array, require that the current input
   be one of the strings in the array.  We return the enum value.  */

static int
find_enum (const mstring *m)
{
  int i;

  i = gfc_string2code (m, atom_name);
  if (i >= 0)
    return i;

  bad_module ("find_enum(): Enum not found");

  /* Not reached.  */
}


/* Read a string. The caller is responsible for freeing.  */

static char*
read_string (void)
{
  char* p;
  require_atom (ATOM_STRING);
  p = atom_string;
  atom_string = NULL;
  return p;
}


/* Compare pointers when searching by pointer.  Used when writing a
   module.  */

static int
compare_pointers (void *_sn1, void *_sn2)
{
  pointer_info *sn1, *sn2;

  sn1 = (pointer_info *) _sn1;
  sn2 = (pointer_info *) _sn2;

  if (sn1->u.pointer < sn2->u.pointer)
    return -1;
  if (sn1->u.pointer > sn2->u.pointer)
    return 1;

  return 0;
}


/* Compare integers when searching by integer.  Used when reading a
   module.  */

static int
compare_integers (void *_sn1, void *_sn2)
{
  pointer_info *sn1, *sn2;

  sn1 = (pointer_info *) _sn1;
  sn2 = (pointer_info *) _sn2;

  if (sn1->integer < sn2->integer)
    return -1;
  if (sn1->integer > sn2->integer)
    return 1;

  return 0;
}


/* Initialize the pointer_info tree.  */

static void
init_pi_tree (void)
{
  compare_fn compare;
  pointer_info *p;

  pi_root = NULL;
  compare = (iomode == IO_INPUT) ? compare_integers : compare_pointers;

  /* Pointer 0 is the NULL pointer.  */
  p = gfc_get_pointer_info ();
  p->u.pointer = NULL;
  p->integer = 0;
  p->type = P_OTHER;

  gfc_insert_bbt (&pi_root, p, compare);

  /* Pointer 1 is the current namespace.  */
  p = gfc_get_pointer_info ();
  p->u.pointer = gfc_current_ns;
  p->integer = 1;
  p->type = P_NAMESPACE;

  gfc_insert_bbt (&pi_root, p, compare);

  symbol_number = 2;
}



/* Given an integer during reading, find it in the pointer_info tree,
   creating the node if not found.  */

static pointer_info *
get_integer (int integer)
{
  pointer_info *p, t;
  int c;

  t.integer = integer;

  p = pi_root;
  while (p != NULL)
    {
      c = compare_integers (&t, p);
      if (c == 0)
        break;

      p = (c < 0) ? p->left : p->right;
    }

  if (p != NULL)
    return p;

  p = gfc_get_pointer_info ();
  p->integer = integer;
  p->u.pointer = NULL;

  gfc_insert_bbt (&pi_root, p, compare_integers);

  return p;
}


/* Recursive function to find a pointer within a tree by brute force.  */

static pointer_info *
fp2 (pointer_info *p, const void *target)
{
  pointer_info *q;

  if (p == NULL)
    return NULL;

  if (p->u.pointer == target)
    return p;

  q = fp2 (p->left, target);
  if (q != NULL)
    return q;

  return fp2 (p->right, target);
}


/* During reading, find a pointer_info node from the pointer value.
   This amounts to a brute-force search.  */

static pointer_info *
find_pointer2 (void *p)
{
  return fp2 (pi_root, p);
}


/* Resolve any fixups using a known pointer.  */

static void
resolve_fixups (fixup_t *f, void *gp)
{
  fixup_t *next;

  for (; f; f = next)
    {
      next = f->next;
      *(f->pointer) = gp;
      free (f);
    }
}


/* Convert a string such that it starts with a lower-case character. Used
   to convert the symtree name of a derived-type to the symbol name or to
   the name of the associated generic function.  */

static const char *
dt_lower_string (const char *name)
{
  if (name[0] != (char) TOLOWER ((unsigned char) name[0]))
    return gfc_get_string ("%c%s", (char) TOLOWER ((unsigned char) name[0]),
                           &name[1]);
  return gfc_get_string (name);
}


/* Convert a string such that it starts with an upper-case character. Used to
   return the symtree-name for a derived type; the symbol name itself and the
   symtree/symbol name of the associated generic function start with a lower-
   case character.  */

static const char *
dt_upper_string (const char *name)
{
  if (name[0] != (char) TOUPPER ((unsigned char) name[0]))
    return gfc_get_string ("%c%s", (char) TOUPPER ((unsigned char) name[0]),
                           &name[1]);
  return gfc_get_string (name);
}


/* Call here during module reading when we know what pointer to
   associate with an integer.  Any fixups that exist are resolved at
   this time.  */

static void
associate_integer_pointer (pointer_info *p, void *gp)
{
  if (p->u.pointer != NULL)
    gfc_internal_error ("associate_integer_pointer(): Already associated");

  p->u.pointer = gp;

  resolve_fixups (p->fixup, gp);

  p->fixup = NULL;
}


/* During module reading, given an integer and a pointer to a pointer,
   either store the pointer from an already-known value or create a
   fixup structure in order to store things later.  Returns zero if
   the reference has been actually stored, or nonzero if the reference
   must be fixed later (i.e., associate_integer_pointer must be called
   sometime later.  Returns the pointer_info structure.  */

static pointer_info *
add_fixup (int integer, void *gp)
{
  pointer_info *p;
  fixup_t *f;
  char **cp;

  p = get_integer (integer);

  if (p->integer == 0 || p->u.pointer != NULL)
    {
      cp = (char **) gp;
      *cp = (char *) p->u.pointer;
    }
  else
    {
      f = XCNEW (fixup_t);

      f->next = p->fixup;
      p->fixup = f;

      f->pointer = (void **) gp;
    }

  return p;
}


/* Given a name and a number, inst, return the inst name
   under which to load this symbol. Returns NULL if this
   symbol shouldn't be loaded. If inst is zero, returns
   the number of instances of this name. If interface is
   true, a user-defined operator is sought, otherwise only
   non-operators are sought.  */

static const char *
find_use_name_n (const char *name, int *inst, bool interface)
{
  gfc_use_rename *u;
  const char *low_name = NULL;
  int i;

  /* For derived types.  */
  if (name[0] != (char) TOLOWER ((unsigned char) name[0]))
    low_name = dt_lower_string (name);

  i = 0;
  for (u = gfc_rename_list; u; u = u->next)
    {
      if ((!low_name && strcmp (u->use_name, name) != 0)
          || (low_name && strcmp (u->use_name, low_name) != 0)
          || (u->op == INTRINSIC_USER && !interface)
          || (u->op != INTRINSIC_USER &&  interface))
        continue;
      if (++i == *inst)
        break;
    }

  if (!*inst)
    {
      *inst = i;
      return NULL;
    }

  if (u == NULL)
    return only_flag ? NULL : name;

  u->found = 1;

  if (low_name)
    {
      if (u->local_name[0] == '\0')
        return name;
      return dt_upper_string (u->local_name);
    }

  return (u->local_name[0] != '\0') ? u->local_name : name;
}

/* Given a name, return the name under which to load this symbol.
   Returns NULL if this symbol shouldn't be loaded.  */

static const char *
find_use_name (const char *name, bool interface)
{
  int i = 1;
  return find_use_name_n (name, &i, interface);
}


/* Given a real name, return the number of use names associated with it.  */

static int
number_use_names (const char *name, bool interface)
{
  int i = 0;
  find_use_name_n (name, &i, interface);
  return i;
}


/* Try to find the operator in the current list.  */

static gfc_use_rename *
find_use_operator (gfc_intrinsic_op op)
{
  gfc_use_rename *u;

  for (u = gfc_rename_list; u; u = u->next)
    if (u->op == op)
      return u;

  return NULL;
}


/***************** Mid-level I/O subroutines *****************/

/* These subroutines let their caller read or write atoms without
   caring about which of the two is actually happening.  This lets a
   subroutine concentrate on the actual format of the data being
   written.  */

static void mio_expr (gfc_expr **);
pointer_info *mio_symbol_ref (gfc_symbol **);

/* Read or write an enumerated value.  On writing, we return the input
   value for the convenience of callers.  We avoid using an integer
   pointer because enums are sometimes inside bitfields.  */

static int
mio_name (int t, const mstring *m)
{
/*
  if (iomode == IO_OUTPUT)
    write_atom (ATOM_NAME, gfc_code2string (m, t));
  else
*/
    {
      require_atom (ATOM_NAME);
      t = find_enum (m);
    }

  return t;
}

/* Specialization of mio_name.  */

#define DECL_MIO_NAME(TYPE) \
 static inline TYPE \
 MIO_NAME(TYPE) (TYPE t, const mstring *m) \
 { \
   return (TYPE) mio_name ((int) t, m); \
 }
#define MIO_NAME(TYPE) mio_name_##TYPE


static void
mio_lparen (void)
{
/*
  if (iomode == IO_OUTPUT)
    write_atom (ATOM_LPAREN, NULL);
  else
*/
    require_atom (ATOM_LPAREN);
}


static void
mio_rparen (void)
{
/*
  if (iomode == IO_OUTPUT)
    write_atom (ATOM_RPAREN, NULL);
  else
*/
    require_atom (ATOM_RPAREN);
}


static void
mio_integer (int *ip)
{
/*
  if (iomode == IO_OUTPUT)
    write_atom (ATOM_INTEGER, ip);
  else
*/
    {
      require_atom (ATOM_INTEGER);
      *ip = atom_int;
    }
}


/* Read or write a gfc_intrinsic_op value.  */

static void
mio_intrinsic_op (gfc_intrinsic_op* op)
{
  /* FIXME: Would be nicer to do this via the operators symbolic name.  */
/*
  if (iomode == IO_OUTPUT)
    {
      int converted = (int) *op;
      write_atom (ATOM_INTEGER, &converted);
    }
  else
*/
    {
      require_atom (ATOM_INTEGER);
      *op = (gfc_intrinsic_op) atom_int;
    }
}


/* Read or write a character pointer that points to a string on the heap.  */

static const char *
mio_allocated_string (const char *s)
{
/*
  if (iomode == IO_OUTPUT)
    {
      write_atom (ATOM_STRING, s);
      return s;
    }
  else
*/
    {
      require_atom (ATOM_STRING);
      return atom_string;
    }
}


static gfc_char_t *
unquote_string (const char *s)
{
  size_t len, i;
  const char *p;
  gfc_char_t *res;

  for (p = s, len = 0; *p; p++, len++)
    {
      if (*p != '\\')
        continue;

      if (p[1] == '\\')
        p++;
      else if (p[1] == 'U')
        p += 9; /* That is a "\U????????". */
      else
        gfc_internal_error ("unquote_string(): got bad string");
    }

  res = gfc_get_wide_string (len + 1);
  for (i = 0, p = s; i < len; i++, p++)
    {
      gcc_assert (*p);

      if (*p != '\\')
        res[i] = (unsigned char) *p;
      else if (p[1] == '\\')
        {
          res[i] = (unsigned char) '\\';
          p++;
        }
      else
        {
          /* We read the 8-digits hexadecimal constant that follows.  */
          int j;
          unsigned n;
          gfc_char_t c = 0;

          gcc_assert (p[1] == 'U');
          for (j = 0; j < 8; j++)
            {
              c = c << 4;
              gcc_assert (sscanf (&p[j+2], "%01x", &n) == 1);
              c += n;
            }

          res[i] = c;
          p += 9;
        }
    }

  res[len] = '\0';
  return res;
}

/* Read or write a character pointer that points to a wide string on the
   heap, performing quoting/unquoting of nonprintable characters using the
   form \U???????? (where each ? is a hexadecimal digit).
   Length is the length of the string, only known and used in output mode.  */

static const gfc_char_t *
mio_allocated_wide_string (const gfc_char_t *s, const size_t length)
{
/*
  if (iomode == IO_OUTPUT)
    {
      char *quoted = quote_string (s, length);
      write_atom (ATOM_STRING, quoted);
      free (quoted);
      return s;
    }
  else
*/
    {
      gfc_char_t *unquoted;

      require_atom (ATOM_STRING);
      unquoted = unquote_string (atom_string);
      free (atom_string);
      return unquoted;
    }
}


/* Read or write a string that is in static memory.  */

static void
mio_pool_string (const char **stringp)
{
  /* TODO: one could write the string only once, and refer to it via a
     fixup pointer.  */

  /* As a special case we have to deal with a NULL string.  This
     happens for the 'module' member of 'gfc_symbol's that are not in a
     module.  We read / write these as the empty string.  */
/*
  if (iomode == IO_OUTPUT)
    {
      const char *p = *stringp == NULL ? "" : *stringp;
      write_atom (ATOM_STRING, p);
    }
  else
*/
    {
      require_atom (ATOM_STRING);
      *stringp = atom_string[0] == '\0' ? NULL : gfc_get_string (atom_string);
      free (atom_string);
    }
}


/* Read or write a string that is inside of some already-allocated
   structure.  */

static void
mio_internal_string (char *string)
{
/*
  if (iomode == IO_OUTPUT)
    write_atom (ATOM_STRING, string);
  else
*/
    {
      require_atom (ATOM_STRING);
      strcpy (string, atom_string);
      free (atom_string);
    }
}


typedef enum
{ AB_ALLOCATABLE, AB_DIMENSION, AB_EXTERNAL, AB_INTRINSIC, AB_OPTIONAL,
  AB_POINTER, AB_TARGET, AB_DUMMY, AB_RESULT, AB_DATA,
  AB_IN_NAMELIST, AB_IN_COMMON, AB_FUNCTION, AB_SUBROUTINE, AB_SEQUENCE,
  AB_ELEMENTAL, AB_PURE, AB_RECURSIVE, AB_GENERIC, AB_ALWAYS_EXPLICIT,
  AB_CRAY_POINTER, AB_CRAY_POINTEE, AB_THREADPRIVATE,
  AB_ALLOC_COMP, AB_POINTER_COMP, AB_PROC_POINTER_COMP, AB_PRIVATE_COMP,
  AB_VALUE, AB_VOLATILE, AB_PROTECTED, AB_LOCK_COMP,
  AB_IS_BIND_C, AB_IS_C_INTEROP, AB_IS_ISO_C, AB_ABSTRACT, AB_ZERO_COMP,
  AB_IS_CLASS, AB_PROCEDURE, AB_PROC_POINTER, AB_ASYNCHRONOUS, AB_CODIMENSION,
  AB_COARRAY_COMP, AB_VTYPE, AB_VTAB, AB_CONTIGUOUS, AB_CLASS_POINTER,
  AB_IMPLICIT_PURE
}
ab_attribute;

static const mstring attr_bits[] =
{
    minit ("ALLOCATABLE", AB_ALLOCATABLE),
    minit ("ASYNCHRONOUS", AB_ASYNCHRONOUS),
    minit ("DIMENSION", AB_DIMENSION),
    minit ("CODIMENSION", AB_CODIMENSION),
    minit ("CONTIGUOUS", AB_CONTIGUOUS),
    minit ("EXTERNAL", AB_EXTERNAL),
    minit ("INTRINSIC", AB_INTRINSIC),
    minit ("OPTIONAL", AB_OPTIONAL),
    minit ("POINTER", AB_POINTER),
    minit ("VOLATILE", AB_VOLATILE),
    minit ("TARGET", AB_TARGET),
    minit ("THREADPRIVATE", AB_THREADPRIVATE),
    minit ("DUMMY", AB_DUMMY),
    minit ("RESULT", AB_RESULT),
    minit ("DATA", AB_DATA),
    minit ("IN_NAMELIST", AB_IN_NAMELIST),
    minit ("IN_COMMON", AB_IN_COMMON),
    minit ("FUNCTION", AB_FUNCTION),
    minit ("SUBROUTINE", AB_SUBROUTINE),
    minit ("SEQUENCE", AB_SEQUENCE),
    minit ("ELEMENTAL", AB_ELEMENTAL),
    minit ("PURE", AB_PURE),
    minit ("RECURSIVE", AB_RECURSIVE),
    minit ("GENERIC", AB_GENERIC),
    minit ("ALWAYS_EXPLICIT", AB_ALWAYS_EXPLICIT),
    minit ("CRAY_POINTER", AB_CRAY_POINTER),
    minit ("CRAY_POINTEE", AB_CRAY_POINTEE),
    minit ("IS_BIND_C", AB_IS_BIND_C),
    minit ("IS_C_INTEROP", AB_IS_C_INTEROP),
    minit ("IS_ISO_C", AB_IS_ISO_C),
    minit ("VALUE", AB_VALUE),
    minit ("ALLOC_COMP", AB_ALLOC_COMP),
    minit ("COARRAY_COMP", AB_COARRAY_COMP),
    minit ("LOCK_COMP", AB_LOCK_COMP),
    minit ("POINTER_COMP", AB_POINTER_COMP),
    minit ("PROC_POINTER_COMP", AB_PROC_POINTER_COMP),
    minit ("PRIVATE_COMP", AB_PRIVATE_COMP),
    minit ("ZERO_COMP", AB_ZERO_COMP),
    minit ("PROTECTED", AB_PROTECTED),
    minit ("ABSTRACT", AB_ABSTRACT),
    minit ("IS_CLASS", AB_IS_CLASS),
    minit ("PROCEDURE", AB_PROCEDURE),
    minit ("PROC_POINTER", AB_PROC_POINTER),
    minit ("VTYPE", AB_VTYPE),
    minit ("VTAB", AB_VTAB),
    minit ("CLASS_POINTER", AB_CLASS_POINTER),
    minit ("IMPLICIT_PURE", AB_IMPLICIT_PURE),
    minit ("IMPLICIT_PURE", AB_IMPLICIT_PURE),
    minit (NULL, -1)
};

/* For binding attributes.  */
static const mstring binding_passing[] =
{
    minit ("PASS", 0),
    minit ("NOPASS", 1),
    minit (NULL, -1)
};
static const mstring binding_overriding[] =
{
    minit ("OVERRIDABLE", 0),
    minit ("NON_OVERRIDABLE", 1),
    minit ("DEFERRED", 2),
    minit (NULL, -1)
};
static const mstring binding_generic[] =
{
    minit ("SPECIFIC", 0),
    minit ("GENERIC", 1),
    minit (NULL, -1)
};
static const mstring binding_ppc[] =
{
    minit ("NO_PPC", 0),
    minit ("PPC", 1),
    minit (NULL, -1)
};

/* Specialization of mio_name.  */
DECL_MIO_NAME (ab_attribute)
DECL_MIO_NAME (ar_type)
DECL_MIO_NAME (array_type)
DECL_MIO_NAME (bt)
DECL_MIO_NAME (expr_t)
DECL_MIO_NAME (gfc_access)
DECL_MIO_NAME (gfc_intrinsic_op)
DECL_MIO_NAME (ifsrc)
DECL_MIO_NAME (save_state)
DECL_MIO_NAME (procedure_type)
DECL_MIO_NAME (ref_type)
DECL_MIO_NAME (sym_flavor)
DECL_MIO_NAME (sym_intent)
#undef DECL_MIO_NAME


/* Symbol attributes are stored in list with the first three elements
   being the enumerated fields, while the remaining elements (if any)
   indicate the individual attribute bits.  The access field is not
   saved-- it controls what symbols are exported when a module is
   written.  */

static void
mio_symbol_attribute (symbol_attribute *attr)
{
  atom_type t;
  unsigned ext_attr,extension_level;

  mio_lparen ();

  attr->flavor = MIO_NAME (sym_flavor) (attr->flavor, flavors);
  attr->intent = MIO_NAME (sym_intent) (attr->intent, intents);
  attr->proc = MIO_NAME (procedure_type) (attr->proc, procedures);
  attr->if_source = MIO_NAME (ifsrc) (attr->if_source, ifsrc_types);
  attr->save = MIO_NAME (save_state) (attr->save, save_status);

/******** ********/
  if (mod_version > 0) {
/******** ********/
     ext_attr = attr->ext_attr;
     mio_integer ((int *) &ext_attr);
     attr->ext_attr = ext_attr;

     extension_level = attr->extension;
     mio_integer ((int *) &extension_level);
     attr->extension = extension_level;
/******** ********/
  }
/******** ********/

  if (iomode == IO_OUTPUT)
    {
    }
  else
    {
      for (;;)
        {
          t = parse_atom ();
          if (t == ATOM_RPAREN)
            break;
          if (t != ATOM_NAME)
            bad_module ("Expected attribute bit name");

          switch ((ab_attribute) find_enum (attr_bits))
            {
            case AB_ALLOCATABLE:
              attr->allocatable = 1;
              break;
            case AB_ASYNCHRONOUS:
              attr->asynchronous = 1;
              break;
            case AB_DIMENSION:
              attr->dimension = 1;
              break;
            case AB_CODIMENSION:
              attr->codimension = 1;
              break;
            case AB_CONTIGUOUS:
              attr->contiguous = 1;
              break;
            case AB_EXTERNAL:
              attr->external = 1;
              break;
            case AB_INTRINSIC:
              attr->intrinsic = 1;
              break;
            case AB_OPTIONAL:
              attr->optional = 1;
              break;
            case AB_POINTER:
              attr->pointer = 1;
              break;
            case AB_CLASS_POINTER:
              attr->class_pointer = 1;
              break;
            case AB_PROTECTED:
              attr->is_protected = 1;
              break;
            case AB_VALUE:
              attr->value = 1;
              break;
            case AB_VOLATILE:
              attr->volatile_ = 1;
              break;
            case AB_TARGET:
              attr->target = 1;
              break;
            case AB_THREADPRIVATE:
              attr->threadprivate = 1;
              break;
            case AB_DUMMY:
              attr->dummy = 1;
              break;
            case AB_RESULT:
              attr->result = 1;
              break;
            case AB_DATA:
              attr->data = 1;
              break;
            case AB_IN_NAMELIST:
              attr->in_namelist = 1;
              break;
            case AB_IN_COMMON:
              attr->in_common = 1;
              break;
            case AB_FUNCTION:
              attr->function = 1;
              break;
            case AB_SUBROUTINE:
              attr->subroutine = 1;
              break;
            case AB_GENERIC:
              attr->generic = 1;
              break;
            case AB_ABSTRACT:
              attr->abstract = 1;
              break;
            case AB_SEQUENCE:
              attr->sequence = 1;
              break;
            case AB_ELEMENTAL:
              attr->elemental = 1;
              break;
            case AB_PURE:
              attr->pure = 1;
              break;
            case AB_IMPLICIT_PURE:
              attr->implicit_pure = 1;
              break;
            case AB_RECURSIVE:
              attr->recursive = 1;
              break;
            case AB_ALWAYS_EXPLICIT:
              attr->always_explicit = 1;
              break;
            case AB_CRAY_POINTER:
              attr->cray_pointer = 1;
              break;
            case AB_CRAY_POINTEE:
              attr->cray_pointee = 1;
              break;
            case AB_IS_BIND_C:
              attr->is_bind_c = 1;
              break;
            case AB_IS_C_INTEROP:
              attr->is_c_interop = 1;
              break;
            case AB_IS_ISO_C:
              attr->is_iso_c = 1;
              break;
            case AB_ALLOC_COMP:
              attr->alloc_comp = 1;
              break;
            case AB_COARRAY_COMP:
              attr->coarray_comp = 1;
              break;
            case AB_LOCK_COMP:
              attr->lock_comp = 1;
              break;
            case AB_POINTER_COMP:
              attr->pointer_comp = 1;
              break;
            case AB_PROC_POINTER_COMP:
              attr->proc_pointer_comp = 1;
              break;
            case AB_PRIVATE_COMP:
              attr->private_comp = 1;
              break;
            case AB_ZERO_COMP:
              attr->zero_comp = 1;
              break;
            case AB_IS_CLASS:
              attr->is_class = 1;
              break;
            case AB_PROCEDURE:
              attr->procedure = 1;
              break;
            case AB_PROC_POINTER:
              attr->proc_pointer = 1;
              break;
            case AB_VTYPE:
              attr->vtype = 1;
              break;
            case AB_VTAB:
              attr->vtab = 1;
              break;
            }
        }
    }
}


static const mstring bt_types[] = {
    minit ("INTEGER", BT_INTEGER),
    minit ("REAL", BT_REAL),
    minit ("COMPLEX", BT_COMPLEX),
    minit ("LOGICAL", BT_LOGICAL),
    minit ("CHARACTER", BT_CHARACTER),
    minit ("DERIVED", BT_DERIVED),
    minit ("CLASS", BT_CLASS),
    minit ("PROCEDURE", BT_PROCEDURE),
    minit ("UNKNOWN", BT_UNKNOWN),
    minit ("VOID", BT_VOID),
    minit (NULL, -1)
};



static void
mio_charlen (gfc_charlen **clp)
{
  gfc_charlen *cl;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      cl = *clp;
      if (cl != NULL)
        mio_expr (&cl->length);
    }
  else
*/
    {
      if (peek_atom () != ATOM_RPAREN)
        {
          cl = gfc_new_charlen (gfc_current_ns, NULL);
          mio_expr (&cl->length);
          *clp = cl;
        }
    }

  mio_rparen ();
}



/* See if a name is a generated name.  */

static int
check_unique_name (const char *name)
{
  return *name == '@';
}


static void
mio_typespec (gfc_typespec *ts)
{
  mio_lparen ();

  ts->type = MIO_NAME (bt) (ts->type, bt_types);

  if (ts->type != BT_DERIVED && ts->type != BT_CLASS)
    mio_integer (&ts->kind);
  else
    mio_symbol_ref (&ts->u.derived);

  mio_symbol_ref (&ts->interface);

  /* Add info for C interop and is_iso_c.  */
  mio_integer (&ts->is_c_interop);
/******** ********/
  if (mod_version > 0) {
/******** ********/
      mio_integer (&ts->is_iso_c);
/******** ********/
  }
/******** ********/

  /* If the typespec is for an identifier either from iso_c_binding, or
     a constant that was initialized to an identifier from it, use the
     f90_type.  Otherwise, use the ts->type, since it shouldn't matter.  */
  if (ts->is_iso_c)
    ts->f90_type = MIO_NAME (bt) (ts->f90_type, bt_types);
  else
    ts->f90_type = MIO_NAME (bt) (ts->type, bt_types);

  if (ts->type != BT_CHARACTER)
    {
      /* ts->u.cl is only valid for BT_CHARACTER.  */
      mio_lparen ();
      mio_rparen ();
    }
  else
    mio_charlen (&ts->u.cl);

  /* So as not to disturb the existing API, use an ATOM_NAME to
     transmit deferred characteristic for characters (F2003).  */
  if (iomode == IO_OUTPUT)
    {
/*
      if (ts->type == BT_CHARACTER && ts->deferred)
        write_atom (ATOM_NAME, "DEFERRED_CL");
*/
    }
  else if (peek_atom () != ATOM_RPAREN)
    {
      if (parse_atom () != ATOM_NAME)
        bad_module ("Expected string");
      ts->deferred = 1;
    }

  mio_rparen ();
}


static const mstring array_spec_types[] = {
    minit ("EXPLICIT", AS_EXPLICIT),
    minit ("ASSUMED_SHAPE", AS_ASSUMED_SHAPE),
    minit ("DEFERRED", AS_DEFERRED),
    minit ("ASSUMED_SIZE", AS_ASSUMED_SIZE),
    minit (NULL, -1)
};


static void
mio_array_spec (gfc_array_spec **asp)
{
  gfc_array_spec *as;
  int i;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      if (*asp == NULL)
        goto done;
      as = *asp;
    }
  else
*/
    {
      if (peek_atom () == ATOM_RPAREN)
        {
          *asp = NULL;
          goto done;
        }

      *asp = as = gfc_get_array_spec ();
    }

  mio_integer (&as->rank);
/******** ********/
  if (mod_version > 0) {
/******** ********/
     mio_integer (&as->corank);
/******** ********/
  }
/******** ********/
  as->type = MIO_NAME (array_type) (as->type, array_spec_types);

  if (iomode == IO_INPUT && as->corank)
    as->cotype = (as->type == AS_DEFERRED) ? AS_DEFERRED : AS_EXPLICIT;

  for (i = 0; i < as->rank + as->corank; i++)
    {
      mio_expr (&as->lower[i]);
      mio_expr (&as->upper[i]);
    }

done:
  mio_rparen ();
}


/* Given a pointer to an array reference structure (which lives in a
   gfc_ref structure), find the corresponding array specification
   structure.  Storing the pointer in the ref structure doesn't quite
   work when loading from a module. Generating code for an array
   reference also needs more information than just the array spec.  */

static const mstring array_ref_types[] = {
    minit ("FULL", AR_FULL),
    minit ("ELEMENT", AR_ELEMENT),
    minit ("SECTION", AR_SECTION),
    minit (NULL, -1)
};


static void
mio_array_ref (gfc_array_ref *ar)
{
  int i;

  mio_lparen ();
  ar->type = MIO_NAME (ar_type) (ar->type, array_ref_types);
  mio_integer (&ar->dimen);

  switch (ar->type)
    {
    case AR_FULL:
      break;

    case AR_ELEMENT:
      for (i = 0; i < ar->dimen; i++)
        mio_expr (&ar->start[i]);

      break;

    case AR_SECTION:
      for (i = 0; i < ar->dimen; i++)
        {
          mio_expr (&ar->start[i]);
          mio_expr (&ar->end[i]);
          mio_expr (&ar->stride[i]);
        }

      break;

    case AR_UNKNOWN:
      gfc_internal_error ("mio_array_ref(): Unknown array ref");
    }

  /* Unfortunately, ar->dimen_type is an anonymous enumerated type so
     we can't call mio_integer directly.  Instead loop over each element
     and cast it to/from an integer.  */
/*
  if (iomode == IO_OUTPUT)
    {
      for (i = 0; i < ar->dimen; i++)
        {
          int tmp = (int)ar->dimen_type[i];
          write_atom (ATOM_INTEGER, &tmp);
        }
    }
  else
*/
    {
      for (i = 0; i < ar->dimen; i++)
        {
          require_atom (ATOM_INTEGER);
          ar->dimen_type[i] = (enum gfc_array_ref_dimen_type) atom_int;
        }
    }

  if (iomode == IO_INPUT)
    {
      ar->where = gfc_current_locus;

      for (i = 0; i < ar->dimen; i++)
        ar->c_where[i] = gfc_current_locus;
    }

  mio_rparen ();
}


/* Saves or restores a pointer.  The pointer is converted back and
   forth from an integer.  We return the pointer_info pointer so that
   the caller can take additional action based on the pointer type.  */

static pointer_info *
mio_pointer_ref (void *gp)
{
  pointer_info *p;
/*
  if (iomode == IO_OUTPUT)
    {
      p = get_pointer (*((char **) gp));
      write_atom (ATOM_INTEGER, &p->integer);
    }
  else
*/
    {
      require_atom (ATOM_INTEGER);
      p = add_fixup (atom_int, gp);
    }

  return p;
}


/* Save and load references to components that occur within
   expressions.  We have to describe these references by a number and
   by name.  The number is necessary for forward references during
   reading, and the name is necessary if the symbol already exists in
   the namespace and is not loaded again.  */

static void
mio_component_ref (gfc_component **cp, gfc_symbol *sym)
{
  char name[GFC_MAX_SYMBOL_LEN + 1];
  gfc_component *q;
  pointer_info *p;

  p = mio_pointer_ref (cp);
  if (p->type == P_UNKNOWN)
    p->type = P_COMPONENT;
/*
  if (iomode == IO_OUTPUT)
    mio_pool_string (&(*cp)->name);
  else
*/
    {
      mio_internal_string (name);

      if (sym && sym->attr.is_class)
        sym = sym->components->ts.u.derived;

      /* It can happen that a component reference can be read before the
         associated derived type symbol has been loaded. Return now and
         wait for a later iteration of load_needed.  */
      if (sym == NULL)
        return;

      if (sym->components != NULL && p->u.pointer == NULL)
        {
          /* Symbol already loaded, so search by name.  */
          q = gfc_find_component (sym, name, true, true);

          if (q)
            associate_integer_pointer (p, q);
        }

      /* Make sure this symbol will eventually be loaded.  */
      p = find_pointer2 (sym);
      if (p->u.rsym.state == UNUSED)
        p->u.rsym.state = NEEDED;
    }
}


static void mio_namespace_ref (gfc_namespace **nsp);
static void mio_formal_arglist (gfc_formal_arglist **formal);
static void mio_typebound_proc (gfc_typebound_proc** proc);

static void
mio_component (gfc_component *c, int vtype)
{
  pointer_info *p;
  int n;
  gfc_formal_arglist *formal;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      p = get_pointer (c);
      mio_integer (&p->integer);
    }
  else
*/
    {
      mio_integer (&n);
      p = get_integer (n);
      associate_integer_pointer (p, c);
    }

  if (p->type == P_UNKNOWN)
    p->type = P_COMPONENT;

  mio_pool_string (&c->name);
  mio_typespec (&c->ts);
  mio_array_spec (&c->as);

  mio_symbol_attribute (&c->attr);
  if (c->ts.type == BT_CLASS)
    c->attr.class_ok = 1;
  c->attr.access = MIO_NAME (gfc_access) (c->attr.access, access_types);

  if (!vtype)
    mio_expr (&c->initializer);

  if (c->attr.proc_pointer)
    {
/*
      if (iomode == IO_OUTPUT)
        {
          formal = c->formal;
          while (formal && !formal->sym)
            formal = formal->next;

          if (formal)
            mio_namespace_ref (&formal->sym->ns);
          else
            mio_namespace_ref (&c->formal_ns);
        }
      else
*/
        {
          mio_namespace_ref (&c->formal_ns);
          /* TODO: if (c->formal_ns)
            {
              c->formal_ns->proc_name = c;
              c->refs++;
            }*/
        }

      mio_formal_arglist (&c->formal);

      mio_typebound_proc (&c->tb);
    }

  mio_rparen ();
}


static void
mio_component_list (gfc_component **cp, int vtype)
{
  gfc_component *c, *tail;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (c = *cp; c; c = c->next)
        mio_component (c, vtype);
    }
  else
*/
    {
      *cp = NULL;
      tail = NULL;

      for (;;)
        {
          if (peek_atom () == ATOM_RPAREN)
            break;

          c = gfc_get_component ();
          mio_component (c, vtype);

          if (tail == NULL)
            *cp = c;
          else
            tail->next = c;

          tail = c;
        }
    }

  mio_rparen ();
}


static void
mio_actual_arg (gfc_actual_arglist *a)
{
  mio_lparen ();
  mio_pool_string (&a->name);
  mio_expr (&a->expr);
  mio_rparen ();
}


static void
mio_actual_arglist (gfc_actual_arglist **ap)
{
  gfc_actual_arglist *a, *tail;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (a = *ap; a; a = a->next)
        mio_actual_arg (a);

    }
  else
*/
    {
      tail = NULL;

      for (;;)
        {
          if (peek_atom () != ATOM_LPAREN)
            break;

          a = gfc_get_actual_arglist ();

          if (tail == NULL)
            *ap = a;
          else
            tail->next = a;

          tail = a;
          mio_actual_arg (a);
        }
    }

  mio_rparen ();
}


/* Read and write formal argument lists.  */

static void
mio_formal_arglist (gfc_formal_arglist **formal)
{
  gfc_formal_arglist *f, *tail;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (f = *formal; f; f = f->next)
        mio_symbol_ref (&f->sym);
    }
  else
*/
    {
      *formal = tail = NULL;

      while (peek_atom () != ATOM_RPAREN)
        {
          f = gfc_get_formal_arglist ();
          mio_symbol_ref (&f->sym);

          if (*formal == NULL)
            *formal = f;
          else
            tail->next = f;

          tail = f;
        }
    }

  mio_rparen ();
}


/* Save or restore a reference to a symbol node.  */

pointer_info *
mio_symbol_ref (gfc_symbol **symp)
{
  pointer_info *p;

  p = mio_pointer_ref (symp);
  if (p->type == P_UNKNOWN)
    p->type = P_SYMBOL;

  if (iomode == IO_OUTPUT)
    {
      if (p->u.wsym.state == UNREFERENCED)
        p->u.wsym.state = NEEDS_WRITE;
    }
  else
    {
      if (p->u.rsym.state == UNUSED)
        p->u.rsym.state = NEEDED;
    }
  return p;
}


/* Save or restore a reference to a symtree node.  */

static void
mio_symtree_ref (gfc_symtree **stp)
{
  pointer_info *p;
  fixup_t *f;
/*
  if (iomode == IO_OUTPUT)
    mio_symbol_ref (&(*stp)->n.sym);
  else
*/
    {
      require_atom (ATOM_INTEGER);
      p = get_integer (atom_int);

      /* An unused equivalence member; make a symbol and a symtree
         for it.  */
      if (in_load_equiv && p->u.rsym.symtree == NULL)
        {
          /* Since this is not used, it must have a unique name.  */
          p->u.rsym.symtree = gfc_get_unique_symtree (gfc_current_ns);

          /* Make the symbol.  */
          if (p->u.rsym.sym == NULL)
            {
              p->u.rsym.sym = gfc_new_symbol (p->u.rsym.true_name,
                                              gfc_current_ns);
              p->u.rsym.sym->module = gfc_get_string (p->u.rsym.module);
            }

          p->u.rsym.symtree->n.sym = p->u.rsym.sym;
          p->u.rsym.symtree->n.sym->refs++;
          p->u.rsym.referenced = 1;

          /* If the symbol is PRIVATE and in COMMON, load_commons will
             generate a fixup symbol, which must be associated.  */
          if (p->fixup)
            resolve_fixups (p->fixup, p->u.rsym.sym);
          p->fixup = NULL;
        }

      if (p->type == P_UNKNOWN)
        p->type = P_SYMBOL;

      if (p->u.rsym.state == UNUSED)
        p->u.rsym.state = NEEDED;

      if (p->u.rsym.symtree != NULL)
        {
          *stp = p->u.rsym.symtree;
        }
      else
        {
          f = XCNEW (fixup_t);

          f->next = p->u.rsym.stfixup;
          p->u.rsym.stfixup = f;

          f->pointer = (void **) stp;
        }
    }
}



static void
mio_iterator (gfc_iterator **ip)
{
  gfc_iterator *iter;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      if (*ip == NULL)
        goto done;
    }
  else
*/
    {
      if (peek_atom () == ATOM_RPAREN)
        {
          *ip = NULL;
          goto done;
        }

      *ip = gfc_get_iterator ();
    }

  iter = *ip;

  mio_expr (&iter->var);
  mio_expr (&iter->start);
  mio_expr (&iter->end);
  mio_expr (&iter->step);

done:
  mio_rparen ();
}


static void
mio_constructor (gfc_constructor_base *cp)
{
  gfc_constructor *c;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (c = gfc_constructor_first (*cp); c; c = gfc_constructor_next (c))
        {
          mio_lparen ();
          mio_expr (&c->expr);
          mio_iterator (&c->iterator);
          mio_rparen ();
        }
    }
  else
*/
    {
      while (peek_atom () != ATOM_RPAREN)
        {
          c = gfc_constructor_append_expr (cp, NULL, NULL);

          mio_lparen ();
          mio_expr (&c->expr);
          mio_iterator (&c->iterator);
          mio_rparen ();
        }
    }

  mio_rparen ();
}

static const mstring ref_types[] = {
    minit ("ARRAY", REF_ARRAY),
    minit ("COMPONENT", REF_COMPONENT),
    minit ("SUBSTRING", REF_SUBSTRING),
    minit (NULL, -1)
};

static void
mio_ref (gfc_ref **rp)
{
  gfc_ref *r;

  mio_lparen ();

  r = *rp;
  r->type = MIO_NAME (ref_type) (r->type, ref_types);

  switch (r->type)
    {
    case REF_ARRAY:
      mio_array_ref (&r->u.ar);
      break;

    case REF_COMPONENT:
      mio_symbol_ref (&r->u.c.sym);
      mio_component_ref (&r->u.c.component, r->u.c.sym);
      break;

    case REF_SUBSTRING:
      mio_expr (&r->u.ss.start);
      mio_expr (&r->u.ss.end);
      mio_charlen (&r->u.ss.length);
      break;
    }

  mio_rparen ();
}


static void
mio_ref_list (gfc_ref **rp)
{
  gfc_ref *ref, *head, *tail;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (ref = *rp; ref; ref = ref->next)
        mio_ref (&ref);
    }
  else
*/
    {
      head = tail = NULL;

      while (peek_atom () != ATOM_RPAREN)
        {
          if (head == NULL)
            head = tail = gfc_get_ref ();
          else
            {
              tail->next = gfc_get_ref ();
              tail = tail->next;
            }

          mio_ref (&tail);
        }

      *rp = head;
    }

  mio_rparen ();
}


/* Read and write an integer value.  */

static void
mio_gmp_integer (mpz_t *integer)
{
  char *p;
/*
  if (iomode == IO_INPUT)
*/
    {
      if (parse_atom () != ATOM_STRING)
        bad_module ("Expected integer string");

      mpz_init (*integer);
      if (mpz_set_str (*integer, atom_string, 10))
        bad_module ("Error converting integer");

      free (atom_string);
    }
/*
  else
    {
      p = mpz_get_str (NULL, 10, *integer);
      write_atom (ATOM_STRING, p);
      free (p);
    }
*/
}


/* Save/restore lists of gfc_interface structures.  When loading an
   interface, we are really appending to the existing list of
   interfaces.  Checking for duplicate and ambiguous interfaces has to
   be done later when all symbols have been loaded.  */

pointer_info *
mio_interface_rest (gfc_interface **ip)
{
  gfc_interface *tail, *p;
  pointer_info *pi = NULL;
/*
  if (iomode == IO_OUTPUT)
    {
      if (ip != NULL)
        for (p = *ip; p; p = p->next)
          mio_symbol_ref (&p->sym);
    }
  else
*/
    {
      if (*ip == NULL)
        tail = NULL;
      else
        {
          tail = *ip;
          while (tail->next)
            tail = tail->next;
        }

      for (;;)
        {
          if (peek_atom () == ATOM_RPAREN)
            break;

          p = gfc_get_interface ();
          p->where = gfc_current_locus;
          pi = mio_symbol_ref (&p->sym);

          if (tail == NULL)
            *ip = p;
          else
            tail->next = p;

          tail = p;
        }
    }

  mio_rparen ();
  return pi;
}


static void
mio_namespace_ref (gfc_namespace **nsp)
{
  gfc_namespace *ns;
  pointer_info *p;

  p = mio_pointer_ref (nsp);

  if (p->type == P_UNKNOWN)
    p->type = P_NAMESPACE;

  if (iomode == IO_INPUT && p->integer != 0)
    {
      ns = (gfc_namespace *) p->u.pointer;
      if (ns == NULL)
        {
          ns = gfc_get_namespace (NULL, 0);
          associate_integer_pointer (p, ns);
        }
      else
        ns->refs++;
    }
}


/* Save/restore the f2k_derived namespace of a derived-type symbol.  */

static gfc_namespace* current_f2k_derived;

static void
mio_typebound_proc (gfc_typebound_proc** proc)
{
  int flag;
  int overriding_flag;

  if (iomode == IO_INPUT)
    {
      *proc = gfc_get_typebound_proc (NULL);
      (*proc)->where = gfc_current_locus;
    }
  gcc_assert (*proc);

  mio_lparen ();

  (*proc)->access = MIO_NAME (gfc_access) ((*proc)->access, access_types);

  /* IO the NON_OVERRIDABLE/DEFERRED combination.  */
  gcc_assert (!((*proc)->deferred && (*proc)->non_overridable));
  overriding_flag = ((*proc)->deferred << 1) | (*proc)->non_overridable;
  overriding_flag = mio_name (overriding_flag, binding_overriding);
  (*proc)->deferred = ((overriding_flag & 2) != 0);
  (*proc)->non_overridable = ((overriding_flag & 1) != 0);
  gcc_assert (!((*proc)->deferred && (*proc)->non_overridable));

  (*proc)->nopass = mio_name ((*proc)->nopass, binding_passing);
  (*proc)->is_generic = mio_name ((*proc)->is_generic, binding_generic);
  (*proc)->ppc = mio_name((*proc)->ppc, binding_ppc);

  mio_pool_string (&((*proc)->pass_arg));

  flag = (int) (*proc)->pass_arg_num;
  mio_integer (&flag);
  (*proc)->pass_arg_num = (unsigned) flag;

  if ((*proc)->is_generic)
    {
      gfc_tbp_generic* g;
      int iop;

      mio_lparen ();

      if (iomode == IO_OUTPUT)
        for (g = (*proc)->u.generic; g; g = g->next)
          {
            iop = (int) g->is_operator;
            mio_integer (&iop);
            mio_allocated_string (g->specific_st->name);
          }
      else
        {
          (*proc)->u.generic = NULL;
          while (peek_atom () != ATOM_RPAREN)
            {
              gfc_symtree** sym_root;

              g = gfc_get_tbp_generic ();
              g->specific = NULL;

              mio_integer (&iop);
              g->is_operator = (bool) iop;

              require_atom (ATOM_STRING);
              sym_root = &current_f2k_derived->tb_sym_root;
              g->specific_st = gfc_get_tbp_symtree (sym_root, atom_string);
              free (atom_string);

              g->next = (*proc)->u.generic;
              (*proc)->u.generic = g;
            }
        }

      mio_rparen ();
    }
  else if (!(*proc)->ppc)
    mio_symtree_ref (&(*proc)->u.specific);

  mio_rparen ();
}

/* Save and restore the shape of an array constructor.  */

static void
mio_shape (mpz_t **pshape, int rank)
{
  mpz_t *shape;
  atom_type t;
  int n;

  /* A NULL shape is represented by ().  */
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      shape = *pshape;
      if (!shape)
        {
          mio_rparen ();
          return;
        }
    }
  else
*/
    {
      t = peek_atom ();
      if (t == ATOM_RPAREN)
        {
          *pshape = NULL;
          mio_rparen ();
          return;
        }

      shape = gfc_get_shape (rank);
      *pshape = shape;
    }

  for (n = 0; n < rank; n++)
    mio_gmp_integer (&shape[n]);

  mio_rparen ();
}


static const mstring expr_types[] = {
    minit ("OP", EXPR_OP),
    minit ("FUNCTION", EXPR_FUNCTION),
    minit ("CONSTANT", EXPR_CONSTANT),
    minit ("VARIABLE", EXPR_VARIABLE),
    minit ("SUBSTRING", EXPR_SUBSTRING),
    minit ("STRUCTURE", EXPR_STRUCTURE),
    minit ("ARRAY", EXPR_ARRAY),
    minit ("NULL", EXPR_NULL),
    minit ("COMPCALL", EXPR_COMPCALL),
    minit (NULL, -1)
};



/* INTRINSIC_ASSIGN is missing because it is used as an index for
   generic operators, not in expressions.  INTRINSIC_USER is also
   replaced by the correct function name by the time we see it.  */

static const mstring intrinsics[] =
{
    minit ("UPLUS", INTRINSIC_UPLUS),
    minit ("UMINUS", INTRINSIC_UMINUS),
    minit ("PLUS", INTRINSIC_PLUS),
    minit ("MINUS", INTRINSIC_MINUS),
    minit ("TIMES", INTRINSIC_TIMES),
    minit ("DIVIDE", INTRINSIC_DIVIDE),
    minit ("POWER", INTRINSIC_POWER),
    minit ("CONCAT", INTRINSIC_CONCAT),
    minit ("AND", INTRINSIC_AND),
    minit ("OR", INTRINSIC_OR),
    minit ("EQV", INTRINSIC_EQV),
    minit ("NEQV", INTRINSIC_NEQV),
    minit ("EQ_SIGN", INTRINSIC_EQ),
    minit ("EQ", INTRINSIC_EQ_OS),
    minit ("NE_SIGN", INTRINSIC_NE),
    minit ("NE", INTRINSIC_NE_OS),
    minit ("GT_SIGN", INTRINSIC_GT),
    minit ("GT", INTRINSIC_GT_OS),
    minit ("GE_SIGN", INTRINSIC_GE),
    minit ("GE", INTRINSIC_GE_OS),
    minit ("LT_SIGN", INTRINSIC_LT),
    minit ("LT", INTRINSIC_LT_OS),
    minit ("LE_SIGN", INTRINSIC_LE),
    minit ("LE", INTRINSIC_LE_OS),
    minit ("NOT", INTRINSIC_NOT),
    minit ("PARENTHESES", INTRINSIC_PARENTHESES),
    minit (NULL, -1)
};


/* Remedy a couple of situations where the gfc_expr's can be defective.  */

static void
fix_mio_expr (gfc_expr *e)
{
  gfc_symtree *ns_st = NULL;
  const char *fname;
/*
  if (iomode != IO_OUTPUT)
    return;
*/
  if (e->symtree)
    {
      /* If this is a symtree for a symbol that came from a contained module
         namespace, it has a unique name and we should look in the current
         namespace to see if the required, non-contained symbol is available
         yet. If so, the latter should be written.  */
      if (e->symtree->n.sym && check_unique_name (e->symtree->name))
        {
          const char *name = e->symtree->n.sym->name;
          if (e->symtree->n.sym->attr.flavor == FL_DERIVED)
            name = dt_upper_string (name);
          ns_st = gfc_find_symtree (gfc_current_ns->sym_root, name);
        }

      /* On the other hand, if the existing symbol is the module name or the
         new symbol is a dummy argument, do not do the promotion.  */
      if (ns_st && ns_st->n.sym
          && ns_st->n.sym->attr.flavor != FL_MODULE
          && !e->symtree->n.sym->attr.dummy)
        e->symtree = ns_st;
    }
  else if (e->expr_type == EXPR_FUNCTION && e->value.function.name)
    {
      gfc_symbol *sym;

      /* In some circumstances, a function used in an initialization
         expression, in one use associated module, can fail to be
         coupled to its symtree when used in a specification
         expression in another module.  */
      fname = e->value.function.esym ? e->value.function.esym->name
                                     : e->value.function.isym->name;
      e->symtree = gfc_find_symtree (gfc_current_ns->sym_root, fname);

      if (e->symtree)
        return;

      /* This is probably a reference to a private procedure from another
         module.  To prevent a segfault, make a generic with no specific
         instances.  If this module is used, without the required
         specific coming from somewhere, the appropriate error message
         is issued.  */
      gfc_get_symbol (fname, gfc_current_ns, &sym);
      sym->attr.flavor = FL_PROCEDURE;
      sym->attr.generic = 1;
      e->symtree = gfc_find_symtree (gfc_current_ns->sym_root, fname);
      gfc_commit_symbol (sym);
    }
}


/* Reai and write expressions.  The form "()" is allowed to indicate a
   NULL expression.  */

static void
mio_expr (gfc_expr **ep)
{
  gfc_expr *e;
  atom_type t;
  int flag;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      if (*ep == NULL)
        {
          mio_rparen ();
          return;
        }

      e = *ep;
      MIO_NAME (expr_t) (e->expr_type, expr_types);
    }
  else
*/
    {
      t = parse_atom ();
      if (t == ATOM_RPAREN)
        {
          *ep = NULL;
          return;
        }

      if (t != ATOM_NAME)
        bad_module ("Expected expression type");

      e = *ep = gfc_get_expr ();
      e->where = gfc_current_locus;
      e->expr_type = (expr_t) find_enum (expr_types);
    }

  mio_typespec (&e->ts);
  mio_integer (&e->rank);

  fix_mio_expr (e);

  switch (e->expr_type)
    {
    case EXPR_OP:
      e->value.op.op
        = MIO_NAME (gfc_intrinsic_op) (e->value.op.op, intrinsics);

      switch (e->value.op.op)
        {
        case INTRINSIC_UPLUS:
        case INTRINSIC_UMINUS:
        case INTRINSIC_NOT:
        case INTRINSIC_PARENTHESES:
          mio_expr (&e->value.op.op1);
          break;

        case INTRINSIC_PLUS:
        case INTRINSIC_MINUS:
        case INTRINSIC_TIMES:
        case INTRINSIC_DIVIDE:
        case INTRINSIC_POWER:
        case INTRINSIC_CONCAT:
        case INTRINSIC_AND:
        case INTRINSIC_OR:
        case INTRINSIC_EQV:
        case INTRINSIC_NEQV:
        case INTRINSIC_EQ:
        case INTRINSIC_EQ_OS:
        case INTRINSIC_NE:
        case INTRINSIC_NE_OS:
        case INTRINSIC_GT:
        case INTRINSIC_GT_OS:
        case INTRINSIC_GE:
        case INTRINSIC_GE_OS:
        case INTRINSIC_LT:
        case INTRINSIC_LT_OS:
        case INTRINSIC_LE:
        case INTRINSIC_LE_OS:
          mio_expr (&e->value.op.op1);
          mio_expr (&e->value.op.op2);
          break;

        default:
          bad_module ("Bad operator");
        }

      break;

    case EXPR_FUNCTION:
      mio_symtree_ref (&e->symtree);
      mio_actual_arglist (&e->value.function.actual);
/*
      if (iomode == IO_OUTPUT)
        {
          e->value.function.name
            = mio_allocated_string (e->value.function.name);
          flag = e->value.function.esym != NULL;
          mio_integer (&flag);
          if (flag)
            mio_symbol_ref (&e->value.function.esym);
          else
            write_atom (ATOM_STRING, e->value.function.isym->name);
        }
      else
*/
        {
          require_atom (ATOM_STRING);
          e->value.function.name = gfc_get_string (atom_string);
          free (atom_string);

          mio_integer (&flag);
          if (flag)
            mio_symbol_ref (&e->value.function.esym);
          else
            {
              require_atom (ATOM_STRING);
              e->value.function.isym = gfc_find_function (atom_string);
              free (atom_string);
            }
        }

      break;

    case EXPR_VARIABLE:
      mio_symtree_ref (&e->symtree);
      mio_ref_list (&e->ref);
      break;

    case EXPR_SUBSTRING:
/*
#ifdef _RESOLUTION_
*/
      e->value.character.string
        = CONST_CAST (gfc_char_t *,
                      mio_allocated_wide_string (e->value.character.string,
                                                 e->value.character.length));
/*
#endif
*/
      mio_ref_list (&e->ref);
      break;

    case EXPR_STRUCTURE:
    case EXPR_ARRAY:
      mio_constructor (&e->value.constructor);
      mio_shape (&e->shape, e->rank);
      break;

    case EXPR_CONSTANT:
      switch (e->ts.type)
        {
        case BT_INTEGER:
          mio_gmp_integer (&e->value.integer);
          break;

        case BT_REAL:
          gfc_set_model_kind (e->ts.kind);
          mio_gmp_real (&e->value.real);
          break;

        case BT_COMPLEX:
          gfc_set_model_kind (e->ts.kind);
#ifdef _MPCLIB_
          mio_gmp_real (&mpc_realref (e->value.complex));
          mio_gmp_real (&mpc_imagref (e->value.complex));
#else
          mio_gmp_real (&e->value.complex.r);
          mio_gmp_real (&e->value.complex.i);
#endif
          break;

        case BT_LOGICAL:
          mio_integer (&e->value.logical);
          break;

        case BT_CHARACTER:
          mio_integer (&e->value.character.length);
/*
#ifdef _RESOLUTION_
*/
          e->value.character.string
            = CONST_CAST (gfc_char_t *,
                          mio_allocated_wide_string (e->value.character.string,
                                                     e->value.character.length));
/*
#endif
*/
          break;

        default:
          bad_module ("Bad type in constant expression");
        }

      break;

    case EXPR_NULL:
      break;

    case EXPR_COMPCALL:
    case EXPR_PPC:
      gcc_unreachable ();
      break;
    }

  mio_rparen ();
}


/* Read and write namelists.  */

static void
mio_namelist (gfc_symbol *sym)
{
  gfc_namelist *n, *m;
  const char *check_name;

  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      for (n = sym->namelist; n; n = n->next)
        mio_symbol_ref (&n->sym);
    }
  else
*/
    {
      /* This departure from the standard is flagged as an error.
         It does, in fact, work correctly. TODO: Allow it
         conditionally?  */
      if (sym->attr.flavor == FL_NAMELIST)
        {
          check_name = find_use_name (sym->name, false);
          if (check_name && strcmp (check_name, sym->name) != 0)
            gfc_error ("Namelist %s cannot be renamed by USE "
                       "association to %s", sym->name, check_name);
        }

      m = NULL;
      while (peek_atom () != ATOM_RPAREN)
        {
          n = gfc_get_namelist ();
          mio_symbol_ref (&n->sym);

          if (sym->namelist == NULL)
            sym->namelist = n;
          else
            m->next = n;

          m = n;
        }
      sym->namelist_tail = m;
    }

  mio_rparen ();
}


/* Walker-callback function for this purpose.  */
static void
mio_typebound_symtree (gfc_symtree* st)
{
/*
  if (iomode == IO_OUTPUT && !st->n.tb)
    return;
  if (iomode == IO_OUTPUT)
    {
      mio_lparen ();
      mio_allocated_string (st->name);
    }
*/
  /* For IO_INPUT, the above is done in mio_f2k_derived.  */

  mio_typebound_proc (&st->n.tb);
  mio_rparen ();
}


/* IO a full symtree (in all depth).  */
static void
mio_full_typebound_tree (gfc_symtree** root)
{
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    gfc_traverse_symtree (*root, &mio_typebound_symtree);
  else
*/
    {
      while (peek_atom () == ATOM_LPAREN)
        {
          gfc_symtree* st;

          mio_lparen ();

          require_atom (ATOM_STRING);
          st = gfc_get_tbp_symtree (root, atom_string);
          free (atom_string);

          mio_typebound_symtree (st);
        }
    }

  mio_rparen ();
}

static void
mio_finalizer (gfc_finalizer **f)
{
/*
  if (iomode == IO_OUTPUT)
    {
      gcc_assert (*f);
      gcc_assert ((*f)->proc_tree);  Should already be resolved.  *
      mio_symtree_ref (&(*f)->proc_tree);
    }
  else
*/
    {
      *f = gfc_get_finalizer ();
      (*f)->where = gfc_current_locus; /* Value should not matter.  */
      (*f)->next = NULL;

      mio_symtree_ref (&(*f)->proc_tree);
      (*f)->proc_sym = NULL;
    }
}

gfc_free (void *p)
{
  if (p != NULL)
    free (p);
}


static void
mio_f2k_derived (gfc_namespace *f2k)
{
  current_f2k_derived = f2k;

  /* Handle the list of finalizer procedures.  */
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      gfc_finalizer *f;
      for (f = f2k->finalizers; f; f = f->next)
        mio_finalizer (&f);
    }
  else
*/
    {
      f2k->finalizers = NULL;
      while (peek_atom () != ATOM_RPAREN)
        {
          gfc_finalizer *cur = NULL;
          mio_finalizer (&cur);
          cur->next = f2k->finalizers;
          f2k->finalizers = cur;
        }
    }
  mio_rparen ();

/******** ********/
  if (mod_version == 0) {
/******** ********/

  /* Handle type-bound procedures.  */
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    gfc_traverse_symtree (f2k->sym_root, &mio_typebound_symtree);
  else
*/
    {
      while (peek_atom () == ATOM_LPAREN)
        {
          gfc_symtree* st;

          mio_lparen ();

          require_atom (ATOM_STRING);
          gfc_get_sym_tree (atom_string, f2k, &st, false);
          gfc_free (atom_string);

          mio_typebound_symtree (st);
        }
    }
/******** ********/
  } else {
/******** ********/
  /* Handle type-bound procedures.  */
  mio_full_typebound_tree (&f2k->tb_sym_root);

  /* Type-bound user operators.  */
  mio_full_typebound_tree (&f2k->tb_uop_root);

  /* Type-bound intrinsic operators.  */
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      int op;
      for (op = GFC_INTRINSIC_BEGIN; op != GFC_INTRINSIC_END; ++op)
        {
          gfc_intrinsic_op realop;

          if (op == INTRINSIC_USER || !f2k->tb_op[op])
            continue;

          mio_lparen ();
          realop = (gfc_intrinsic_op) op;
          mio_intrinsic_op (&realop);
          mio_typebound_proc (&f2k->tb_op[op]);
          mio_rparen ();
        }
    }
  else
*/
    while (peek_atom () != ATOM_RPAREN)
      {
        gfc_intrinsic_op op = GFC_INTRINSIC_BEGIN; /* Silence GCC.  */

        mio_lparen ();
        mio_intrinsic_op (&op);
        mio_typebound_proc (&f2k->tb_op[op]);
        mio_rparen ();
      }
/******** ********/
   }
/******** ********/
  mio_rparen ();
}

static void
mio_full_f2k_derived (gfc_symbol *sym)
{
  mio_lparen ();
/*
  if (iomode == IO_OUTPUT)
    {
      if (sym->f2k_derived)
        mio_f2k_derived (sym->f2k_derived);
    }
  else
*/
    {
      if (peek_atom () != ATOM_RPAREN)
        {
          sym->f2k_derived = gfc_get_namespace (NULL, 0);
          mio_f2k_derived (sym->f2k_derived);
        }
      else
        gcc_assert (!sym->f2k_derived);
    }

  mio_rparen ();
}


/* Unlike most other routines, the address of the symbol node is already
   fixed on input and the name/module has already been filled in.  */

static void
mio_symbol (gfc_symbol *sym)
{
  int intmod = INTMOD_NONE;

  mio_lparen ();

  mio_symbol_attribute (&sym->attr);

  if (mod_version == 12)
    {
      /* Note that components are always saved, even if they are supposed
         to be private.  Component access is checked during searching.  */
      mio_component_list (&sym->components, sym->attr.vtype);
      if (sym->components != NULL)
        sym->component_access
          = MIO_NAME (gfc_access) (sym->component_access, access_types);
    }

  mio_typespec (&sym->ts);
  if (sym->ts.type == BT_CLASS)
    sym->attr.class_ok = 1;
/*
  if (iomode == IO_OUTPUT)
    mio_namespace_ref (&sym->formal_ns);
  else
*/
    {
      mio_namespace_ref (&sym->formal_ns);
      if (sym->formal_ns)
        {
          sym->formal_ns->proc_name = sym;
          sym->refs++;
        }
    }

  /* Save/restore common block links.  */
  mio_symbol_ref (&sym->common_next);

  mio_formal_arglist (&sym->formal);

  if (sym->attr.flavor == FL_PARAMETER)
    mio_expr (&sym->value);

  mio_array_spec (&sym->as);

  mio_symbol_ref (&sym->result);

  if (sym->attr.cray_pointee)
    mio_symbol_ref (&sym->cp_pointer);

  if (mod_version < 12)
    {
      /* Note that components are always saved, even if they are supposed
         to be private.  Component access is checked during searching.  */
      mio_component_list (&sym->components, sym->attr.vtype);
      if (sym->components != NULL)
        sym->component_access
          = MIO_NAME (gfc_access) (sym->component_access, access_types);
    }

  /* Load/save the f2k_derived namespace of a derived-type symbol.  */
  mio_full_f2k_derived (sym);

  mio_namelist (sym);

  /* Add the fields that say whether this is from an intrinsic module,
     and if so, what symbol it is within the module.  */
/*   mio_integer (&(sym->from_intmod)); */
/*
  if (iomode == IO_OUTPUT)
    {
      intmod = sym->from_intmod;
      mio_integer (&intmod);
    }
  else
*/
    {
      mio_integer (&intmod);
      sym->from_intmod = (intmod_id) intmod;
    }

  mio_integer (&(sym->intmod_sym_id));

/******** ********/
  if (mod_version > 0) {
/******** ********/
     if (sym->attr.flavor == FL_DERIVED)
       mio_integer (&(sym->hash_value));
/******** ********/
     }
/******** ********/

  mio_rparen ();
}


/* Save/restore a nameless operator interface.  */

static void
mio_interface (gfc_interface **ip)
{
  mio_lparen ();
  mio_interface_rest (ip);
}

/* Given a root symtree node and a symbol, try to find a symtree that
   references the symbol that is not a unique name.  */

static gfc_symtree *
find_symtree_for_symbol (gfc_symtree *st, gfc_symbol *sym)
{
  gfc_symtree *s = NULL;

  if (st == NULL)
    return s;

  s = find_symtree_for_symbol (st->right, sym);
  if (s != NULL)
    return s;
  s = find_symtree_for_symbol (st->left, sym);
  if (s != NULL)
    return s;

  if (st->n.sym == sym && !check_unique_name (st->name))
    return st;

  return s;
}

/* A recursive function to look for a specific symbol by name and by
   module.  Whilst several symtrees might point to one symbol, its
   is sufficient for the purposes here than one exist.  Note that
   generic interfaces are distinguished as are symbols that have been
   renamed in another module.  */
static gfc_symtree *
find_symbol (gfc_symtree *st, const char *name,
             const char *module, int generic)
{
  int c;
  gfc_symtree *retval, *s;

  if (st == NULL || st->n.sym == NULL)
    return NULL;

  c = strcmp (name, st->n.sym->name);
  if (c == 0 && st->n.sym->module
             && strcmp (module, st->n.sym->module) == 0
             && !check_unique_name (st->name))
    {
      s = gfc_find_symtree (gfc_current_ns->sym_root, name);

      /* Detect symbols that are renamed by use association in another
         module by the absence of a symtree and null attr.use_rename,
         since the latter is not transmitted in the module file.  */
      if (((!generic && !st->n.sym->attr.generic)
                || (generic && st->n.sym->attr.generic))
            && !(s == NULL && !st->n.sym->attr.use_rename))
        return st;
    }

  retval = find_symbol (st->left, name, module, generic);

  if (retval == NULL)
    retval = find_symbol (st->right, name, module, generic);

  return retval;
}


/* Skip a list between balanced left and right parens.  */

static void
skip_list (void)
{
  int level;

  level = 0;
  do
    {
      switch (parse_atom ())
        {
        case ATOM_LPAREN:
          level++;
          break;

        case ATOM_RPAREN:
          level--;
          break;

        case ATOM_STRING:
          free (atom_string);
          break;

        case ATOM_NAME:
        case ATOM_INTEGER:
          break;
        }
    }
  while (level > 0);
}


/* Load operator interfaces from the module.  Interfaces are unusual
   in that they attach themselves to existing symbols.  */

static void
load_operator_interfaces (void)
{
  const char *p;
  char name[GFC_MAX_SYMBOL_LEN + 1], module[GFC_MAX_SYMBOL_LEN + 1];
  gfc_user_op *uop;
  pointer_info *pi = NULL;
  int n, i;

  mio_lparen ();

  while (peek_atom () != ATOM_RPAREN)
    {
      mio_lparen ();

      mio_internal_string (name);
      mio_internal_string (module);

      n = number_use_names (name, true);
      n = n ? n : 1;

      for (i = 1; i <= n; i++)
        {
          /* Decide if we need to load this one or not.  */
          p = find_use_name_n (name, &i, true);

          if (p == NULL)
            {
              while (parse_atom () != ATOM_RPAREN);
              continue;
            }

          if (i == 1)
            {
              uop = gfc_get_uop (p);
              pi = mio_interface_rest (&uop->op);
            }
          else
            {
              if (gfc_find_uop (p, NULL))
                continue;
              uop = gfc_get_uop (p);
              uop->op = gfc_get_interface ();
              uop->op->where = gfc_current_locus;
              add_fixup (pi->integer, &uop->op->sym);
            }
        }
    }

  mio_rparen ();
}


/* Load interfaces from the module.  Interfaces are unusual in that
   they attach themselves to existing symbols.  */

static void
load_generic_interfaces (void)
{
  const char *p;
  char name[GFC_MAX_SYMBOL_LEN + 1], module[GFC_MAX_SYMBOL_LEN + 1];
  gfc_symbol *sym;
  gfc_interface *generic = NULL, *gen = NULL;
  int n, i, renamed;
  bool ambiguous_set = false;

  mio_lparen ();

  while (peek_atom () != ATOM_RPAREN)
    {
      mio_lparen ();

      mio_internal_string (name);
      mio_internal_string (module);

      n = number_use_names (name, false);
      renamed = n ? 1 : 0;
      n = n ? n : 1;

      for (i = 1; i <= n; i++)
        {
          gfc_symtree *st;
          /* Decide if we need to load this one or not.  */
          p = find_use_name_n (name, &i, false);

          st = find_symbol (gfc_current_ns->sym_root,
                            name, module_name, 1);

          if (!p || gfc_find_symbol (p, NULL, 0, &sym))
            {
              /* Skip the specific names for these cases.  */
              while (i == 1 && parse_atom () != ATOM_RPAREN);

              continue;
            }

          /* If the symbol exists already and is being USEd without being
             in an ONLY clause, do not load a new symtree(11.3.2).  */
          if (!only_flag && st)
            sym = st->n.sym;

          if (!sym)
            {
              if (st)
                {
                  sym = st->n.sym;
                  if (strcmp (st->name, p) != 0)
                    {
                      st = gfc_new_symtree (&gfc_current_ns->sym_root, p);
                      st->n.sym = sym;
                      sym->refs++;
                    }
                }

              /* Since we haven't found a valid generic interface, we had
                 better make one.  */
              if (!sym)
                {
                  gfc_get_symbol (p, NULL, &sym);
                  sym->name = gfc_get_string (name);
                  sym->module = module_name;
                  sym->attr.flavor = FL_PROCEDURE;
                  sym->attr.generic = 1;
                  sym->attr.use_assoc = 1;
                }
            }
          else
            {
              /* Unless sym is a generic interface, this reference
                 is ambiguous.  */
              if (st == NULL)
                st = gfc_find_symtree (gfc_current_ns->sym_root, p);

              sym = st->n.sym;

              if (st && !sym->attr.generic
                     && !st->ambiguous
                     && sym->module
                     && strcmp(module, sym->module))
                {
                  ambiguous_set = true;
                  st->ambiguous = 1;
                }
            }

          sym->attr.use_only = only_flag;
          sym->attr.use_rename = renamed;
          if (i == 1)
            {
              mio_interface_rest (&sym->generic);
              generic = sym->generic;
            }
          else if (!sym->generic)
            {
              sym->generic = generic;
              sym->attr.generic_copy = 1;
            }

          /* If a procedure that is not generic has generic interfaces
             that include itself, it is generic! We need to take care
             to retain symbols ambiguous that were already so.  */
          if (sym->attr.use_assoc
                && !sym->attr.generic
                && sym->attr.flavor == FL_PROCEDURE)
            {
              for (gen = generic; gen; gen = gen->next)
                {
                  if (gen->sym == sym)
                    {
                      sym->attr.generic = 1;
                      if (ambiguous_set)
                        st->ambiguous = 0;
                      break;
                    }
                }
            }

        }
    }

  mio_rparen ();
}


/* Load common blocks.  */

static void
load_commons (void)
{
  char name[GFC_MAX_SYMBOL_LEN + 1];
  gfc_common_head *p;

  mio_lparen ();

  while (peek_atom () != ATOM_RPAREN)
    {
      int flags;
      char* label;
      mio_lparen ();
      mio_internal_string (name);

      p = gfc_get_common (name, 1);

      mio_symbol_ref (&p->head);
      mio_integer (&flags);
      if (flags & 1)
        p->saved = 1;
      if (flags & 2)
        p->threadprivate = 1;
      p->use_assoc = 1;

      /* Get whether this was a bind(c) common or not.  */
      mio_integer (&p->is_bind_c);
      /* Get the binding label.  */
      label = read_string ();
#ifdef _RESOLUTION_
      if (strlen (label))
        p->binding_label = IDENTIFIER_POINTER (get_identifier (label));
#endif
      XDELETEVEC (label);

      mio_rparen ();
    }

  mio_rparen ();
}


/* Load equivalences.  The flag in_load_equiv informs mio_expr_ref of this
   so that unused variables are not loaded and so that the expression can
   be safely freed.  */

static void
load_equiv (void)
{
  gfc_equiv *head, *tail, *end, *eq;
  bool unused;

  mio_lparen ();
  in_load_equiv = true;

  end = gfc_current_ns->equiv;
  while (end != NULL && end->next != NULL)
    end = end->next;

  while (peek_atom () != ATOM_RPAREN) {
    mio_lparen ();
    head = tail = NULL;

    while(peek_atom () != ATOM_RPAREN)
      {
        if (head == NULL)
          head = tail = gfc_get_equiv ();
        else
          {
            tail->eq = gfc_get_equiv ();
            tail = tail->eq;
          }

        mio_pool_string (&tail->module);
        mio_expr (&tail->expr);
      }

    /* Unused equivalence members have a unique name.  In addition, it
       must be checked that the symbols are from the same module.  */
    unused = true;
    for (eq = head; eq; eq = eq->eq)
      {
        if (eq->expr->symtree->n.sym->module
              && head->expr->symtree->n.sym->module
              && strcmp (head->expr->symtree->n.sym->module,
                         eq->expr->symtree->n.sym->module) == 0
              && !check_unique_name (eq->expr->symtree->name))
          {
            unused = false;
            break;
          }
      }

    if (unused)
      {
        for (eq = head; eq; eq = head)
          {
            head = eq->eq;
            gfc_free_expr (eq->expr);
            free (eq);
          }
      }

    if (end == NULL)
      gfc_current_ns->equiv = head;
    else
      end->next = head;

    if (head != NULL)
      end = head;

    mio_rparen ();
  }

  mio_rparen ();
  in_load_equiv = false;
}


/* This function loads the sym_root of f2k_derived with the extensions to
   the derived type.  */
static void
load_derived_extensions (void)
{
  int symbol, j;
  gfc_symbol *derived;
  gfc_symbol *dt;
  gfc_symtree *st;
  pointer_info *info;
  char name[GFC_MAX_SYMBOL_LEN + 1];
  char module[GFC_MAX_SYMBOL_LEN + 1];
  const char *p;

  mio_lparen ();
  while (peek_atom () != ATOM_RPAREN)
    {
      mio_lparen ();
      mio_integer (&symbol);
      info = get_integer (symbol);
      derived = info->u.rsym.sym;

      /* This one is not being loaded.  */
      if (!info || !derived)
        {
          while (peek_atom () != ATOM_RPAREN)
            skip_list ();
          continue;
        }

      gcc_assert (derived->attr.flavor == FL_DERIVED);
      if (derived->f2k_derived == NULL)
        derived->f2k_derived = gfc_get_namespace (NULL, 0);

      while (peek_atom () != ATOM_RPAREN)
        {
          mio_lparen ();
          mio_internal_string (name);
          mio_internal_string (module);

          /* Only use one use name to find the symbol.  */
          j = 1;
          p = find_use_name_n (name, &j, false);
          if (p)
            {
              st = gfc_find_symtree (gfc_current_ns->sym_root, p);
              dt = st->n.sym;
              st = gfc_find_symtree (derived->f2k_derived->sym_root, name);
              if (st == NULL)
                {
                  /* Only use the real name in f2k_derived to ensure a single
                    symtree.  */
                  st = gfc_new_symtree (&derived->f2k_derived->sym_root, name);
                  st->n.sym = dt;
                  st->n.sym->refs++;
                }
            }
          mio_rparen ();
        }
      mio_rparen ();
    }
  mio_rparen ();
}


/* Recursive function to traverse the pointer_info tree and load a
   needed symbol.  We return nonzero if we load a symbol and stop the
   traversal, because the act of loading can alter the tree.  */

static int
load_needed (pointer_info *p)
{
  gfc_namespace *ns;
  pointer_info *q;
  gfc_symbol *sym;
  int rv;

  rv = 0;
  if (p == NULL)
    return rv;

  rv |= load_needed (p->left);
  rv |= load_needed (p->right);

  if (p->type != P_SYMBOL || p->u.rsym.state != NEEDED)
    return rv;

  p->u.rsym.state = USED;

  set_module_locus (&p->u.rsym.where);

  sym = p->u.rsym.sym;
  if (sym == NULL)
    {
      q = get_integer (p->u.rsym.ns);

      ns = (gfc_namespace *) q->u.pointer;
      if (ns == NULL)
        {
          /* Create an interface namespace if necessary.  These are
             the namespaces that hold the formal parameters of module
             procedures.  */

          ns = gfc_get_namespace (NULL, 0);
          associate_integer_pointer (q, ns);
        }

      /* Use the module sym as 'proc_name' so that gfc_get_symbol_decl
         doesn't go pear-shaped if the symbol is used.  */
      if (!ns->proc_name)
        gfc_find_symbol (p->u.rsym.module, gfc_current_ns,
                                 1, &ns->proc_name);

      sym = gfc_new_symbol (p->u.rsym.true_name, ns);
      sym->name = dt_lower_string (p->u.rsym.true_name);
      sym->module = gfc_get_string (p->u.rsym.module);
#ifdef _RESOLUTION_
      if (p->u.rsym.binding_label)
        sym->binding_label = IDENTIFIER_POINTER (get_identifier
                                                 (p->u.rsym.binding_label));
#endif

      associate_integer_pointer (p, sym);
    }

  mio_symbol (sym);
  sym->attr.use_assoc = 1;

  /* Mark as only or rename for later diagnosis for explicitly imported
     but not used warnings; don't mark internal symbols such as __vtab,
     __def_init etc.  */
  if (only_flag && sym->name[0] != '_' && sym->name[1] != '_')
    sym->attr.use_only = 1;
  if (p->u.rsym.renamed)
    sym->attr.use_rename = 1;

  return 1;
}



/* Compare two true_name structures.  */

static int
compare_true_names (void *_t1, void *_t2)
{
  true_name *t1, *t2;
  int c;

  t1 = (true_name *) _t1;
  t2 = (true_name *) _t2;

  c = ((t1->sym->module > t2->sym->module)
       - (t1->sym->module < t2->sym->module));
  if (c != 0)
    return c;

  return strcmp (t1->name, t2->name);
}


/* Given a true name, search the true name tree to see if it exists
   within the main namespace.  */

static gfc_symbol *
find_true_name (const char *name, const char *module)
{
  true_name t, *p;
  gfc_symbol sym;
  int c;

  t.name = gfc_get_string (name);
  if (module != NULL)
    sym.module = gfc_get_string (module);
  else
    sym.module = NULL;
  t.sym = &sym;

  p = true_name_root;
  while (p != NULL)
    {
      c = compare_true_names ((void *) (&t), (void *) p);
      if (c == 0)
        return p->sym;

      p = (c < 0) ? p->left : p->right;
    }

  return NULL;
}

/* Given a gfc_symbol pointer that is not in the true name tree, add it.  */

static void
add_true_name (gfc_symbol *sym)
{
  true_name *t;

  t = XCNEW (true_name);
  t->sym = sym;
  if (sym->attr.flavor == FL_DERIVED)
    t->name = dt_upper_string (sym->name);
  else
    t->name = sym->name;

  gfc_insert_bbt (&true_name_root, t, compare_true_names);
}

/* Recursive function to build the initial true name tree by
   recursively traversing the current namespace.  */

static void
build_tnt (gfc_symtree *st)
{
  const char *name;
  if (st == NULL)
    return;

  build_tnt (st->left);
  build_tnt (st->right);

  if (st->n.sym->attr.flavor == FL_DERIVED)
    name = dt_upper_string (st->n.sym->name);
  else
    name = st->n.sym->name;

  if (find_true_name (name, st->n.sym->module) != NULL)
    return;

  add_true_name (st->n.sym);
}


/* Initialize the true name tree with the current namespace.  */

static void
init_true_name_tree (void)
{
  true_name_root = NULL;
  build_tnt (gfc_current_ns->sym_root);
}


/* Recursively free a true name tree node.  */

static void
free_true_name (true_name *t)
{
  if (t == NULL)
    return;
  free_true_name (t->left);
  free_true_name (t->right);

  free (t);
}


/* Recursive function for cleaning up things after a module has been read.  */

static void
read_cleanup (pointer_info *p)
{
  gfc_symtree *st;
  pointer_info *q;

  if (p == NULL)
    return;

  read_cleanup (p->left);
  read_cleanup (p->right);

  if (p->type == P_SYMBOL && p->u.rsym.state == USED && !p->u.rsym.referenced)
    {
      gfc_namespace *ns;
      /* Add hidden symbols to the symtree.  */
      q = get_integer (p->u.rsym.ns);
      ns = (gfc_namespace *) q->u.pointer;

      if (!p->u.rsym.sym->attr.vtype
            && !p->u.rsym.sym->attr.vtab)
        st = gfc_get_unique_symtree (ns);
      else
        {
          /* There is no reason to use 'unique_symtrees' for vtabs or
             vtypes - their name is fine for a symtree and reduces the
             namespace pollution.  */
          st = gfc_find_symtree (ns->sym_root, p->u.rsym.sym->name);
          if (!st)
            st = gfc_new_symtree (&ns->sym_root, p->u.rsym.sym->name);
        }

      st->n.sym = p->u.rsym.sym;
      st->n.sym->refs++;

      /* Fixup any symtree references.  */
      p->u.rsym.symtree = st;
      resolve_fixups (p->u.rsym.stfixup, st);
      p->u.rsym.stfixup = NULL;
    }

  /* Free unused symbols.  */
  if (p->type == P_SYMBOL && p->u.rsym.state == UNUSED)
    gfc_free_symbol (p->u.rsym.sym);
}


/* It is not quite enough to check for ambiguity in the symbols by
   the loaded symbol and the new symbol not being identical.  */
static bool
check_for_ambiguous (gfc_symbol *st_sym, pointer_info *info)
{
  gfc_symbol *rsym;
  module_locus locus;
  symbol_attribute attr;

  if (st_sym->ns->proc_name && st_sym->name == st_sym->ns->proc_name->name)
    {
      gfc_error ("'%s' of module '%s', imported at %C, is also the name of the "
                 "current program unit", st_sym->name, module_name);
      return true;
    }

  rsym = info->u.rsym.sym;
  if (st_sym == rsym)
    return false;

  if (st_sym->attr.vtab || st_sym->attr.vtype)
    return false;

  /* If the existing symbol is generic from a different module and
     the new symbol is generic there can be no ambiguity.  */
  if (st_sym->attr.generic
        && st_sym->module
        && st_sym->module != module_name)
    {
      /* The new symbol's attributes have not yet been read.  Since
         we need attr.generic, read it directly.  */
      get_module_locus (&locus);
      set_module_locus (&info->u.rsym.where);
      mio_lparen ();
      attr.generic = 0;
      mio_symbol_attribute (&attr);
      set_module_locus (&locus);
      if (attr.generic)
        return false;
    }

  return true;
}

/* Read a module file.  */

static void
read_module (void)
{
  module_locus operator_interfaces, user_operators, extensions;
  const char *p;
  char name[GFC_MAX_SYMBOL_LEN + 1];
  int i;
  int ambiguous, j, nuse, symbol;
  pointer_info *info, *q;
  gfc_use_rename *u = NULL;
  gfc_symtree *st;
  gfc_symbol *sym;

  get_module_locus (&operator_interfaces);      /* Skip these for now.  */
  skip_list ();

  get_module_locus (&user_operators);
  skip_list ();
  skip_list ();

  /* Skip commons, equivalences and derived type extensions for now.  */
  skip_list ();
  skip_list ();

/******** ********/
  if (mod_version > 0) {
/******** ********/
     get_module_locus (&extensions);
     skip_list ();
/******** ********/
  }
/******** ********/

  mio_lparen ();

  /* Create the fixup nodes for all the symbols.  */

  while (peek_atom () != ATOM_RPAREN)
    {
      char* bind_label;
      require_atom (ATOM_INTEGER);
      info = get_integer (atom_int);

      info->type = P_SYMBOL;
      info->u.rsym.state = UNUSED;

      info->u.rsym.true_name = read_string ();
      info->u.rsym.module = read_string ();
      bind_label = read_string ();
      if (strlen (bind_label))
        info->u.rsym.binding_label = bind_label;
      else
        XDELETEVEC (bind_label);

      require_atom (ATOM_INTEGER);
      info->u.rsym.ns = atom_int;

      get_module_locus (&info->u.rsym.where);
      skip_list ();

      /* See if the symbol has already been loaded by a previous module.
         If so, we reference the existing symbol and prevent it from
         being loaded again.  This should not happen if the symbol being
         read is an index for an assumed shape dummy array (ns != 1).  */

      sym = find_true_name (info->u.rsym.true_name, info->u.rsym.module);

      if (sym == NULL
          || (sym->attr.flavor == FL_VARIABLE && info->u.rsym.ns !=1))
        continue;

      info->u.rsym.state = USED;
      info->u.rsym.sym = sym;

      /* Some symbols do not have a namespace (eg. formal arguments),
         so the automatic "unique symtree" mechanism must be suppressed
         by marking them as referenced.  */
      q = get_integer (info->u.rsym.ns);
      if (q->u.pointer == NULL)
        {
          info->u.rsym.referenced = 1;
          continue;
        }

      /* If possible recycle the symtree that references the symbol.
         If a symtree is not found and the module does not import one,
         a unique-name symtree is found by read_cleanup.  */
      st = find_symtree_for_symbol (gfc_current_ns->sym_root, sym);
      if (st != NULL)
        {
          info->u.rsym.symtree = st;
          info->u.rsym.referenced = 1;
        }
    }

  mio_rparen ();

  /* Parse the symtree lists.  This lets us mark which symbols need to
     be loaded.  Renaming is also done at this point by replacing the
     symtree name.  */

  mio_lparen ();

  while (peek_atom () != ATOM_RPAREN)
    {
      mio_internal_string (name);
      mio_integer (&ambiguous);
      mio_integer (&symbol);

      info = get_integer (symbol);

      /* See how many use names there are.  If none, go through the start
         of the loop at least once.  */
      nuse = number_use_names (name, false);
      info->u.rsym.renamed = nuse ? 1 : 0;

      if (nuse == 0)
        nuse = 1;

      for (j = 1; j <= nuse; j++)
        {
          /* Get the jth local name for this symbol.  */
          p = find_use_name_n (name, &j, false);

          if (p == NULL && strcmp (name, module_name) == 0)
            p = name;

          /* Exception: Always import vtabs & vtypes.  */
          if (p == NULL && name[0] == '_'
              && (strncmp (name, "__vtab_", 5) == 0
                  || strncmp (name, "__vtype_", 6) == 0))
            p = name;

          /* Skip symtree nodes not in an ONLY clause, unless there
             is an existing symtree loaded from another USE statement.  */
          if (p == NULL)
            {
              st = gfc_find_symtree (gfc_current_ns->sym_root, name);
              if (st != NULL)
                info->u.rsym.symtree = st;
              continue;
            }

          /* If a symbol of the same name and module exists already,
             this symbol, which is not in an ONLY clause, must not be
             added to the namespace(11.3.2).  Note that find_symbol
             only returns the first occurrence that it finds.  */
          if (!only_flag && !info->u.rsym.renamed
                && strcmp (name, module_name) != 0
                && find_symbol (gfc_current_ns->sym_root, name,
                                module_name, 0))
            continue;

          st = gfc_find_symtree (gfc_current_ns->sym_root, p);

          if (st != NULL)
            {
              /* Check for ambiguous symbols.  */
              if (check_for_ambiguous (st->n.sym, info))
                st->ambiguous = 1;
              info->u.rsym.symtree = st;
            }
          else
            {
              st = gfc_find_symtree (gfc_current_ns->sym_root, name);

              /* Create a symtree node in the current namespace for this
                 symbol.  */
              st = check_unique_name (p)
                   ? gfc_get_unique_symtree (gfc_current_ns)
                   : gfc_new_symtree (&gfc_current_ns->sym_root, p);
              st->ambiguous = ambiguous;

              sym = info->u.rsym.sym;

              /* Create a symbol node if it doesn't already exist.  */
              if (sym == NULL)
                {
                  info->u.rsym.sym = gfc_new_symbol (info->u.rsym.true_name,
                                                     gfc_current_ns);
                  info->u.rsym.sym->name = dt_lower_string (info->u.rsym.true_name);
                  sym = info->u.rsym.sym;
                  sym->module = gfc_get_string (info->u.rsym.module);
#ifdef _RESOLUTION_
                  if (info->u.rsym.binding_label)
                    sym->binding_label =
                      IDENTIFIER_POINTER (get_identifier
                                          (info->u.rsym.binding_label));
#endif
                }

              st->n.sym = sym;
              st->n.sym->refs++;

              if (strcmp (name, p) != 0)
                sym->attr.use_rename = 1;

              if (name[0] != '_'
                  || (strncmp (name, "__vtab_", 5) != 0
                      && strncmp (name, "__vtype_", 6) != 0))
                sym->attr.use_only = only_flag;

              /* Store the symtree pointing to this symbol.  */
              info->u.rsym.symtree = st;

              if (info->u.rsym.state == UNUSED)
                info->u.rsym.state = NEEDED;
              info->u.rsym.referenced = 1;
            }
        }
    }

  mio_rparen ();

  /* Load intrinsic operator interfaces.  */
  set_module_locus (&operator_interfaces);
  mio_lparen ();

  for (i = GFC_INTRINSIC_BEGIN; i != GFC_INTRINSIC_END; i++)
    {
      if (i == INTRINSIC_USER)
        continue;

      if (only_flag)
        {
          u = find_use_operator ((gfc_intrinsic_op) i);

          if (u == NULL)
            {
              skip_list ();
              continue;
            }

          u->found = 1;
        }

      mio_interface (&gfc_current_ns->op[i]);
      if (u && !gfc_current_ns->op[i])
        u->found = 0;
    }

  mio_rparen ();

  /* Load generic and user operator interfaces.  These must follow the
     loading of symtree because otherwise symbols can be marked as
     ambiguous.  */

  set_module_locus (&user_operators);

  load_operator_interfaces ();
  load_generic_interfaces ();

#if 0
  load_commons ();
  load_equiv ();
#endif

  /* At this point, we read those symbols that are needed but haven't
     been loaded yet.  If one symbol requires another, the other gets
     marked as NEEDED if its previous state was UNUSED.  */

  while (load_needed (pi_root));

  /* Make sure all elements of the rename-list were found in the module.  */

#if 1
  for (u = gfc_rename_list; u; u = u->next)
    {
      if (u->found)
        continue;

      if (u->op == INTRINSIC_NONE)
        {
          gfc_error ("Symbol '%s' referenced at %L not found in module '%s'",
                     u->use_name, &u->where, module_name);
          continue;
        }

      if (u->op == INTRINSIC_USER)
        {
          gfc_error ("User operator '%s' referenced at %L not found "
                     "in module '%s'", u->use_name, &u->where, module_name);
          continue;
        }

      gfc_error ("Intrinsic operator '%s' referenced at %L not found "
                 "in module '%s'", gfc_op2string (u->op), &u->where,
                 module_name);
    }

#endif
  /* Now we should be in a position to fill f2k_derived with derived type
     extensions, since everything has been loaded.  */
/******** ********/
  if (mod_version > 0) {
/******** ********/
     set_module_locus (&extensions);
     load_derived_extensions ();
/******** ********/
  }
/******** ********/

  /* Clean up symbol nodes that were never loaded, create references
     to hidden symbols.  */

  read_cleanup (pi_root);


}

unsigned char
check_file_type( char *filename)
{
  FILE *fp;
  char line[29];
  unsigned char ver = 0;

#ifdef _ZLIB_
  fp = fopen(filename,"r");
  if (fp!=NULL)
    {
      if (fgets(line, 29, fp)!=NULL)
        {
          if (strstr(line,"GFORTRAN module version '0' "))
            {
              ver = 0;
            }
          else if (strstr(line,"GFORTRAN module version '9' "))
            {
              ver = 0;
            }
          else
            {
              ver = 1;
            }
        }
      fclose(fp);
    }
  else
    {
      //printf("cannot open %s .\n",filename);
      exit(1);
    }
#endif /* _ZLIB_ */

  return ver;
}

#ifdef _ZLIB_
static void
read_module_to_tmpbuf ()
{
  /* We don't know the uncompressed size, so enlarge the buffer as
     needed.  */
  int cursz = 4096;
  int rsize = cursz;
  int len = 0;

  module_content = XNEWVEC (char, cursz);

  while (1)
    {
      int nread = gzread (module_fp, module_content + len, rsize);
      len += nread;
      if (nread < rsize)
        break;
      cursz *= 2;
      module_content = XRESIZEVEC (char, module_content, cursz);
      rsize = cursz - len;
    }

  module_content = XRESIZEVEC (char, module_content, len + 1);
  module_content[len] = '\0';

  module_pos = 0;

#if 0
  FILE *fp = fopen(".tmp.mod","w");
  fputs(module_content, fp);
  fclose(fp);
#endif

}
#endif /*_ZLIB_*/

void
import_module (gfc_use_list *module)
/*gfc_use_module (gfc_use_list *module)*/
{
  char *filename;
  int c, line, start, toLine;
  gfc_symtree *mod_symtree;
  gfc_use_list *use_stmt;
  locus old_locus = gfc_current_locus;


  gfc_current_locus = module->where;
  module_name = module->module_name;
  gfc_rename_list = module->rename;
  only_flag = module->only_flag;

  filename = XALLOCAVEC (char, strlen(modincludeDirv) + strlen(module_name) + strlen(MODULE_EXTENSION + 1) + 1);
  strcpy (filename, modincludeDirv);
  if (strlen(modincludeDirv)>0)
    {
      strcat (filename, "/"        );
    }
  strcat (filename, module_name);
  strcat (filename, MODULE_EXTENSION);

/********  *******/
  gzType = check_file_type( filename );
/********  *******/

  /* First, try to find an non-intrinsic module, unless the USE statement
     specified that the module is intrinsic.  */
  if (!gzType)
    {
      module_fp = NULL;
      if (!module->intrinsic)
        module_fp = gfc_open_included_file (filename, true, true);
      if (module_fp == NULL)
        gfc_fatal_error ("Can't open module file '%s' for reading at %C: %s",
                         filename, xstrerror(errno));
    }
#ifdef _ZLIB_
  else
    {
      module_fp = NULL;
      if (!module->intrinsic)
        module_fp = gzopen_included_file (filename, true, true);
      if (module_fp == NULL)
        gfc_fatal_error ("Can't open module file '%s' for reading at %C: %s",
                         filename, xstrerror (errno));
    }
#endif /*_ZLIB_*/

  /* Check that we haven't already USEd an intrinsic module with the
     same name.  */

  mod_symtree = gfc_find_symtree (gfc_current_ns->sym_root, module_name);
  if (mod_symtree && mod_symtree->n.sym->attr.intrinsic)
    gfc_error ("Use of non-intrinsic module '%s' at %C conflicts with "
               "intrinsic module name used previously", module_name);


  iomode = IO_INPUT;
  module_line = 1;
  module_column = 1;
  start = 0;

#ifdef _ZLIB_
  if (gzType)
    {
      read_module_to_tmpbuf ();
      gzclose (module_fp);
    }
#endif /*_ZLIB_*/

  if (gzType)
    toLine = 1;
  else
    toLine = 2;
  /* Skip the first two lines of the module, after checking that this is
     a gfortran module file.  */
  line = 0;
  while (line < toLine)
    {
      c = module_char ();
      if (c == EOF)
        bad_module ("Unexpected end of module");
      if (start++ < 3)
        parse_name (c);
      if ((start == 1 && strcmp (atom_name, "GFORTRAN") != 0)
          || (start == 2 && strcmp (atom_name, " module") != 0))
        gfc_fatal_error ("File '%s' opened at %C is not a GFORTRAN module "
                         "file", filename);
      if (start == 3)
        {
          if (strcmp (atom_name, " version") != 0
              || module_char () != ' '
              || parse_atom () != ATOM_STRING)
            gfc_fatal_error ("Parse error when checking module version"
                             " for file '%s' opened at %C", filename);

        /*if (strcmp (atom_string, MOD_VERSION))*/
          if (strcmp (atom_string, "0") && strcmp (atom_string, "9") && strcmp (atom_string, "12"))
            {
              gfc_fatal_error ("Wrong module version '%s' (expected '%s') "
                               "for file '%s' opened at %C", atom_string,
                               MOD_VERSION, filename);
            }
          
          mod_version = atoi(atom_string);

          free (atom_string);
        }
      if (c == '\n')
        line++;
    }

  init_pi_tree ();
  init_true_name_tree ();

  read_module ();

#if 0
  free_true_name (true_name_root);
  true_name_root = NULL;

  free_pi_tree (pi_root);
  pi_root = NULL;

  fclose (module_fp);

  use_stmt = gfc_get_use_list ();
  *use_stmt = *module;
  use_stmt->next = gfc_current_ns->use_stmts;
  gfc_current_ns->use_stmts = use_stmt;

  gfc_current_locus = old_locus;
#endif
}

void
gfc_free_use_stmts (gfc_use_list *use_stmts)
{
  gfc_use_list *next;
  for (; use_stmts; use_stmts = next)
    {
      gfc_use_rename *next_rename;

      for (; use_stmts->rename; use_stmts->rename = next_rename)
        {
          next_rename = use_stmts->rename->next;
          free (use_stmts->rename);
        }
      next = use_stmts->next;
      free (use_stmts);
    }
}

