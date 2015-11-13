/********************************************************************/
/********************************************************************/

#ifndef TRANS_MODULE_H
#define TRANS_MODULE_H

#include "import_module.h"
#include "hwint.h"
#include "safe-ctype.h"
#include "gfortran.h"
#if 0
#include "tsystem.h"
#endif
#include "insn-modes.h"
#include "export_module.h"


void gfc_init_options (unsigned int , 
                       struct cl_decoded_option *);
void gfc_use_module (gfc_use_list *);
void gfc_init_kinds (void);
void gfc_symbol_init_2 (void);
void gfc_use_module (gfc_use_list *);

int export_module(const SYMBOL, ID, expv);

#endif /* TRANS_MODULE_H */

