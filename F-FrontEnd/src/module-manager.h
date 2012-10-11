
/**
 * \file module-manager.h
 *
 * This file should include module manager interfaces.
 * Don't include compiler funtions.
 */

#ifndef _MODULE_MANAGER_H_
#define _MODULE_MANAGER_H_

#include "C-expr.h"  // SYMBOL
#include "F-ident.h" // ID

/**
 * use association information about ID.
 */
struct use_assoc_info {
    SYMBOL module_name;       /* name of module which the ID declared. */
    SYMBOL original_name;     /* original name of the ID. */
};

/**
 * list of module name.
 */
struct depend_module {
    struct depend_module * next;
    SYMBOL module_name;
};

/**
 * fortran module.
 */
struct module {
    struct module * next;
    struct {
        struct depend_module * head;
        struct depend_module * last;
    } depend;                 /* list of module name which this module depends on. */
    SYMBOL name;              /* name of this module. */
    ID head;                  /* public elements of this module. */
    ID last;
};

/**
 * import module form module manager.
 */
int import_module(const SYMBOL, struct module **);

/**
 * export public identifiers to module-manager.
 */
int export_module(const SYMBOL, ID, expv);

#endif /* _MODULE_MANAGER_H_ */
