#include "F-front.h"
#include "F-output-xcodeml.h"
#include "F-input-xmod.h"
#include "module-manager.h"

/**
 * \file module-manager.c
 */

/**
 * collection of fortran modules.
 */
struct module_manager {
    struct module * head;
    struct module * tail;
} MODULE_MANAGER;

static void
add_module(struct module * mod)
{
    if(MODULE_MANAGER.head == NULL) {
        MODULE_MANAGER.head = mod;
    }else {
        MODULE_MANAGER.tail->next = mod;
    }
    MODULE_MANAGER.tail = mod;
    MODULE_MANAGER.tail->next = NULL;
}

static void
add_module_id(struct module * mod, ID id)
{
    // NOTE: 'mid' stand for module id.
    TYPE_DESC tp;
    ID mid = XMALLOC(ID, sizeof(*mid));
    *mid = *id;

    // HACK: it seems too dirty
    if (ID_CLASS(mid) == CL_PARAM && ID_STORAGE(mid) == STG_UNKNOWN)
        ID_STORAGE(mid) = STG_SAVE;

    if (mid->use_assoc == NULL) {
        mid->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
        mid->use_assoc->module = mod;
        mid->use_assoc->module_name = MODULE_NAME(mod);
        mid->use_assoc->original_name = id->name;
    }

    /* NOTE: dirty code, but
     *       tagname of type may differs from one in LOCAL_SYMBOLS.
     *       So this code rescues the derived-type comparison.
     */
    tp = ID_TYPE(id);
    while (tp != NULL) {
        if (TYPE_TAGNAME(tp)) {
            ID tagname = TYPE_TAGNAME(tp);
            if (tagname->use_assoc == NULL) {
                tagname->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
                tagname->use_assoc->module = mod;
                tagname->use_assoc->module_name = MODULE_NAME(mod);
                tagname->use_assoc->original_name = tagname->name;
            }
        }
        tp = TYPE_REF(tp);
    }

    ID_LINK_ADD(mid, MODULE_ID_LIST(mod), MODULE_ID_LIST_LAST(mod));
}

#if 0
// TODO check this
#define AVAILABLE_ID(id)                           \
    ID_CLASS(id)   == CL_NAMELIST ||               \
    (ID_TYPE(id) && !TYPE_IS_PRIVATE(ID_TYPE(id))  \
    && (      ID_CLASS(id)   == CL_VAR             \
           || ID_CLASS(id)   == CL_ENTRY           \
           || ID_CLASS(id)   == CL_PARAM           \
           || ID_CLASS(id)   == CL_CONTAINS        \
           || ID_CLASS(id)   == CL_TAGNAME         \
           || ID_CLASS(id)   == CL_UNKNOWN         \
           || ID_CLASS(id)   == CL_GENERICS        \
           || ID_CLASS(id)   == CL_PROC            \
       ) \
    && (      ID_STORAGE(id) != STG_NONE           \
        ))
#else
#define AVAILABLE_ID(id)                           \
    ID_CLASS(id)   == CL_NAMELIST ||               \
    (TRUE                                          \
    && (      ID_CLASS(id)   == CL_VAR             \
           || ID_CLASS(id)   == CL_ENTRY           \
           || ID_CLASS(id)   == CL_PARAM           \
           || ID_CLASS(id)   == CL_CONTAINS        \
           || ID_CLASS(id)   == CL_TAGNAME         \
           || ID_CLASS(id)   == CL_UNKNOWN         \
           || ID_CLASS(id)   == CL_GENERICS        \
           || ID_CLASS(id)   == CL_PROC            \
       ) \
    && (      ID_STORAGE(id) != STG_NONE           \
        ))
#endif

/**
 * export public identifiers to module-manager.
 */
int
export_xmod(SYMBOL mod_name, SYMBOL submod_name, ID ids, expv use_decls)
{
    ID id;
    list lp;
    struct depend_module * dep;
    struct module * mod = XMALLOC(struct module *, sizeof(struct module));
    extern int flag_do_module_cache;

    *mod = (struct module){0};
    MODULE_NAME(mod) = mod_name;
    if (submod_name) {
        SUBMODULE_NAME(mod) = submod_name;
    }

    // add public id
    FOREACH_ID(id, ids) {
        if(AVAILABLE_ID(id))
            add_module_id(mod, id);
    }

    // make the list of module name which this module uses.
    FOR_ITEMS_IN_LIST(lp, use_decls) {
        dep = XMALLOC(struct depend_module *, sizeof(struct depend_module));
        MOD_DEP_NAME(dep) = EXPR_SYM(LIST_ITEM(lp));
        if (MODULE_DEPEND_LAST(mod) == NULL) {
            MODULE_DEPEND_HEAD(mod) = dep;
            MODULE_DEPEND_LAST(mod) = dep;
        } else {
            MOD_DEP_NEXT(MODULE_DEPEND_LAST(mod)) = dep;
            MODULE_DEPEND_LAST(mod) = dep;
        }
    }

    if (flag_do_module_cache == TRUE)
        add_module(mod);

    if (nerrors == 0)
        output_module_file(mod);

#if 0
    /* debug */
    printf("debug=");
    expr_print(CURRENT_STATEMENTS,stdout);
    printf("\n");
#endif

    return TRUE;
}

/**
 * export public identifiers to module-manager.
 */
int
export_module(SYMBOL sym, ID ids, expv use_decls)
{
    return export_xmod(sym, NULL, ids, use_decls);
}


/**
 * export public identifiers to module-manager.
 */
int
export_submodule(SYMBOL submod, SYMBOL mod, ID ids, expv use_decls)
{
    assert(submod != NULL);
    return export_xmod(mod, submod, ids, use_decls);
}

static int
import_xmod(const SYMBOL name, const SYMBOL submodule_name, struct module ** pmod)
{
    struct module * mod;
    for(mod = MODULE_MANAGER.head; mod != NULL; mod = MODULE_NEXT(mod)) {
        if(MODULE_NAME(mod) == name && SUBMODULE_NAME(mod) == submodule_name) {
            *pmod = mod;
            return TRUE;
        }
    }
    if(input_module_file(name, &mod)) {
        *pmod = mod;
        add_module(mod);
        return TRUE;
    }

    error("failed to import module '%s'", SYM_NAME(name));
    exit(1);
}

/**
 * import public identifiers from module-manager.
 */
int
import_module(const SYMBOL name, struct module ** pmod)
{
    return import_xmod(name, NULL, pmod);
}

/**
 * import public identifiers from module-manager.
 */
int
import_submodule(const SYMBOL name, const SYMBOL submodule_name, struct module ** pmod)
{
    return import_xmod(name, submodule_name, pmod);
}
