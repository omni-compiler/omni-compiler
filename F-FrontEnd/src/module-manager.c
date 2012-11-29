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
    ID mid = XMALLOC(ID, sizeof(*mid));
    *mid = *id;

    // HACK: it seems too dirty
    if(ID_CLASS(mid) == CL_PARAM && ID_STORAGE(mid) == STG_UNKNOWN)
        ID_STORAGE(mid) = STG_SAVE;

    if(mid->use_assoc == NULL) {
        mid->use_assoc = XMALLOC(struct use_assoc_info *, sizeof(*(id->use_assoc)));
        mid->use_assoc->module_name = mod->name;
        mid->use_assoc->original_name = id->name;
    }
    ID_LINK_ADD(mid, mod->head, mod->last);
}

// TODO check this
#define AVAILABLE_ID(id)                           \
    ID_CLASS(id)   == CL_NAMELIST ||               \
    (!TYPE_IS_PRIVATE(ID_TYPE(id))                 \
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

/**
 * export public identifiers to module-manager.
 */
int
export_module(SYMBOL sym, ID ids, expv use_decls)
{
    ID id;
    list lp;
    struct depend_module * dep;
    struct module * mod = XMALLOC(struct module *, sizeof(struct module));
    extern int flag_do_module_cache;

    *mod = (struct module){0};
    mod->name = sym;

    FOREACH_ID(id, ids) {
        if(AVAILABLE_ID(id))
            add_module_id(mod, id);
    }

    FOR_ITEMS_IN_LIST(lp, use_decls) {
        dep = XMALLOC(struct depend_module *, sizeof(struct depend_module));
        dep->module_name = EXPR_SYM(LIST_ITEM(lp));
        if (mod->depend.last == NULL) {
            mod->depend.head = dep;
            mod->depend.last = dep;
        } else {
            mod->depend.last->next = dep;
            mod->depend.last = dep;
        }
    }

    if (flag_do_module_cache == TRUE)
        add_module(mod);

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
 * import public identifiers from module-manager.
 */
int
import_module(const SYMBOL name, struct module ** pmod)
{
    struct module * mod;
    for(mod = MODULE_MANAGER.head; mod != NULL; mod = mod->next) {
        if(mod->name == name) {
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
