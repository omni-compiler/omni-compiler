#include "F-module-procedure.h"

static HashTable genProcTbl;
static int isInitialized = FALSE;

static void
destroy_module_procedure(mod_proc_t mp) {
    if (mp != NULL) {
        free((void *)MOD_PROC_NAME(mp));
        free((void *)mp);
    }
}


static mod_proc_t
create_module_procedure(const char *genName, const char *modName,
                        TYPE_DESC tp, expv args) {
    int succeeded = FALSE;
    mod_proc_t ret = NULL;
    gen_proc_t gp = NULL;
    HashEntry *hPtr = NULL;
    int isNew = FALSE;

    if (!(isValidString(genName)) || !(isValidString(modName))) {
        goto Done;
    }

    gp = add_generic_procedure(genName, NULL);
    if (gp != NULL) {
        ret = (mod_proc_t)malloc(sizeof(*ret));
        if (ret != NULL) {
            (void)memset((void *)ret, 0, sizeof(*ret));
            MOD_PROC_NAME(ret) = (const char *)strdup(modName);
            MOD_PROC_GEN_PROC(ret) = gp;
            MOD_PROC_TYPE(ret) = tp;
            MOD_PROC_ARGS(ret) = args;
            hPtr = CreateHashEntry(GEN_PROC_MOD_TABLE(gp),
                                   MOD_PROC_NAME(ret),
                                   &isNew);
            if (hPtr != NULL && isNew == 1) {
                SetHashValue(hPtr, (ClientData)ret);
                MOD_PROC_HASH_ENTRY(ret) = hPtr;
                succeeded = TRUE;
            }
        }
    }

    Done:
    if (succeeded == FALSE) {
        if (hPtr != NULL && isNew == 1) {
            DeleteHashEntry(hPtr);
        }
        if (ret != NULL) {
            destroy_module_procedure(ret);
            ret = NULL;
        }
    }

    return ret;
}


static void
destroy_generic_procedure(gen_proc_t gp) {
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr;
            HashSearch sCtx;
            mod_proc_t mp;

            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                mp = (mod_proc_t)GetHashValue(hPtr);
                if (mp != NULL) {
                    destroy_module_procedure(mp);
                }
            }

            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                DeleteHashEntry(hPtr);
            }

            DeleteHashTable(tPtr);
        }

        free((void *)(GEN_PROC_NAME(gp)));

        free((void *)gp);
    }
}


static void
destroy_all_generic_procedures(void) {
    HashEntry *hPtr;
    HashSearch sCtx;
    gen_proc_t gp;

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        gp = (gen_proc_t)GetHashValue(hPtr);
        if (gp != NULL) {
            destroy_generic_procedure(gp);
        }
    }

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        DeleteHashEntry(hPtr);
    }
}


static gen_proc_t
create_generic_procedure(const char *name) {
    HashEntry *hPtr = NULL;
    int isNew = 0;
    int succeeded = FALSE;
    gen_proc_t ret = (gen_proc_t)malloc(sizeof(*ret));
    if (ret != NULL) {
        (void)memset((void *)ret, 0, sizeof(*ret));
        GEN_PROC_MOD_TABLE(ret) = (HashTable *)malloc(sizeof(HashTable));
        if (GEN_PROC_MOD_TABLE(ret) != NULL) {
            InitHashTable(GEN_PROC_MOD_TABLE(ret), HASH_STRING_KEYS);
            GEN_PROC_NAME(ret) = (const char *)strdup(name);
            hPtr = CreateHashEntry(&genProcTbl, GEN_PROC_NAME(ret), &isNew);
            if (hPtr != NULL && isNew == 1) {
                SetHashValue(hPtr, (ClientData)ret);
                GEN_PROC_HASH_ENTRY(ret) = hPtr;
                succeeded = TRUE;
            }
        }
    }
    if (succeeded == FALSE) {
        if (hPtr != NULL && isNew == 1) {
            DeleteHashEntry(hPtr);
        }
        if (ret != NULL) {
            destroy_generic_procedure(ret);
            ret = NULL;
        }
    }

    return ret;
}

void
module_procedure_manager_init(void) {
    if (is_in_module() == FALSE) {
        if (isInitialized == TRUE) {
            destroy_all_generic_procedures();
        }
        InitHashTable(&genProcTbl, HASH_STRING_KEYS);
        isInitialized = TRUE;
    }
}

gen_proc_t
find_generic_procedure(const char *name) {
    gen_proc_t gp = NULL;
    if (isValidString(name)) {
        HashEntry *hPtr = FindHashEntry(&genProcTbl, name);
        if (hPtr != NULL) {
            gp = (gen_proc_t)GetHashValue(hPtr);
        }
    }
    return gp;
}


gen_proc_t
add_generic_procedure(const char *name, int *isNewPtr) {
    gen_proc_t ret = find_generic_procedure(name);
    if (ret == NULL) {
        ret = create_generic_procedure(name);
        if (isNewPtr != NULL) {
            *isNewPtr = TRUE;
        }
    } else {
        if (isNewPtr != NULL) {
            *isNewPtr = FALSE;
        }
    }

    if (ret == NULL) {
        if (isNewPtr != NULL) {
            *isNewPtr = FALSE;
        }
    }

    return ret;
}


void
delete_generic_procedure(gen_proc_t gp) {
    if (gp != NULL) {
        HashEntry *hPtr = GEN_PROC_HASH_ENTRY(gp);
        destroy_generic_procedure(gp);
        DeleteHashEntry(hPtr);
    }
}


void
delete_generic_procedure_by_name(const char *name) {
    if (isValidString(name)) {
        gen_proc_t gp = find_generic_procedure(name);
        if (gp != NULL) {
            delete_generic_procedure(gp);
        }
    }
}


mod_proc_t
find_module_procedure(const char *genName, const char *modName) {
    gen_proc_t gp = NULL;
    mod_proc_t ret = NULL;

    if (!(isValidString(genName)) || !(isValidString(modName))) {
        goto Done;
    }

    gp = find_generic_procedure(genName);
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr = FindHashEntry(tPtr, modName);
            if (hPtr != NULL) {
                ret = (mod_proc_t)GetHashValue(hPtr);
            }
        }
    }

    Done:
    return ret;
}


mod_proc_t
add_module_procedure(const char *genName, const char *modName,
                     TYPE_DESC tp, expv args, int *isNewPtr) {
    mod_proc_t ret = find_module_procedure(genName, modName);
    if (ret == NULL) {
        ret = create_module_procedure(genName, modName, tp, args);
        if (isNewPtr != NULL) {
            *isNewPtr = TRUE;
        }
    } else {
        if (isNewPtr != NULL) {
            *isNewPtr = FALSE;
        }
    }

    if (ret == NULL) {
        if (isNewPtr != NULL) {
            *isNewPtr = FALSE;
        }
    }

    return ret;
}


void
delete_module_procedure(mod_proc_t mp) {
    if (mp != NULL) {
        HashEntry *hPtr = MOD_PROC_HASH_ENTRY(mp);
        destroy_module_procedure(mp);
        DeleteHashEntry(hPtr);
    }
}


void
delete_module_procdure_by_name(const char *genName, const char *modName) {
    if (isValidString(genName) && isValidString(modName)) {
        mod_proc_t mp = find_module_procedure(genName, modName);
        if (mp != NULL) {
            delete_module_procedure(mp);
        }
    }
}


static expv
generate_module_procedure_dummy_args(expv args) {
    expv ret = list0(LIST);
    expv v;
    list lp;
    TYPE_DESC tp;
    char symBuf[4096];
    SYMBOL s;

    FOR_ITEMS_IN_LIST(lp, args) {
        v = LIST_ITEM(lp);
        tp = EXPV_TYPE(v);
        if (tp == NULL) {
            tp = EXPV_TYPE(EXPR_ARG1(v));
        }
        if (tp != NULL) {
            tp = copy_type_partially(tp, FALSE);
        }
        snprintf(symBuf, sizeof(symBuf), "$[dummy]:%s$",
                 SYM_NAME(EXPR_SYM(EXPR_ARG1(v))));
        s = find_symbol(symBuf);
        v = expv_sym_term(IDENT, tp, s);
        list_put_last(ret, v);
    }

    return ret;
}


void
fixup_module_procedure(mod_proc_t mp) {
    if (mp != NULL) {
        const char *modName = NULL;
        SYMBOL s = NULL;

        if ((modName = MOD_PROC_NAME(mp)) != NULL &&
            (s = find_symbol(modName)) != NULL) {
            ID id = NULL;
            EXT_ID eId = NULL;
            TYPE_DESC tp = NULL;
            expv args = NULL;

            eId = find_ext_id(s);
            if (eId == NULL) {
                id = find_ident(s);
                if (id != NULL) {
                    eId = PROC_EXT_ID(id);
                }
            }
            if (eId != NULL &&
                (EXT_PROC_CLASS(eId) == EP_MODULE_PROCEDURE ||
                 EXT_PROC_CLASS(eId) == EP_PROC)) {
                tp = EXT_PROC_TYPE(eId);
                if (tp != NULL && TYPE_BASIC_TYPE(tp) == TYPE_FUNCTION) {
                    tp = TYPE_REF(tp);
                }
                args = EXT_PROC_ARGS(eId);
            }

            /*
             * NOTE:
             *	In order to avoid messing the type table up with
             *	duplicated types, create new types for the return type
             *	and the dummy argument types, in which don't have any
             *	attrinutes.
             */
            if (MOD_PROC_TYPE(mp) == NULL && tp != NULL) {
                MOD_PROC_TYPE(mp) = copy_type_partially(tp, FALSE);
            }
            if (MOD_PROC_ARGS(mp) == NULL && args != NULL) {
                expv newArgs = generate_module_procedure_dummy_args(args);
                if (newArgs == args) {
                    fatal("noenoe");
                }
                MOD_PROC_ARGS(mp) = newArgs;
            }
            if (eId != NULL) {
                MOD_PROC_EXT_ID(mp) = eId;
            }
        }
    }
}


void
fixup_module_procedures(gen_proc_t gp) {
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr;
            HashSearch sCtx;

            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                fixup_module_procedure((mod_proc_t)GetHashValue(hPtr));
            }
        }
    }
}


void
fixup_all_module_procedures(void) {
    HashEntry *hPtr;
    HashSearch sCtx;

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        fixup_module_procedures((gen_proc_t)GetHashValue(hPtr));
    }
}


static void
collect_module_procedure_types(mod_proc_t mp, expr l) {
    if (mp != NULL) {
        list lp;
        expv v;
        TYPE_DESC tp;

        if (MOD_PROC_TYPE(mp) != NULL) {
            v = list3(LIST, 
                      expv_int_term(INT_CONSTANT, type_INT, 1),
                      expv_any_term(IDENT, (void *)MOD_PROC_TYPE(mp)),
                      expv_any_term(IDENT, (void *)MOD_PROC_EXT_ID(mp)));
            list_put_last(l, v);
        }

        FOR_ITEMS_IN_LIST(lp, MOD_PROC_ARGS(mp)) {
            v = LIST_ITEM(lp);
            if (v != NULL) {
                tp = EXPV_TYPE(v);
                if (tp != NULL) {
                    v = list2(LIST,
                              expv_int_term(INT_CONSTANT, type_INT, 0),
                              expv_any_term(IDENT, (void *)tp));
                    list_put_last(l, v);
                }
            }
        }
    }
}


static void
collect_module_procedures_types(gen_proc_t gp, expr l) {
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr;
            HashSearch sCtx;

            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                collect_module_procedure_types((mod_proc_t)GetHashValue(hPtr),
                                               l);
            }
        }
    }
}


expr
collect_all_module_procedures_types(void) {
    expr ret = list0(LIST);
    HashEntry *hPtr;
    HashSearch sCtx;

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        collect_module_procedures_types((gen_proc_t)GetHashValue(hPtr), ret);
    }

    return ret;
}


void
dump_module_procedure(mod_proc_t mp, FILE *fd) {
    if (mp != NULL) {
        gen_proc_t gp = MOD_PROC_GEN_PROC(mp);
        const char *genName = (gp != NULL) ? GEN_PROC_NAME(gp) : "";
        const char *modName = MOD_PROC_NAME(mp);
        fprintf(fd, "'%s:%s', returns: ", genName, modName);
        if (MOD_PROC_TYPE(mp) != NULL) {
            print_type(MOD_PROC_TYPE(mp), fd, TRUE);
        } else {
            fprintf(fd, "unknown");
        }
        fprintf(fd, "\nargs =\n");
        if (MOD_PROC_ARGS(mp) != NULL) {
            list lp;
            expv v;
            FOR_ITEMS_IN_LIST(lp, MOD_PROC_ARGS(mp)) {
                v = LIST_ITEM(lp);
                fprintf(fd, "\t'%s', ", SYM_NAME(EXPR_SYM(v)));
                print_type(EXPV_TYPE(v), fd, TRUE);
            }
        } else {
            fprintf(stderr, "()");
        }
        fprintf(stderr, "\n");
    }
}


void
dump_module_procedures(gen_proc_t gp, FILE *fd) {
    if (gp != NULL) {
        HashTable *tPtr = GEN_PROC_MOD_TABLE(gp);
        if (tPtr != NULL) {
            HashEntry *hPtr;
            HashSearch sCtx;

            fprintf(fd, "interface '%s':\n", GEN_PROC_NAME(gp));
            FOREACH_IN_HASH(hPtr, &sCtx, tPtr) {
                dump_module_procedure((mod_proc_t)GetHashValue(hPtr), fd);
            }
            fprintf(fd, "end interface '%s'\n", GEN_PROC_NAME(gp));
        }
    }
}


void
dump_all_module_procedures(FILE *fd) {
    HashEntry *hPtr;
    HashSearch sCtx;

    FOREACH_IN_HASH(hPtr, &sCtx, &genProcTbl) {
        dump_module_procedures((gen_proc_t)GetHashValue(hPtr), fd);
    }
}
