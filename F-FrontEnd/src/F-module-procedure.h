/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */


#ifndef _F_MODULE_PROCEDURE_H_
#define _F_MODULE_PROCEDURE_H_
#include "F-front.h"
#include "hash.h"





#define FOREACH_IN_HASH(h, s, tab) \
    for (h = FirstHashEntry(tab, s); \
         h != NULL; \
         h = NextHashEntry(s))


#define isValidString(x) ((x) != NULL && *(x) != '\0')





struct generic_procedure_info_record {
    const char *genericProcName;	/* A name of a generic
                                         * procedure. Also used as a
                                         * hash key. */
    HashEntry *hPtr;			/* A hash entry. */
    HashTable *modProcTbl;		/* A hash table of module
                                         * procedures which this
                                         * generic procedure consists
                                         * of. */
};
typedef struct generic_procedure_info_record *gen_proc_t;

#define GEN_PROC_NAME(gp)	((gp)->genericProcName)
#define GEN_PROC_HASH_ENTRY(gp)	((gp)->hPtr)
#define GEN_PROC_MOD_TABLE(gp)	((gp)->modProcTbl)


struct module_procedure_info_record {
    const char *modProcName;	/* A name of a module procedure. Also
                                 * used as a hash key. */

    gen_proc_t belongsTo;	/* A generic procedure which consists
                                 * of this module procedure. */
    HashEntry *hPtr;		/* A hash entry. */
    TYPE_DESC retType;		/* The return type of this procedure. */
    expv args;			/* A dummy arguments list of this
                                 * procedure. */
    /*
     * Note:
     *	The retType and EXPV_TYPE() of each element in argSpec must
     *	exist in regular manner.
     */
};
typedef struct module_procedure_info_record *mod_proc_t;

#define MOD_PROC_NAME(mp)	((mp)->modProcName)
#define MOD_PROC_HASH_ENTRY(gp)	((gp)->hPtr)
#define MOD_PROC_GEN_PROC(mp)	((mp)->belongsTo)
#define MOD_PROC_GEN_NAME(mp)	(GEN_PROC_NAME(MOD_PROC_GEN_PROC(mp)))
#define MOD_PROC_TYPE(mp)	((mp)->retType)
#define MOD_PROC_ARGS(mp)	((mp)->args)




extern void		module_procedure_manager_init(void);

extern gen_proc_t	find_generic_procedure(const char *name);
extern gen_proc_t	add_generic_procedure(const char *name,
                                              int *isNewPtr);
extern void		delete_generic_procedure(gen_proc_t gp);
extern void		delete_generic_procedure_by_name(const char *name);

extern mod_proc_t	find_module_procedure(const char *genName,
                                              const char *modName);
extern mod_proc_t	add_module_procedure(const char *genName,
                                             const char *modName,
                                             TYPE_DESC tp, expv args,
                                             int *isNewPtr);
extern void		delete_module_procedure(mod_proc_t mp);
extern void		delete_module_procdure_by_name(const char *genName,
                                                       const char *modName);

extern void		fixup_module_procedure(mod_proc_t mp);
extern void		fixup_module_procedures(gen_proc_t gp);
extern void		fixup_all_module_procedures(void);

extern void		dump_module_procedure(mod_proc_t mp, FILE *fd);
extern void		dump_module_procedures(gen_proc_t gp, FILE *fd);
extern void		dump_all_module_procedures(FILE *fd);

#endif /* ! _F_MODULE_PROCEDURE_H_ */
