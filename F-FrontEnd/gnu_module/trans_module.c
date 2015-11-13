
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "trans_module.h"

char *modincludeDirv;

char *current_module_name = (char *) NULL;
#define INMODULE()    (current_module_name != NULL)

static struct cl_decoded_option *g77_new_decoded_options;

/* program unit control stack */

extern char *file_names[];
expv use_decls = NULL;
int mod_version = 0;
lineno_info *current_line;

int doImplicitUndef = FALSE;
char generic_result_on = 0;

gfc_use_list *module_list;
extern gfc_namespace *gfc_current_ns;
extern generic_procedure *top_gene_pro;
extern generic_procedure *cur_gene_pro;
extern generic_procedure *end_gene_pro;

/* default variable type */
BASIC_DATA_TYPE defaultIntType = TYPE_INT;
BASIC_DATA_TYPE defaultSingleRealType = TYPE_REAL;
BASIC_DATA_TYPE defaultDoubleRealType = TYPE_DREAL;
enum storage_class default_stg = STG_SAVE;

ID id_list = NULL;
ID top_gen_id = NULL;
ID end_gen_id = NULL;
ID last_ip = NULL;

typedef struct relation_list
{
    struct relation *next;
    gfc_symbol      *sym;
    ID               id;
    struct external_symbol *ext;
} relation;

relation  *top_relation = NULL;
relation  *cur_relation = NULL;
relation  *end_relation = NULL;

typedef struct symbol_tree_list
{
    gfc_symtree              *st;
    struct symbol_tree_list  *next;
}
symtree;

symtree *top_symtree = NULL;
symtree *cur_symtree = NULL;
symtree *end_symtree = NULL;


/*
 * C-expr-mem.c
 */
expr
make_enode(code,v)
     enum expr_code code;
     void *v;
{
    expr ep;

    ep = XMALLOC(expr,sizeof(*ep));
    ep->e_code = code;
    ep->e_line = current_line;
    ep->v.e_gen = v;
    return(ep);
}

expr list0(code)
     enum expr_code code;
{
    return(make_enode(code,NULL));
}

/*
 * F-compile-decl.c
 */
static void
set_implicit_type_declared_uc(UNIT_CTL uc, int c)
{
    int i = 1;
    i <<= (c - 'a');
    UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) |= i;
}

static int
is_implicit_type_declared_uc(UNIT_CTL uc, int c)
{
    int i = 1;
    i <<= (c - 'a');
    return UNIT_CTL_IMPLICIT_TYPE_DECLARED(uc) & i;
}


void
set_implicit_type_uc(UNIT_CTL uc, TYPE_DESC tp, int c1, int c2,
                     int ignore_declared_flag)
{
    int i;

    if (c1 == 0 || c2 == 0)
        return;

    if (c1 > c2) {
        error("characters out of order in IMPLICIT:%c-%c", c1, c2);
        return;
    }

    if (tp)
        TYPE_SET_IMPLICIT(tp);

    for (i = c1 ; i <= c2 ; ++i) {
        if (ignore_declared_flag) {
            UNIT_CTL_IMPLICIT_TYPES(uc)[i - 'a'] = tp;
        } else {
            if (!is_implicit_type_declared_uc(uc, i)) {
                UNIT_CTL_IMPLICIT_TYPES(uc)[i - 'a'] = tp;
                set_implicit_type_declared_uc(uc, i);
            } else {
                error("character '%c' already has IMPLICIT type", i);
            }
        }
    }
}

/**
 * cleanup UNIT_CTL for each procedure
 * Notes: local_external_symbols is not null cleared.
 */
void
cleanup_unit_ctl(UNIT_CTL uc)
{
    UNIT_CTL_CURRENT_PROC_NAME(uc) = NULL;
    UNIT_CTL_CURRENT_PROC_CLASS(uc) = CL_UNKNOWN;
    UNIT_CTL_CURRENT_PROCEDURE(uc) = NULL;
    UNIT_CTL_CURRENT_STATEMENTS(uc) = NULL;
    UNIT_CTL_CURRENT_BLK_LEVEL(uc) = 1;
    UNIT_CTL_CURRENT_EXT_ID(uc) = NULL;
    UNIT_CTL_CURRENT_STATE(uc) = OUTSIDE;
    UNIT_CTL_LOCAL_SYMBOLS(uc) = NULL;
    UNIT_CTL_LOCAL_STRUCT_DECLS(uc) = NULL;
    UNIT_CTL_LOCAL_COMMON_SYMBOLS(uc) = NULL;
    UNIT_CTL_LOCAL_LABELS(uc) = NULL;
    UNIT_CTL_IMPLICIT_DECLS(uc) = list0(LIST);
    UNIT_CTL_EQUIV_DECLS(uc) = list0(LIST);
    UNIT_CTL_USE_DECLS(uc) = list0(LIST);

}

static uint32_t get_type_attr ( symbol_attribute s_attr )
{
    uint32_t attr;

    attr = 0;
    if(s_attr.allocatable          )  attr |= TYPE_ATTR_ALLOCATABLE;
    if(s_attr.external             )  attr |= TYPE_ATTR_EXTERNAL;
    if(s_attr.intrinsic            )  attr |= TYPE_ATTR_INTRINSIC;
    if(s_attr.optional             )  attr |= TYPE_ATTR_OPTIONAL;
    if(s_attr.pointer              )  attr |= TYPE_ATTR_POINTER;
    if(s_attr.target               )  attr |= TYPE_ATTR_TARGET;
    if(s_attr.save==SAVE_EXPLICIT  )  attr |= TYPE_ATTR_SAVE;
    if(s_attr.intent==INTENT_IN    )  attr |= TYPE_ATTR_INTENT_IN;
    if(s_attr.intent==INTENT_OUT   )  attr |= TYPE_ATTR_INTENT_OUT;
    if(s_attr.intent==INTENT_INOUT )  attr |= TYPE_ATTR_INTENT_INOUT;
    if(s_attr.flavor == FL_PARAMETER) attr |= TYPE_ATTR_PARAMETER;

    return attr;
}

static uint32_t get_type_attr1( symbol_attribute s_attr )
{
    uint32_t attr;

    attr = 0;
    if(s_attr.allocatable          )  attr |= TYPE_ATTR_ALLOCATABLE;
    if(s_attr.external             )  attr |= TYPE_ATTR_EXTERNAL;
    if(s_attr.intrinsic            )  attr |= TYPE_ATTR_INTRINSIC;
    if(s_attr.optional             )  attr |= TYPE_ATTR_OPTIONAL;
    if(s_attr.pointer              )  attr |= TYPE_ATTR_POINTER;
    if(s_attr.target               )  attr |= TYPE_ATTR_TARGET;
    if(s_attr.save==SAVE_EXPLICIT  )  attr |= TYPE_ATTR_SAVE;
   /*
    if(s_attr.intent==INTENT_IN    )  attr |= TYPE_ATTR_INTENT_IN;
    if(s_attr.intent==INTENT_OUT   )  attr |= TYPE_ATTR_INTENT_OUT;
    if(s_attr.intent==INTENT_INOUT )  attr |= TYPE_ATTR_INTENT_INOUT;
    */
    if(s_attr.flavor == FL_PARAMETER) attr |= TYPE_ATTR_PARAMETER;

    return attr;
}

static BASIC_DATA_TYPE get_basic_type( gfc_typespec ts )
{
    BASIC_DATA_TYPE basic_type;

    switch(ts.type) {
       case BT_INTEGER:
            basic_type = TYPE_INT;
         break;
       case BT_LOGICAL:
            basic_type = TYPE_LOGICAL;
            break;
       case BT_REAL:
            if (ts.kind == 4) {
               basic_type = TYPE_REAL;
            } else if (ts.kind == 8) {
               basic_type = TYPE_DREAL;
            } else {
               exit(99);
            } 
            break;
       case BT_COMPLEX:
            if (ts.kind == 4) {
               basic_type = TYPE_COMPLEX;
            } else if (ts.kind == 8) {
               basic_type = TYPE_DCOMPLEX;
            } else {
               exit(99);
            } 
            break;
       case BT_DERIVED:
            basic_type = TYPE_STRUCT;
            break;
       case BT_CHARACTER:
            basic_type = TYPE_CHAR;
            break;
       case BT_PROCEDURE:
            basic_type = TYPE_SUBR;
            basic_type = TYPE_FUNCTION;
            break;
       case BT_CLASS:
       case BT_HOLLERITH:
       case BT_VOID:
       default:
            basic_type = TYPE_UNKNOWN;
    }

    return basic_type;
}

static ID symbol2id( gfc_symbol *sym )
{
    ID       id;
    relation *rela;

    id   = NULL;
    rela = top_relation;
    while (rela!=NULL) {
       if (rela->sym == sym) {
          id = rela->id;
          break;
       }
       rela = rela->next;
    }
    return id;
}

static void add_use_module ( gfc_symbol *sym )
{
    struct list_node        *list, *list_t;

    if (sym->module == NULL) return;
    if (strcasecmp(module_list->module_name, sym->module)!=0) {
       list = use_decls->v.e_lp;
       while (list!=NULL) {
          if (strcasecmp(list->l_item->v.e_sym->s_name, sym->module)==0) {
             return;
          }
          list = list->l_next;
       }

       list                  = XMALLOC(struct list_node *, sizeof(*list));
       list->l_item          = XMALLOC(expv             *, sizeof(*list->l_item));
       list->l_item->v.e_sym = XMALLOC(SYMBOL           *, sizeof(struct symbol));
       list->l_item->e_line  = XMALLOC(struct line_info *, sizeof(*list->l_item->e_line));
       list->l_item->v.e_sym->s_name = sym->module;
       list->l_item->v.e_sym->s_type = S_IDENT;
       list->l_item->e_code = IDENT;

       list_t = use_decls->v.e_lp;
       list->l_next = list_t;
       use_decls->v.e_lp = list;
    }
}


static struct use_assoc_info *make_assoc_info( gfc_symbol *sym , int flag )
{
    struct use_assoc_info  *as_info = NULL;

    if (strcasecmp(module_list->module_name, sym->module)!=0) {
       as_info = XMALLOC(struct use_assoc_info *, sizeof(*as_info));
       as_info->original_name = XMALLOC(struct symbol *, sizeof(*as_info->original_name));
       as_info->module_name   = XMALLOC(struct symbol *, sizeof(*as_info->module_name  ));

       as_info->module_name->s_name   = sym->module;
       as_info->module_name->s_type   = S_IDENT;
       as_info->original_name->s_name = sym->name;
       as_info->original_name->s_type = S_IDENT;
    }

    return as_info;
}

static TYPE_DESC make_type_w( gfc_symbol * , int );

static TYPE_DESC make_type( gfc_typespec      ts ,
                            symbol_attribute  attr,
                            gfc_array_spec    *as ,
                            int flag  )
{
    struct expression_node  *kind;
    struct type_descriptor  *type, *ref0, *ref1, *e_type, *pre_type, *top_type;
    BASIC_DATA_TYPE basic_type;
    struct gfc_symtree      *symt;
    struct gfc_symbol       *sym;

    int rank, dim;

    basic_type = get_basic_type( ts );
    type = new_type_desc();
    top_type = type;

    if(!(attr.dimension)) {

     /*type = type_basic(basic_type);*/
       type->basic_type = basic_type;
       type->array_info.n_dim = 0;

       if (ts.type == BT_CHARACTER) {
          if (flag != 2) {
             if (ts.u.cl->length == NULL) {

                type->ref = new_type_desc();
                type->ref->basic_type = basic_type;
                type->size = -1;
                type->ref->size = -1;

             } else {
                e_type = type_basic(TYPE_INT);

                kind = XMALLOC(expv, sizeof(*kind));
                kind->e_code = INT_CONSTANT;
                kind->v.e_llval = *(ts.u.cl->length->value.character.string);

                ref0 = new_type_desc();
                TYPE_BASIC_TYPE(ref0) = basic_type;
                ref0->size = *(ts.u.cl->length->value.character.string); 

                type->ref = ref0;
                type->ref->leng = kind;
                type->ref->leng->e_type = e_type;
                type->size = *(ts.u.cl->length->value.character.string); 

                type->ref->attr.type_attr_flags = get_type_attr(attr);

             }
          } else {
             e_type = type_basic(TYPE_INT);

             kind = XMALLOC(expv, sizeof(*kind));
             kind->e_code = INT_CONSTANT;
             kind->v.e_llval = *(ts.u.cl->length->value.character.string);
             kind->e_type = e_type;

             type->size = *(ts.u.cl->length->value.character.string);
             type->leng = kind;
          }

       } else if (ts.type == BT_DERIVED) {

          if (attr.flavor == FL_PARAMETER) {
            ;
          } else if (attr.flavor == FL_DERIVED) {
            ;
          } else {
            ;
          }

       } else {

          if (attr.flavor == FL_PARAMETER) {
             e_type = type_basic(TYPE_INT);

             kind = XMALLOC(expv, sizeof(*kind));
             kind->e_code = INT_CONSTANT;
             kind->v.e_llval = ts.kind;

             type->kind = kind;
             type->kind->e_type = e_type;
             
             type->ref = new_type_desc();
             type->ref->basic_type = basic_type;
             type->ref->kind = kind;
          } else if (attr.flavor == FL_DERIVED) {
             basic_type = TYPE_STRUCT;
             type->basic_type = basic_type;
          } else {
#if 1
             if (ts.kind == 4 && (flag == 0 || flag == 2) && attr.generic) {
#else
             if (attr.generic) {
#endif
                type->kind = NULL;
                type->ref  = NULL;
             } else {
                e_type = new_type_desc();
                e_type->basic_type = TYPE_INT;
 
                kind = XMALLOC(expv, sizeof(*kind));
                kind->e_code = INT_CONSTANT;
                kind->v.e_llval = ts.kind;
   
                type->kind = kind;
                type->kind->e_type = e_type;
             
                if(attr.external) {
                   type->ref = NULL;
                } else {
                   /*if (attr.function) {*/
                   if(flag == 2) {
                      type->ref = NULL;
                   /*} else if (attr.dummy) {*/
                   } else {
                      type->ref = new_type_desc();
                      type->ref->basic_type =basic_type;
                      type->ref->kind = kind;
                   }
                }
             }
          }
       }

       type->attr.type_attr_flags = get_type_attr(attr);

    } else {  /* (attr.dimension) */

       type->basic_type = get_basic_type( ts ) ;
       type->ref = new_type_desc();
       type->ref->basic_type = get_basic_type( ts ) ;
          
       type->kind = XMALLOC(struct expression_node *, sizeof(*type->kind));
       type->kind->e_code = INT_CONSTANT;
       type->kind->v.e_llval = ts.kind;

       rank = as->rank;

       if(!(attr.allocatable)) {
          ;
       }
       if(!(attr.pointer)) {
          ;
       }

       pre_type = type;
       for(dim=rank;0<dim;dim--) {
          if(dim<rank) {
             type = new_type_desc();
          }

          type->basic_type = TYPE_ARRAY;
          if (rank==dim) {
             type->attr.type_attr_flags = get_type_attr(attr);
          } else {
             type->attr.type_attr_flags = get_type_attr1(attr);
          }
          type->array_info.dim_fixed  = 1;
          type->array_info.n_dim      = dim;

          if(attr.allocatable || attr.pointer) {
             type->array_info.assume_kind  = ASSUMED_SHAPE;
          } else {
             type->array_info.assume_kind  = ASSUMED_NONE;
             if(as->type == AS_ASSUMED_SHAPE) {
                type->array_info.assume_kind  = ASSUMED_SHAPE;
             }
             if(as->type == AS_ASSUMED_SIZE) {
                type->array_info.assume_kind  = ASSUMED_SIZE;
             }
             if(as->type != AS_ASSUMED_SHAPE && as->type != AS_ASSUMED_SIZE) {

                type->array_info.assume_kind  = ASSUMED_NONE;
                type->array_info.dim_lower = XMALLOC(expv , sizeof(*type->array_info.dim_lower));
                type->array_info.dim_upper = XMALLOC(expv , sizeof(*type->array_info.dim_upper));

                if (as->lower[dim-1]->expr_type == EXPR_VARIABLE) {

                   sym = as->lower[dim-1]->symtree->n.sym;
                   type->array_info.dim_lower->e_type = make_type_w(sym, 0);
                   type->array_info.dim_lower->v.e_sym = XMALLOC(SYMBOL, sizeof(struct symbol));
                   type->array_info.dim_lower->v.e_sym->s_name = sym->name;
                   type->array_info.dim_lower->v.e_sym->s_type = S_IDENT;
                   type->array_info.dim_lower->e_code = F_VAR ;

                } else if (as->lower[dim-1]->expr_type == EXPR_OP) {
                   ;
                } else {

                   type->array_info.dim_lower->e_code    = INT_CONSTANT;
                   type->array_info.dim_lower->v.e_llval = *as->lower[dim-1]->value.integer->_mp_d
                                                         * as->lower[dim-1]->value.integer->_mp_size;
                }

                if (as->upper[dim-1]->expr_type == EXPR_VARIABLE) {

                   sym = as->upper[dim-1]->symtree->n.sym;
                   type->array_info.dim_upper->e_type = make_type_w(sym, 0);
                   type->array_info.dim_upper->v.e_sym = XMALLOC(SYMBOL, sizeof(struct symbol));
                   type->array_info.dim_upper->v.e_sym->s_name = sym->name;
                   type->array_info.dim_upper->v.e_sym->s_type = S_IDENT;
                   type->array_info.dim_upper->e_code = F_VAR ;

                } else if (as->upper[dim-1]->expr_type == EXPR_OP) {
                   ;
                } else {

                   type->array_info.dim_upper->e_code    = INT_CONSTANT;
                   type->array_info.dim_upper->v.e_llval = *as->upper[dim-1]->value.integer->_mp_d
                                                         * as->upper[dim-1]->value.integer->_mp_size;
                }

                if (as->lower[dim-1]->expr_type == EXPR_VARIABLE || 
                    as->upper[dim-1]->expr_type == EXPR_VARIABLE) {
                   type->array_info.dim_size  = XMALLOC(expv , sizeof(*type->array_info.dim_size));
                   type->array_info.dim_size->e_code = DIV_EXPR;
                }
             }
          }

          if(dim<rank) {
             pre_type->ref = type;
             pre_type = type;
          }
       }
       ref0  = XMALLOC(struct type_descriptor *, sizeof(*ref0));
       ref0->basic_type = basic_type;

       if(!attr.dummy) {
          ref0->attr.type_attr_flags = get_type_attr(attr);
       } else {
          ref0->attr.type_attr_flags = get_type_attr1(attr);
       }

       pre_type->ref = ref0;
       pre_type = ref0;

       if (ts.type == BT_CHARACTER) {
           e_type = type_basic(TYPE_INT);

          if (ts.u.cl->length != NULL) {

             ref1 = new_type_desc();
             ref1->basic_type = basic_type;
             if(!attr.dummy && attr.flavor != FL_PARAMETER) {
                ref1->attr.type_attr_flags = get_type_attr(attr);
             }

             ref0->ref = ref1;
             pre_type = ref1;

             kind = XMALLOC(struct expression_node *, sizeof(*kind));
             kind->e_code = INT_CONSTANT;
             kind->v.e_llval = *(ts.u.cl->length->value.character.string);

             kind->e_type  = new_type_desc();
             kind->e_type->basic_type = TYPE_INT;
             kind->e_type->ref = NULL;


             ref1->leng = kind;
             ref0->size = *(ts.u.cl->length->value.character.string); 
             ref1->size = *(ts.u.cl->length->value.character.string); 

          } else {
             ref1 = new_type_desc();
             ref1->basic_type = basic_type;
             if(!attr.dummy && attr.flavor != FL_PARAMETER) {
                ref1->attr.type_attr_flags = get_type_attr(attr);
             }

             ref0->ref = ref1;
             pre_type = ref1;

             ref0->size = -1; 
             ref1->size = -1;
          }
       } else {
#if 1
/*
          if(attr.flavor == FL_PARAMETER) 
*/
          {
             ref1 = new_type_desc();
             ref1->basic_type = basic_type;
/******/
           /*if (attr.flavor != FL_PARAMETER) {*/
             if (attr.flavor != FL_PARAMETER && attr.flavor != FL_UNKNOWN) {
                if (!attr.dummy) {
                    ref1->attr.type_attr_flags = get_type_attr (attr);
                } else {
                    ref1->attr.type_attr_flags = get_type_attr1(attr);
                }
             }
/******/

             ref0->ref = ref1;
             pre_type = ref1;
          }
#endif
          kind  = XMALLOC(struct expression_node *, sizeof(*kind));
          kind->e_code = INT_CONSTANT;
          kind->v.e_llval = ts.kind;

          kind->e_type  = XMALLOC(struct type_descriptor *, sizeof(*kind->e_type));
          kind->e_type->basic_type = TYPE_INT;
          kind->e_type->ref = NULL;
          ref0->kind = kind;
          ref1->kind = kind;
       }
    }
    return top_type;

}


static struct external_symbol *make_extID(gfc_symbol *sym, int flag)
{
    gfc_formal_arglist *fa;
    struct expression_node  *args1, *result;
    struct expression_node  *l_item0, *l_item1, *l_item2;
    struct list_node        *e_lp0, *e_lp1, *e_lp2, *pre_node;
    struct external_symbol  *extid;
    SYMBOL arg_name;

    fa = sym->formal;

    args1 = NULL;
    if (fa != NULL || !sym->attr.generic) {
       args1 = XMALLOC(struct expression_node *, sizeof(*args1));
       args1->e_code = LIST;
       args1->v.e_lp = NULL;
    }

    while(fa!=NULL){

       e_lp1    = XMALLOC(struct list_node       *, sizeof(*e_lp1));
       l_item1  = XMALLOC(expv                    , sizeof(*l_item1));
       e_lp2    = XMALLOC(struct list_node       *, sizeof(*e_lp2));
       l_item2  = XMALLOC(expv                    , sizeof(*l_item2));
       arg_name = XMALLOC(SYMBOL                  , sizeof(struct symbol));

       l_item1->e_code = LIST;
       l_item1->v.e_lp = e_lp2;
       l_item2->e_code = IDENT;
       l_item2->v.e_lp = arg_name;
       e_lp1->l_nItems = 1;
       e_lp1->l_item = l_item1;
       e_lp2->l_nItems = 1;
       e_lp2->l_item = l_item2;

       l_item2->v.e_sym = arg_name;

       if (fa->sym != NULL) {

          arg_name->s_name = fa->sym->name;
          if (fa->sym->attr.flavor == FL_VARIABLE) {
             l_item2->e_type = make_type_w(fa->sym, 1);
          } else if (fa->sym->attr.flavor == FL_PROCEDURE) {
             l_item2->entry_ext_id = make_extID ( fa->sym , fa->sym->name );
             l_item2->entry_ext_id->name->s_name = fa->sym->name;
             l_item2->entry_ext_id->info.proc_info.type = make_type_w(fa->sym, 1);
          } else {
             ;
          }

       }

       if(args1->v.e_lp == NULL) {
          args1->v.e_lp = e_lp1;
          pre_node = e_lp1;
       } else {
          pre_node->l_next = e_lp1;
          pre_node = e_lp1;
       }

       fa = fa->next;
    }

    extid    = XMALLOC(struct external_symbol *, sizeof(*extid));
    extid->name = XMALLOC(SYMBOL               , sizeof(struct symbol));

    extid->info.proc_info.id_list = NULL;
    extid->stg = STG_EXT;
    extid->is_defined = 1;
    extid->info.proc_info.args = args1;
  /*extid->info.proc_info.body    = ;*/
  /*extid->info.proc_info.id_list = ;*/
    if (sym->attr.generic) {
       if (sym->attr.implicit_pure) {
          extid->info.proc_info.ext_proc_class = EP_MODULE_PROCEDURE;
       } else {
          extid->info.proc_info.ext_proc_class = EP_INTERFACE;
       }
    } else {
       extid->info.proc_info.ext_proc_class = EP_PROC;
    }

    /*
    extid->info.proc_info.type = id->type;
    extid->name = id->name;
    */
    return extid;

}


static TYPE_DESC make_type_w( gfc_symbol  *sym ,
                              int          flag )
{
    struct type_descriptor  *type;

    if( sym->type == NULL) {
       type = make_type(sym->ts, sym->attr, sym->as, 0);
       sym->type = (void *)type;
    } else {
       type = (struct type_descriptor *)sym->type;
    }

    return type;
}

static ID  make_id( gfc_symbol *sym , const char *name )
{
    gfc_formal_arglist *fa;
    gfc_component      *component;
    gfc_constructor_base *cp;
    gfc_constructor *c;
    gfc_expr **ep;
    gfc_expr  *e;

    ID                      id, member, pre_member;
    struct expression_node  *args1, *result;
    struct expression_node  *l_item0, *l_item1, *l_item2;
    struct list_node        *e_lp0, *e_lp1, *e_lp2, *pre_node;
  /*struct external_symbol  *extid;*/
    SYMBOL arg_name;

       id = XMALLOC(ID,sizeof(*id));
       id->type = new_type_desc();
       id->name = XMALLOC(SYMBOL, sizeof(struct symbol));
       id->name->s_name = name;

       if (sym->attr.flavor == FL_MODULE) {
   
          id->class = CL_MODULE;
          id->stg   = STG_EXT;
          id->type->basic_type = TYPE_MODULE;
          id->type->attr.exflags = 16;
   
          id->use_assoc = make_assoc_info( sym , 0 );

          add_use_module( sym );

       } else if (sym->attr.flavor == FL_PARAMETER) {

          if(sym->attr.dimension) {
             id->class = CL_VAR;
             id->stg   = STG_SAVE;
          } else {
             id->class = CL_PARAM;
             id->stg   = STG_UNKNOWN;
             if (sym->attr.in_common) {
                id->stg = STG_COMMON;
             }
          }

          id->is_declared  = 1;
          id->order        = 1;
          id->attr.type_attr_flags = 0;
          id->attr.exflags = 2;
   
          id->use_assoc = make_assoc_info( sym , 0 );
          id->type = make_type_w( sym, 4 );
          id->type->attr.type_attr_flags  |= TYPE_ATTR_PARAMETER;

          result = XMALLOC(struct expression_node *, sizeof(*result));
          result->e_type = new_type_desc();
          result->e_type->basic_type = get_basic_type( sym->ts );

          if (!sym->attr.dimension) {

             if (sym->ts.type == BT_INTEGER) {
                result->e_code = INT_CONSTANT;
                result->v.e_llval = *(sym->value->value.integer->_mp_d) 
                                  * sym->value->value.integer->_mp_size ;
             } else if (sym->ts.type == BT_REAL) {
                result->e_code = FLOAT_CONSTANT;
                result->v.e_lfval = *(sym->value->value.real->_mpfr_d)
                                  * sym->value->value.real->_mpfr_sign ;
             } else if (sym->ts.type == BT_COMPLEX) {
                result->e_code = COMPLEX_CONSTANT;
                result->v.e_lp = XMALLOC(struct list_node *, sizeof(*result->v.e_lp));
#ifdef _MPCLIB_
                if (sym->value->value.complex->re->_mpfr_d != NULL) {
#else
                if (sym->value->value.complex.r->_mpfr_d != NULL) {
#endif
                   e_lp1    = XMALLOC(struct list_node       *, sizeof(*e_lp1));
                   e_lp2    = XMALLOC(struct list_node       *, sizeof(*e_lp2));
                   l_item0  = XMALLOC(expv                    , sizeof(*l_item0));
                   l_item1  = XMALLOC(expv                    , sizeof(*l_item1));
                   l_item2  = XMALLOC(expv                    , sizeof(*l_item2));
   
                   l_item0->e_code = FLOAT_CONSTANT;
#ifdef _MPCLIB_
                   l_item0->v.e_llval = *(sym->value->value.complex->re->_mpfr_d)
                                      * sym->value->value.complex->re->_mpfr_sign ;
#else
                   l_item0->v.e_llval = *(sym->value->value.complex.r->_mpfr_d)
                                      * sym->value->value.complex.r->_mpfr_sign ;
#endif
                   l_item0->e_type = new_type_desc();
                   l_item0->e_type->basic_type = TYPE_DREAL;
                   l_item1->e_code = UNARY_MINUS_EXPR;
                   l_item2->e_code = FLOAT_CONSTANT;
#ifdef _MPCLIB_
                   l_item2->v.e_llval = *(sym->value->value.complex->im->_mpfr_d)
                                      * sym->value->value.complex->im->_mpfr_sign ;
#else
                   l_item2->v.e_llval = *(sym->value->value.complex.i->_mpfr_d)
                                      * sym->value->value.complex.i->_mpfr_sign ;
#endif
                   l_item2->e_type = new_type_desc();
                   l_item2->e_type->basic_type = TYPE_DREAL;
      
                   e_lp1->l_item = l_item1;
                   e_lp2->l_item = l_item2;
          
                   result->v.e_lp->l_item = l_item0;
                   result->v.e_lp->l_next = e_lp1;
                   result->v.e_lp->l_next->l_item = l_item1;
                   result->v.e_lp->l_next->l_item->v.e_lp = e_lp2;
                   result->v.e_lp->l_next->l_item->v.e_lp->l_item = l_item2;
                }
             } else if (sym->ts.type == BT_CHARACTER) {
                result->e_code = STRING_CONSTANT;
                result->e_type->size = *(sym->ts.u.cl->length->value.character.string);
                result->v.e_str = *(sym->value->value.character.string);
             } else if (sym->ts.type == BT_LOGICAL) {
                result->e_code = INT_CONSTANT;
                result->v.e_llval = sym->value->value.logical; 
             }
          } else {
             result->v.e_lp = XMALLOC(struct list_node *, sizeof(*result->v.e_lp));
             result->v.e_lp->l_item  = XMALLOC(expv     , sizeof(*result->v.e_lp->l_item));
             result->v.e_lp->l_item->e_code = LIST;
             if (sym->ts.type == BT_CHARACTER) {
                 result->e_type->size = *(sym->ts.u.cl->length->value.character.string);
             }

             cp = &sym->value->value.constructor;
             for (c = gfc_constructor_first (*cp); c; c = gfc_constructor_next (c)) {
                ep = &c->expr;
                e = *ep;

                e_lp0    = XMALLOC(struct list_node       *, sizeof(*e_lp0));
                e_lp0->l_item  = XMALLOC(expv              , sizeof(*e_lp0->l_item));

                if (sym->ts.type == BT_INTEGER) {
                   e_lp0->l_item->e_code = INT_CONSTANT;
                   e_lp0->l_item->v.e_llval = *(e->value.integer->_mp_d)
                                     * e->value.integer->_mp_size ;
                } else if (sym->ts.type == BT_REAL) {
                   e_lp0->l_item->e_code = FLOAT_CONSTANT;
                   e_lp0->l_item->v.e_lfval = *(e->value.real->_mpfr_d)
                                           * e->value.real->_mpfr_sign ;
                } else if (sym->ts.type == BT_COMPLEX) {

                   e_lp0->l_item->e_code = COMPLEX_CONSTANT;
                   e_lp0->l_item->v.e_lp = XMALLOC(struct list_node *, sizeof(*e_lp0->l_item->v.e_lp));
   
                   e_lp1    = XMALLOC(struct list_node       *, sizeof(*e_lp1));
                   e_lp2    = XMALLOC(struct list_node       *, sizeof(*e_lp2));
                   l_item0  = XMALLOC(expv                    , sizeof(*l_item0));
                   l_item1  = XMALLOC(expv                    , sizeof(*l_item1));
                   l_item2  = XMALLOC(expv                    , sizeof(*l_item2));
   
                   l_item0->e_code = FLOAT_CONSTANT;
#ifdef _MPCLIB_
                   l_item0->v.e_llval = *(sym->value->value.complex->re->_mpfr_d)
                                      * sym->value->value.complex->re->_mpfr_sign ;
#else
                   l_item0->v.e_llval = *(sym->value->value.complex.r->_mpfr_d)
                                      * sym->value->value.complex.r->_mpfr_sign ;
#endif
                   l_item0->e_type = new_type_desc();
                   l_item0->e_type->basic_type = TYPE_DREAL;
                   l_item1->e_code = UNARY_MINUS_EXPR;
                   l_item2->e_code = FLOAT_CONSTANT;
#ifdef _MPCLIB_
                   l_item2->v.e_llval = *(sym->value->value.complex->im->_mpfr_d)
                                      * sym->value->value.complex->im->_mpfr_sign ;
#else
                   l_item2->v.e_llval = *(sym->value->value.complex.i->_mpfr_d)
                                      * sym->value->value.complex.i->_mpfr_sign ;
#endif
                   l_item2->e_type = new_type_desc();
                   l_item2->e_type->basic_type = TYPE_DREAL;
      
                   e_lp1->l_item = l_item1;
                   e_lp2->l_item = l_item2;
          
                   e_lp0->l_item->v.e_lp->l_item = l_item0;
                   e_lp0->l_item->v.e_lp->l_next = e_lp1;
                   e_lp0->l_item->v.e_lp->l_next->l_item = l_item1;
                   e_lp0->l_item->v.e_lp->l_next->l_item->v.e_lp = e_lp2;
                   e_lp0->l_item->v.e_lp->l_next->l_item->v.e_lp->l_item = l_item2;
                   
                } else if (sym->ts.type == BT_CHARACTER) {
                   e_lp0->l_item->e_code = STRING_CONSTANT;
                   e_lp0->l_item->v.e_str = *(e->value.character.string);
                } else if (sym->ts.type == BT_LOGICAL) {
                   e_lp0->l_item->e_code = FLOAT_CONSTANT;
                   e_lp0->l_item->v.e_lfval = e->value.logical;
                }

                if (result->v.e_lp->l_item->v.e_lp == NULL) {
                   result->v.e_lp->l_item->v.e_lp = e_lp0;
                } else {
                   pre_node->l_next = e_lp0;
                }
                pre_node = e_lp0;
             }

          }
          id->info.proc_info.result = result;

       } else if (sym->attr.flavor == FL_VARIABLE) {

          id->class = CL_VAR;
          if (sym->attr.in_common) {
             id->stg = STG_COMMON;
          } else {
             id->stg = STG_SAVE;
          }

          id->is_declared  = 1;
          id->attr.exflags = 2;
          id->order        = 1;

          id->use_assoc = make_assoc_info( sym , 0 );
   
          id->type = make_type_w(sym, 0);

       } else if (sym->attr.flavor == FL_DERIVED) {

          id->class = CL_TAGNAME;
          id->stg = STG_TAGNAME;
          id->use_assoc = make_assoc_info( sym , 6 );
          id->type = make_type_w(sym, 6);

          id->type->members = NULL;
          component = sym->components;

          while (component != NULL) {
         
             member = XMALLOC(ID , sizeof(*member)) ;

             member->class = CL_ELEMENT;
             member->stg = STG_UNKNOWN;
             member->name = XMALLOC(SYMBOL, sizeof(struct symbol));
             member->name->s_name = component->name;

             member->type = make_type( component->ts, component->attr,component->as, 1);
             if (component->ts.type != BT_DERIVED) {
                member->type->ref->kind =  member->type->kind;
             } 
              
             if (id->type->members == NULL) {
                id->type->members = member;
             } else {
                pre_member->next = member;
             }
             pre_member = member;

             component = component->next;
          }
 
       } else if (sym->attr.flavor == FL_PROCEDURE) {

          id->use_assoc = make_assoc_info( sym , 0 );
          if ((sym->result != NULL && !sym->attr.generic) || 
              (sym->result != NULL && (sym->attr.generic && generic_result_on))) {
             id->type = make_type(sym->result->ts, sym->result->attr, sym->result->as, 2);
             sym->result->type = (void *) id->type;
          } else {
             id->type = new_type_desc();
             if (sym->attr.generic && (sym->attr.function || sym->gfc_new)) {
                id->type->basic_type = TYPE_GENERIC;
                id->type->attr.exflags = 0;
             } else {
                id->type->basic_type = TYPE_SUBR;
                id->type->attr.exflags = 16;
             }
          }
          id->type->attr.type_attr_flags = 0;

          id->class = CL_PROC;
          id->stg   = STG_EXT;
          id->is_declared  = 1;
          id->attr.exflags = 2;
          id->order        = 1;
          id->info.proc_info.pclass = P_DEFINEDPROC;
          id->info.proc_info.pclass = P_EXTERNAL;

          id->extID = make_extID(sym , 0);
          if (sym->attr.generic) {
             id->type->attr.exflags = 0;
          } else {
             id->type->attr.exflags = 16;
          }
          id->extID->name->s_name = id->name->s_name;
          id->extID->info.proc_info.type = id->type;

       } else {
          ;
       }

       return id;
}


static void trans_symbol(gfc_symtree *st )
{
    ID id;

    if(st == NULL) {
      return;
    }

    trans_symbol(st->left);

    if (!st->flag) {
    
       id = make_id ( st->n.sym , st->name );

       ID_LINK_ADD(id, id_list, last_ip);

       cur_relation = XMALLOC(relation *, sizeof(*cur_relation));
       cur_relation->sym  = st->n.sym;
       cur_relation->id   = id;
       cur_relation->next = NULL;
       if (top_relation == NULL ) {
          top_relation = cur_relation;
       } else {
          end_relation->next = cur_relation;
       }
       end_relation = cur_relation;
 
    }

    trans_symbol(st->right);
}


static void trans_symbol_generic(gfc_symtree *st )
{
    gfc_symbol      *sm;
    gfc_interface   *gene_if;

    ID id, def, new_id, generic_id;
    struct expression_node  *args1;
    struct external_symbol  *extid, *pre_extid, *gen_extid;
    struct list_node        *e_lp0;
    struct interface_info   *if_info;
    char   flag = 0;

    generic_result_on = 1;

    if(st == NULL) {
      return;
    }

    trans_symbol_generic(st->left);

    sm = st->n.sym;

    if (st->n.sym->attr.flavor == FL_MODULE) {
       ;
    } else if (st->n.sym->attr.flavor == FL_PARAMETER) {
       ;
    } else if (st->n.sym->attr.flavor == FL_VARIABLE) {
       ;
    } else if (st->n.sym->attr.flavor == FL_PROCEDURE) {

       if (st->n.sym->attr.generic && !st->flag ) {

          gene_if = st->n.sym->generic;
          while (gene_if != NULL) {
             if (gene_if->sym->attr.flavor == FL_DERIVED) {
                gene_if = gene_if->next;
                continue;
             }
             if(strcasecmp(module_list->module_name,gene_if->sym->module) == 0) {
                flag = 1;
                break;
             }
             gene_if = gene_if->next;
          }

          id = symbol2id( st->n.sym );

          if (id != NULL && flag) {

             if_info = XMALLOC(struct interface_info *, sizeof(struct interface_info));
             if_info->class      = INTF_DECL;
             if_info->operatorId = 0;
             if_info->idlist     = 0;

             id->extID->info.proc_info.interface_info = if_info;

             gene_if = st->n.sym->generic;
             id->extID->info.proc_info.intr_def_external_symbols = NULL;
             while (gene_if != NULL) {

                if (gene_if->sym->attr.flavor == FL_DERIVED) {
                   gene_if = gene_if->next;
                   continue;
                }

                top_gen_id = NULL;

                def = symbol2id( gene_if->sym );

                if (gene_if->sym == st->n.sym) {
                   new_id = make_id( gene_if->sym, gene_if->sym->name );
                   new_id->name = XMALLOC(SYMBOL, sizeof(struct symbol));
                   new_id->name->s_name = def->name->s_name;
                   ID_LINK_ADD(new_id, id_list, last_ip);
                   def = new_id;
                }

                if (def != NULL ) {
                   gen_extid = def->extID;

                   gen_extid->info.proc_info.is_module_specified = 1;
                   if(gene_if->sym->attr.external) {
                      ;
                   } else {
                      gen_extid->info.proc_info.ext_proc_class = EP_MODULE_PROCEDURE;
                   }

                   cur_gene_pro = XMALLOC(generic_procedure *, sizeof(*cur_gene_pro));
                   cur_gene_pro->modProcName =  gen_extid->name->s_name;
                   cur_gene_pro->belongProcName = id->name->s_name;
                   cur_gene_pro->eId = gen_extid;
                   if (top_gene_pro == NULL) {
                      top_gene_pro = cur_gene_pro;
                   } else {
                      end_gene_pro->next = cur_gene_pro;
                   }
                   end_gene_pro = cur_gene_pro;

                   if (id->extID->info.proc_info.intr_def_external_symbols == NULL) {
                      id->extID->info.proc_info.intr_def_external_symbols = gen_extid;
                   } else {
                      pre_extid->next = gen_extid;
                   }
                   pre_extid = gen_extid;

                   generic_id = XMALLOC(ID,sizeof(*id));
                   memcpy(generic_id, def, sizeof(*id));
                   generic_id->next = NULL;
                   generic_id->line = XMALLOC(struct line_info *, sizeof(*generic_id->line));
                   generic_id->line->ln_no = 0;
                   generic_id->line->file_id = 0;
                   generic_id->info.proc_info.pclass = P_DEFINEDPROC;

                   if (top_gen_id == NULL) {
                      top_gen_id = generic_id;
                   } else {
                      end_gen_id->next = generic_id; 
                   }
                   end_gen_id = generic_id;

                   args1 = def->extID->info.proc_info.args;

                   if (args1!=NULL) {

                      e_lp0 = args1->v.e_lp;

                      while (e_lp0!= NULL) {
                         if (e_lp0->l_item!=NULL){
                            generic_id = XMALLOC(ID,sizeof(*id));
                            generic_id->next = NULL;
                            generic_id->class = CL_VAR;
                            generic_id->stg = STG_ARG;
                            generic_id->type = e_lp0->l_item->v.e_lp->l_item->e_type;
                            generic_id->line = XMALLOC(struct line_info *, sizeof(*generic_id->line));
                            generic_id->line->ln_no = 0;
                            generic_id->line->file_id = 0;
                            generic_id->name = XMALLOC(SYMBOL , sizeof(struct symbol));
                            generic_id->name->s_name = e_lp0->l_item->v.e_lp->l_item->v.e_sym->s_name;
                            end_gen_id->next = generic_id; 
                            end_gen_id = generic_id;
                         }
                         e_lp0 = e_lp0->l_next;
                      }

                   }

                   gen_extid->info.proc_info.id_list = top_gen_id;
                }
                gene_if = gene_if->next;
             }
             id->extID->line = XMALLOC(struct line_info *, sizeof(*extid->line));
             id->extID->line->ln_no = 0;
             id->extID->line->file_id = 0;
          }

       }

    } else {
       ;
    }

    trans_symbol_generic(st->right);
}

static void trans_symbol_derived(gfc_symtree *st )
{
    gfc_symbol         *sym;
    gfc_formal_arglist *fa;
    gfc_component      *component;
    ID                      id, def, member;
    struct expression_node  *args;
    struct list_node        *list;

    if(st == NULL) {
      return;
    }

    trans_symbol_derived(st->left);

    sym = st->n.sym;

    if (sym->attr.flavor == FL_MODULE) {
       ;
    } else if (sym->attr.flavor == FL_PARAMETER) {
       ;
    } else if (sym->attr.flavor == FL_VARIABLE       || 
               sym->attr.flavor == FL_DERIVED        ||
               st->n.sym->attr.flavor == FL_PROCEDURE) {

       id = symbol2id( sym );
       if (sym->ts.type==BT_DERIVED && sym->ts.u.derived!=NULL) {

          def = symbol2id( sym->ts.u.derived );
          if (def != NULL) {
             id->type->ref = def->type;
          }
       }
       if (sym->formal != NULL) {
          fa = sym->formal;
          list = id->extID->info.proc_info.args->v.e_lp;

          while (fa != NULL && list != NULL) {
             if (fa->sym != NULL && fa->sym->ts.type==BT_DERIVED && fa->sym->ts.u.derived!=NULL) {
                def = symbol2id( fa->sym->ts.u.derived );
                if (def != NULL) {
                   list->l_item->v.e_lp->l_item->e_type->ref = def->type;
                }
             }
             fa = fa->next;
             list = list->l_next;
          }

       }
       if (sym->components != NULL) {
          component = sym->components;
          member    = id->type->members;

          while (component != NULL && member != NULL) {
             if (component->ts.type==BT_DERIVED && component->ts.u.derived!=NULL) {
                def = symbol2id( component->ts.u.derived );

                if (def != NULL) {
                   member->type->ref = def->type;
                }
             }
             component = component->next;
             member    = member->next;
          }
       }

       ;
    } else {
       ;
    }

    trans_symbol_derived(st->right);
}

static void check_symbol_generic(gfc_symtree *st )
{
    gfc_interface   *gene_if;

    if(st == NULL) {
      return;
    }

    check_symbol_generic(st->left);

    if (st->n.sym->attr.flavor == FL_PROCEDURE) {
       if (st->n.sym->attr.generic) {
          gene_if = st->n.sym->generic;
          while (gene_if != NULL) {
             if (gene_if->sym->attr.implicit_pure) {
                if(strcasecmp(st->n.sym->name,gene_if->sym->module)) {
                   st->n.sym->module = gene_if->sym->module;
                   break;
                }
             }
             gene_if = gene_if->next;
          }
       }
    }

    st->flag = 0;
    cur_symtree = XMALLOC(symtree *, sizeof(*cur_symtree));
    cur_symtree->st = st;
    cur_symtree->next = NULL;
    if (top_symtree == NULL) {
       top_symtree = cur_symtree;
    } else {
       end_symtree->next = cur_symtree;
    }
    end_symtree = cur_symtree;

    check_symbol_generic(st->right);
}

static void check_symbol_derived(gfc_symtree *st )
{
    if(st == NULL) {
      return;
    }

    check_symbol_derived(st->left);

    if (st->n.sym->attr.flavor == FL_PROCEDURE) {
       if (st->n.sym->attr.generic) {
          cur_symtree = top_symtree; 
          while (cur_symtree != NULL) {
             if (cur_symtree->st->n.sym->attr.flavor == FL_DERIVED) {
                if (cur_symtree->st != st) {
                   if (strcasecmp(st->name,cur_symtree->st->name)==0) { 
                      cur_symtree->st->name = st->name;
                      st->flag = 1;
                   }
                }
             }
             cur_symtree = cur_symtree->next;
          }
       }
    }

    check_symbol_derived(st->right);

}

static void trans_module()
{
    gfc_symtree *st;

    use_decls = XMALLOC(expv *, sizeof(*use_decls));
    use_decls->e_code = LIST;

    st = gfc_current_ns->sym_root;

    top_symtree = NULL;
    generic_result_on = 0;

    check_symbol_generic(st);
    check_symbol_derived(st);

    trans_symbol(st);


    trans_symbol_generic(st);
    trans_symbol_derived(st);
}


static void split_file_path(const char *file_pathname, char *path, char *file)
{
    const char *p = NULL;

    p = strrchr(file_pathname, '/');

    if (p == NULL) {
       *path = '\0';
       strcpy(file, file_pathname);
    } else if (p == file_pathname) {
       strcpy(path, "/");
       strcpy(file, p+1);
    } else {
       memcpy(path, file_pathname, (p-file_pathname));
       path[p-path] = '\0';
       strcpy(file, p+1);
    }
}

int main (int argc, char  **argv)
{
    char         *name = NULL;
    unsigned int g77_xargc;
    unsigned int newargsize;

    char *file_path_name;
    char *file_name;
    char *file_path;
    char *dotPost;

    if(argc==2)
     {
       file_path_name = argv[1];
       file_path = (char *)malloc(sizeof(char)*strlen(file_path_name));
       file_name = (char *)malloc(sizeof(char)*strlen(file_path_name));
       file_names[0] = (char *)malloc(sizeof(char)*(strlen(file_path_name)+4));
       strcpy (file_names[0], file_path_name);

       split_file_path(file_path_name, file_path, file_name);

       dotPost = strrchr(file_name, '.');
       if (dotPost != NULL && strcasecmp(dotPost, ".mod") != 0){
           printf("input file-name error (%s/%s)\n",file_path,file_name);
           exit(99);
       }

       name = strtok(file_name, ".");
       modincludeDirv = file_path;
     }
    else
     {
       printf("Error \n");
     }

    module_list = gfc_get_use_list();
    module_list->intrinsic = false;
    module_list->non_intrinsic = true;
    module_list->only_flag = false;
    module_list->rename   = NULL;

    module_list->module_name = (char *)gfc_get_string(name);

    newargsize = (g77_xargc << 2) + 20;       /* This should handle all.  */
    g77_new_decoded_options = XNEWVEC (struct cl_decoded_option, newargsize);

    gfc_init_options (newargsize, g77_new_decoded_options);

    gfc_init_kinds();
    gfc_symbol_init_2 ();

    import_module(module_list);

    initialize_compile();

    trans_module();


    SYMBOL sym = XMALLOC(SYMBOL, sizeof(struct symbol));
    bzero(sym, sizeof(*sym));
    sym->s_name = strdup(name);
    sym->s_next = NULL;
    sym->s_type = S_IDENT;

    export_module(sym, id_list, use_decls);

    free(name);
    free(module_list->module_name);

    exit (0);
}
