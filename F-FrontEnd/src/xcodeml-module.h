/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file xcodeml-module.h
 */

#ifndef _XCODEML_MODULE_H_
#define _XCODEML_MODULE_H_

extern int use_module(const char * module_filename, const char * fortran_filename);
extern int use_module_to(const char * module_filename, FILE * fd);

typedef struct symbol_filterElm_t {
    char * use;
    char * local;
    struct symbol_filterElm_t * next;
} symbol_filterElem;

enum filter_usage {
    RENAME,
    LIMIT,
    ACCEPTANY,
};

typedef struct symbol_filter_t {
    enum filter_usage usage;
    struct symbol_filterElm_t * list;
    struct symbol_filter_t * next;
    int nonUseSymbolNumber;
} symbol_filter;

#define FILTER_USAGE(f) ((f)->usage)

extern symbol_filter * push_new_filter(void);
extern symbol_filter * peek_filter(void);
extern void pop_filter(void);
extern void symbol_filter_addElem(symbol_filter * filter, char * use, char * local);
extern char * apply_filter(symbol_filter * filter, char * symbol);
extern int is_use_symbol(char *symbol);
extern char * convert_to_non_use_symbol(char *orgname);

#endif /* _XCODEML_MODULE_H_ */
