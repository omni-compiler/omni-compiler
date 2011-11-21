/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package xcodeml.f.decompile;

/**
 * Storage classes in decompiler.
 */
enum XfStorageClass {
    FLOCAL          ("flocal"),          /* local variable */
    FSAVE           ("fsave"),           /* module var or var which has save attribute */
    FCOMMON         ("fcommon"),         /* common variable */
    FPARAM          ("fparam"),          /* dummy argument */
    FFUNC           ("ffunc"),           /* func/sub/program name */
    FTYPE_NAME      ("ftype_name"),      /* type name */
    FCOMMON_NAME    ("fcommon_name"),    /* common name */
    FNAMELIST_NAME  ("fnamelist_name"),  /* namelist name */
    ;

    public String toXcodeString() {
        return xcodeStr;
    }

    private String xcodeStr;

    private XfStorageClass(String xcodeStr) {
        this.xcodeStr = xcodeStr;
    }
}
