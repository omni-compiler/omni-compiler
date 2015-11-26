package exc.object;

import xcodeml.util.XmLog;

/**
 * Storage Class
 */
public enum StorageClass
{
    SNULL           (null),                         /* undefined */
    // C
    AUTO            ("auto"),            /* auto variable */
    PARAM           ("param"),           /* paramter */
    EXTERN          ("extern"),          /* extern variable */
    EXTDEF          ("extern_def"),      /* external defition */
    STATIC          ("static"),          /* static variable */
    REGISTER        ("register"),        /* register variable */
    LABEL           ("label"),           /* label */
    ULABEL          ("_label"),          /* undefined label, (temporary) */
    TAGNAME         ("tagname"),         /* tag name for struct/union/enum */
    MOE             ("moe"),             /* member of enum */
    TYPEDEF_NAME    ("typedef_name"),    /* typedef name */
    REG             ("_reg"),            /* register, temporary variable */
    MEMBER          ("_member"),         /* C++ class member */
    GCC_LABEL       ("gccLabel"),        /* gcc block scope label */
    
    // Fortran
    FLOCAL          ("flocal"),          /* local variable */
    FSAVE           ("fsave"),           /* module var or var which has save attribute */
    FCOMMON         ("fcommon"),         /* common variable */
    FPARAM          ("fparam"),          /* dummy argument */
    FFUNC           ("ffunc"),           /* func/sub/program name */
    FTYPE_NAME      ("ftype_name"),      /* type name */
    FCOMMON_NAME    ("fcommon_name"),    /* common name */
    FNAMELIST_NAME  ("fnamelist_name"),  /* namelist name */
    ;

    private String xcodeStr;

    private StorageClass(String xcodeStr)
    {
        this.xcodeStr = xcodeStr;
    }

    public String toXcodeString()
    {
        return xcodeStr;
    }
    
    public boolean canBeAddressed()
    {
        switch(this) {
        case PARAM:
        case AUTO:
        case EXTERN:
        case EXTDEF:
        case STATIC:
            return true;
        }
        
        return false;
    }
    
    public boolean isVarOrFunc()
    {
        switch(this) {
        case PARAM:
        case AUTO:
        case EXTERN:
        case EXTDEF:
        case STATIC:
        case REGISTER:
        case FLOCAL:
        case FSAVE:
        case FCOMMON:
        case FPARAM:
        case FFUNC:
            return true;
        }
        
        return false;
    }
    
    public boolean isBSS()
    {
        switch(this) {
        case EXTERN:
        case EXTDEF:
        case STATIC:
        case FCOMMON:
        case FSAVE:
            return true;
        }
        return false;
    }
    
    public boolean isFuncParam()
    {
        switch(this) {
        case PARAM:
        case FPARAM:
            return true;
        }
        return false;
    }
    
    public static StorageClass get(String s)
    {
        if(s == null)
            return null;
        
        for(StorageClass stg : values()) {
            if(s.equalsIgnoreCase(stg.xcodeStr)) {
                return stg;
            }
        }
        
        XmLog.fatal("unkown class '" + s + "'");
        return null;
    }
}
