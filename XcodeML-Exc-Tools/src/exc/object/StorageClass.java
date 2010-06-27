/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import xcodeml.c.binding.gen.XbcId;
import xcodeml.f.binding.gen.XbfId;
import xcodeml.util.XmLog;

/**
 * Storage Class
 */
public enum StorageClass
{
    SNULL           (null),                         /* undefined */
    // C
    AUTO            (XbcId.SCLASS_AUTO),            /* auto variable */
    PARAM           (XbcId.SCLASS_PARAM),           /* paramter */
    EXTERN          (XbcId.SCLASS_EXTERN),          /* extern variable */
    EXTDEF          (XbcId.SCLASS_EXTERN_DEF),      /* external defition */
    STATIC          (XbcId.SCLASS_STATIC),          /* static variable */
    REGISTER        (XbcId.SCLASS_REGISTER),        /* register variable */
    LABEL           (XbcId.SCLASS_LABEL),           /* label */
    ULABEL          ("_label"),                     /* undefined label, (temporary) */
    TAGNAME         (XbcId.SCLASS_TAGNAME),         /* tag name for struct/union/enum */
    MOE             (XbcId.SCLASS_MOE),             /* member of enum */
    TYPEDEF_NAME    (XbcId.SCLASS_TYPEDEF_NAME),    /* typedef name */
    REG             ("_reg"),                       /* register, temporary variable */
    MEMBER          ("_member"),                    /* C++ class member */
    GCC_LABEL       (XbcId.SCLASS_GCCLABEL),        /* gcc block scope label */
    
    // Fortran
    FLOCAL          (XbfId.SCLASS_FLOCAL),          /* local variable */
    FSAVE           (XbfId.SCLASS_FSAVE),           /* module var or var which has save attribute */
    FCOMMON         (XbfId.SCLASS_FCOMMON),         /* common variable */
    FPARAM          (XbfId.SCLASS_FPARAM),          /* dummy argument */
    FFUNC           (XbfId.SCLASS_FFUNC),           /* func/sub/program name */
    FTYPE_NAME      (XbfId.SCLASS_FTYPE_NAME),      /* type name */
    FCOMMON_NAME    (XbfId.SCLASS_FCOMMON_NAME),    /* common name */
    FNAMELIST_NAME  (XbfId.SCLASS_FNAMELIST_NAME),  /* namelist name */
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
