/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.xcodeml;

import exc.object.StorageClass;
import xcodeml.XmObj;
import xcodeml.c.binding.gen.XbcId;
import xcodeml.f.binding.gen.XbfId;

/**
 * Symbol which represents identifiers in C and Fortran (XbcId and XbfId).
 */
public class XmSymbol
{
    public enum SymbolType {
        OTHER,
        IDENT,
    }
    
    /** id tag */
    private XmObj _xmobj;
    /** name */
    private String _name;
    /** sclass */
    private StorageClass _sclass;
    /** type id */
    private String _typeId;
    /**  symbol type */
    private SymbolType _symbolType;
    
    final StorageClass[] C_IDENT_SCLASSES = {
        StorageClass.AUTO,
        StorageClass.EXTERN,
        StorageClass.EXTDEF,
        StorageClass.PARAM,
        StorageClass.REGISTER,
        StorageClass.STATIC,
    };
    
    final StorageClass[] F_IDENT_SCLASSES = {
        StorageClass.FLOCAL,
        StorageClass.FPARAM,
        StorageClass.FCOMMON,
        StorageClass.FSAVE,
    };
    
    public XmSymbol(XbcId xid)
    {
        _xmobj = xid;
        _name = xid.getName().getContent();
        _sclass = StorageClass.get(xid.getSclass());
        _typeId = xid.getType();
        if(_typeId == null) {
            _typeId = xid.getName().getType();
        }
        setSymbolType(C_IDENT_SCLASSES);
    }
    
    public XmSymbol(XbfId xid)
    {
        _xmobj = xid;
        _name = xid.getName().getContent();
        _sclass = StorageClass.get(xid.getSclass());
        _typeId = xid.getType();
        if(_typeId == null) {
            _typeId = xid.getName().getType();
        }
        setSymbolType(F_IDENT_SCLASSES);
    }
    
    private void setSymbolType(StorageClass[] identSclasses)
    {
        for(StorageClass identSclass : identSclasses) {
            if(identSclass.equals(_sclass)) {
                _symbolType = SymbolType.IDENT;
                return;
            }
        }
        
        _symbolType = SymbolType.OTHER;
    }

    /** get symbol name */
    public String getName()
    {
        return _name;
    }

    /** get storage class */
    public StorageClass getSclass()
    {
        return _sclass;
    }

    /** get type id */
    public String getTypeId()
    {
        return _typeId;
    }
    
    /** get symbol type */
    public SymbolType getSymbolType()
    {
        return _symbolType;
    }
    
    /** return if is identifier */
    public boolean isIdent()
    {
        return _symbolType == SymbolType.IDENT;
    }

    /** get XcodeML object */
    public XmObj getXmObj()
    {
        return _xmobj;
    }
    
    @Override
    public String toString()
    {
        return "[" + _name + ", slcass=" + _sclass + "]";
    }
}
