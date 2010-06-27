/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.openmp;

import xcodeml.util.XmLog;
import xcodeml.util.XmOption;
import exc.object.*;

//
// OMP symbol table, information for each OMP variable
//
public class OMPvar
{
    OMPvar next; // link
    Ident id; // key
    Ident thdprv_local_id; // threadprivate local id
    OMPpragma atr; // original attribute: PRIVATE, SHARED, COPYIN
    boolean is_shared;
    boolean is_private;
    boolean is_first_private;
    boolean is_last_private;
    boolean is_copy_private;
    boolean is_reduction;
    boolean is_copyin;
    OMPpragma reduction_op;
    private Xobject private_addr; // original base
    private Xobject copy_private_addr;
    private Xobject shared_addr;
    private Xobject shared_array;
    private Xobject size_of_array;
    
    private static final String C_FLOAT_MAX_VALUE   = "3.40282347E38";
    private static final String C_DOUBLE_MAX_VALUE  = "1.797693134862315E308";
    private static final String F_REAL4_MAX_VALUE   = "3.40282347E38";
    private static final String F_REAL8_MAX_VALUE   = "1.797693134862315D308";
    private static final String F_REAL16_MAX_VALUE  = "1.1897314953572317650857593266280070E4931_16";

    public OMPvar(Ident id, OMPpragma atr)
    {
        this.id = id;
        this.atr = atr;

        if(OMP.debugFlag)
            System.out.println("OMPvar(" + id.getName() + "," + atr.getName() + ")");
        switch(atr) {
        case DATA_SHARED: /* shared <namelist> */
            is_shared = true;
            break;
        case DATA_PRIVATE: /* private <namelist> */
            is_private = true;
            break;
        case DATA_COPYIN:
            is_copyin = true;
            is_shared = true;
            break;
        case DATA_FIRSTPRIVATE:
            is_private = true;
            is_shared = true;
            is_first_private = true;
            break;
        case DATA_LASTPRIVATE:
            is_private = true;
            is_shared = true;
            is_last_private = true;
            break;
        case DATA_COPYPRIVATE:
            is_private = true;
            is_copy_private = true;
            break;
        case _DATA_PRIVATE_SHARED: // for private function parameter
            is_private = true;
            is_shared = true;
            break;

        default:
            if(atr.isDataReduction()) {
                is_private = true;
                is_shared = true;
                is_reduction = true;
                reduction_op = atr;
            }
            break;
        }
    }

    Xobject getAddr()
    {
        if(private_addr != null)
            return private_addr;
        return shared_addr;
    }
    
    Xobject getPrivateAddr()
    {
        return private_addr;
    }

    void setPrivateAddr(Xobject x)
    {
        private_addr = x;
    }
    
    Xobject getCopyPrivateAddr()
    {
        return copy_private_addr;
    }
    
    void setCopyPrivateAddr(Xobject x)
    {
        copy_private_addr = x;
    }
    
    Xobject getSharedAddr()
    {
        return shared_addr;
    }
    
    void setSharedAddr(Xobject x)
    {
        shared_addr = x;
    }

    Xobject getSharedArray()
    {
        return shared_array;
    }

    void setSharedArray(Xobject x)
    {
        shared_array = x;
    }
    
    Xobject Ref()
    {
        return Xcons.PointerRef(getAddr());
    }

    Xobject SharedRef()
    {
        return Xcons.PointerRef(getSharedAddr());
    }

    Xobject getSize()
    {
        if(size_of_array == null)
            return Xcons.SizeOf(id.Type());
        return size_of_array;
    }
    
    void setSizeOfArray(Xobject e)
    {
        size_of_array = e;
    }
    
    public boolean isVariableArray()
    {
        return size_of_array != null;
    }
    
    public Ident getThdPrvLocalId()
    {
        return thdprv_local_id;
    }
    
    public void setThdPrvLocalId(Ident id)
    {
        thdprv_local_id = id;
    }
    
    private String getFloatMaxValue(Xtype t)
    {
        if(XmOption.isLanguageC()) {
            if(t.getBasicType() == BasicType.DOUBLE)
                return C_DOUBLE_MAX_VALUE;
            else
                return C_FLOAT_MAX_VALUE;
        } else {
            Xobject ko = Xcons.Reduce(t.getFkind());
            int k = 4;
            if(ko == null) {
                if(t.getBasicType() == BasicType.DOUBLE)
                    k = 8;
            } else if(ko instanceof XobjInt) {
                k = ko.getInt();
            } else {
                XmLog.fatal("cannot reduce REAL kind : " + ko);
            }
            
            switch(k) {
            case 4:
                return F_REAL4_MAX_VALUE;
            case 8:
                return F_REAL8_MAX_VALUE;
            case 16:
                return F_REAL16_MAX_VALUE;
            default:
                XmLog.fatal("unsupported REAL kind : " + k);
                return null;
            }
        }
    }
    
    Xobject reductionInitValue()
    {
        int bt;
        Xtype t = id.Type();
        
        switch(reduction_op) {
        case DATA_REDUCTION_PLUS:
        case DATA_REDUCTION_MINUS:
            if(t.isFloating())
                return Xcons.Float(Xcode.FLOAT_CONSTANT, t, 0.0D);
        case DATA_REDUCTION_BITOR:
        case DATA_REDUCTION_BITXOR:
        case DATA_REDUCTION_LOGOR:
        case DATA_REDUCTION_IOR:
        case DATA_REDUCTION_IEOR:
            return Xcons.IntConstant(0);

        case DATA_REDUCTION_MUL:
            if(t.isFloating())
                return Xcons.Float(Xcode.FLOAT_CONSTANT, t, 1.0D);
        case DATA_REDUCTION_LOGAND:
            return Xcons.IntConstant(1);

        case DATA_REDUCTION_BITAND:
            return Xcons.unaryOp(Xcode.BIT_NOT_EXPR, Xcons.IntConstant(0));

        case DATA_REDUCTION_IAND:
            return Xcons.IntConstant(-1);
            
        case DATA_REDUCTION_MIN:
            if(id.Type().getKind() == Xtype.BASIC)
                bt = t.getBasicType();
            else if(t.isEnum())
                bt = BasicType.INT;
            else
                bt = BasicType.UNDEF;
            switch(bt) {
            case BasicType.CHAR:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0x7F);
            case BasicType.UNSIGNED_CHAR:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0xFF);
            case BasicType.SHORT:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0x7FFF);
            case BasicType.UNSIGNED_SHORT:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0xFFFF);
            case BasicType.INT:
            case BasicType.LONG:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0x7FFFFFFF);
            case BasicType.UNSIGNED_INT:
            case BasicType.UNSIGNED_LONG:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0xFFFFFFFF);
            case BasicType.FLOAT:
            case BasicType.DOUBLE:
                return Xcons.Float(Xcode.FLOAT_CONSTANT, t, getFloatMaxValue(t));
            }
            break;

        case DATA_REDUCTION_MAX:
            if(t.getKind() == Xtype.BASIC)
                bt = t.getBasicType();
            else if(t.isEnum())
                bt = BasicType.INT;
            else
                bt = BasicType.UNDEF;
            switch(bt) {
            case BasicType.CHAR:
                return Xcons.Int(Xcode.INT_CONSTANT, t, -128);
            case BasicType.SHORT:
                return Xcons.Int(Xcode.INT_CONSTANT, t, -32768);
            case BasicType.INT:
            case BasicType.LONG:
                // return Xcons.Int(Xcode.INT_CONSTANT,t,-2147483648);
                // to avoid gcc warning
                return Xcons.Int(Xcode.INT_CONSTANT, Xtype.unsignedType, 0x80000000);
            case BasicType.UNSIGNED_CHAR:
            case BasicType.UNSIGNED_SHORT:
            case BasicType.UNSIGNED_INT:
            case BasicType.UNSIGNED_LONG:
                return Xcons.Int(Xcode.INT_CONSTANT, t, 0);

            case BasicType.FLOAT:
            case BasicType.DOUBLE:
                return Xcons.Float(Xcode.FLOAT_CONSTANT, t, "-" + getFloatMaxValue(t));
            }
            
        case DATA_REDUCTION_LOGEQV:
            return Xcons.FlogicalConstant(true);
            
        case DATA_REDUCTION_LOGNEQV:
            return Xcons.FlogicalConstant(false);
        }
        OMP.fatal("OMPvar.reductionInitValue");
        return null;
    }

    @Override
    public String toString()
    {
        return "(" + id + " : " +
            (private_addr != null ? "private_addr=" + private_addr : "shared_addr=" + shared_addr) + ")";
    }
}
