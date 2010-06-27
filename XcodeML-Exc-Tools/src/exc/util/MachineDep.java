/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.util;

import xcodeml.util.XmLog;
import exc.block.Block;
import exc.object.*;

public class MachineDep
{
    private static Xtype uintPtrType;
    
    public static Xtype getUintPtrType()
    {
        if(uintPtrType != null)
            return uintPtrType;
        
        if(MachineDepConst.SIZEOF_VOID_P == MachineDepConst.SIZEOF_UNSIGNED_INT)
        	uintPtrType = Xtype.unsignedType;
        else if(MachineDepConst.SIZEOF_VOID_P == MachineDepConst.SIZEOF_UNSIGNED_LONG)
        	uintPtrType =Xtype.unsignedlongType;
        else if(MachineDepConst.SIZEOF_VOID_P == MachineDepConst.SIZEOF_UNSIGNED_LONG_LONG)
        	uintPtrType =Xtype.unsignedlonglongType;
        else
        	XmLog.fatal("no unsigned integer type applicable to void pointer type about size.");
        return uintPtrType;
    }
    
    public static int getTypeSize(Xtype t, Block b)
    {
        int bt;
        
        switch(t.getKind()) {
        case Xtype.POINTER:
            return MachineDepConst.SIZEOF_VOID_P;
        case Xtype.BASIC:
            bt = t.getBasicType();
            if(t.getFkind() != null) {
                Xobject kv = Xcons.Reduce(t.getFkind(), b);
                if(kv.isIntConstant()) {
                    int k = kv.getInt();
                    switch(k) {
                    case BasicType.CHAR: case BasicType.UNSIGNED_CHAR:
                    case BasicType.SHORT: case BasicType.UNSIGNED_SHORT:
                    case BasicType.INT: case BasicType.UNSIGNED_INT:
                    case BasicType.LONG: case BasicType.UNSIGNED_LONG:
                    case BasicType.LONGLONG: case BasicType.UNSIGNED_LONGLONG:
                    case BasicType.FLOAT: case BasicType.DOUBLE:
                    case BasicType.LONG_DOUBLE: case BasicType.F_NUMERIC:
                        return k;
                    case BasicType.FLOAT_COMPLEX:
                    case BasicType.DOUBLE_COMPLEX:
                    case BasicType.LONG_DOUBLE_COMPLEX:
                        return k * 2;
                    }
                    break;
                }
            }
            
            switch(bt) {
            case BasicType.CHAR:
            case BasicType.UNSIGNED_CHAR:
                return MachineDepConst.SIZEOF_UNSIGNED_CHAR;
            case BasicType.SHORT:
            case BasicType.UNSIGNED_SHORT:
                return MachineDepConst.SIZEOF_UNSIGNED_SHORT;
            case BasicType.INT:
            case BasicType.UNSIGNED_INT:
                return MachineDepConst.SIZEOF_UNSIGNED_INT;
            case BasicType.LONG:
            case BasicType.UNSIGNED_LONG:
                return MachineDepConst.SIZEOF_UNSIGNED_LONG;
            case BasicType.LONGLONG:
            case BasicType.UNSIGNED_LONGLONG:
                return MachineDepConst.SIZEOF_UNSIGNED_LONG_LONG;
            case BasicType.FLOAT:
            case BasicType.FLOAT_IMAGINARY:
                return MachineDepConst.SIZEOF_FLOAT;
            case BasicType.DOUBLE:
            case BasicType.DOUBLE_IMAGINARY:
                return MachineDepConst.SIZEOF_DOUBLE;
            case BasicType.LONG_DOUBLE:
            case BasicType.LONG_DOUBLE_IMAGINARY:
                return MachineDepConst.SIZEOF_LONG_DOUBLE;
            case BasicType.FLOAT_COMPLEX:
                return MachineDepConst.SIZEOF_FLOAT * 2;
            case BasicType.DOUBLE_COMPLEX:
                return MachineDepConst.SIZEOF_DOUBLE * 2;
            case BasicType.LONG_DOUBLE_COMPLEX:
                return MachineDepConst.SIZEOF_LONG_DOUBLE * 2;
            }
            break;
        }
        
        throw new IllegalArgumentException("cannot get size of " + t);
    }
    
    public static boolean sizeEquals(Xtype t1, Xtype t2, Block b)
    {
        return getTypeSize(t1, b) == getTypeSize(t2, b);
    }
}
