package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import java.util.Iterator;

public class XMPutil {
  public static boolean hasCommXMPpragma(BlockList bl) {
    BlockIterator i = new bottomupBlockIterator(bl);
    for(i.init(); !i.end(); i.next()) {
      Block b = i.getBlock();
      if (b.Opcode() == Xcode.XMP_PRAGMA) {
        PragmaBlock pb = (PragmaBlock)b;
        String pragmaName = pb.getPragma();

        switch (XMPpragma.valueOf(pragmaName)) {
          // FIXME case REFLECT: needed???
          case BARRIER:
          case REDUCTION:
          case BCAST:
          // FIXME case GMOVE: needed???
            return true;
          default:
            break;
        }
      }
    }

    return false;
  }

  public static boolean isIntegerType(Xtype type) {
    if (type.getKind() == Xtype.BASIC) {
      BasicType basicType = (BasicType)type;
      switch (basicType.getBasicType()) {
        case BasicType.CHAR:
        case BasicType.UNSIGNED_CHAR:
        case BasicType.SHORT:
        case BasicType.UNSIGNED_SHORT:
        case BasicType.INT:
        case BasicType.UNSIGNED_INT:
        case BasicType.LONG:
        case BasicType.UNSIGNED_LONG:
        case BasicType.LONGLONG:
        case BasicType.UNSIGNED_LONGLONG:
          return true;
        default:
          return false;
      }
    }
    else return false;
  }

  public static String getTypeName(Xtype type) {
    if (type.getKind() == Xtype.BASIC) {
      BasicType basicType = (BasicType)type;
      switch (basicType.getBasicType()) {
        case BasicType.BOOL:			return new String("BOOL");
        case BasicType.CHAR:			return new String("CHAR");
        case BasicType.UNSIGNED_CHAR:		return new String("UNSIGNED_CHAR");
        case BasicType.SHORT:			return new String("SHORT");
        case BasicType.UNSIGNED_SHORT:		return new String("UNSIGNED_SHORT");
        case BasicType.INT:			return new String("INT");
        case BasicType.UNSIGNED_INT:		return new String("UNSIGNED_INT");
        case BasicType.LONG:			return new String("LONG");
        case BasicType.UNSIGNED_LONG:		return new String("UNSIGNED_LONG");
        case BasicType.LONGLONG:		return new String("LONGLONG");
        case BasicType.UNSIGNED_LONGLONG:	return new String("UNSIGNED_LONGLONG");
        case BasicType.FLOAT:			return new String("FLOAT");
        case BasicType.DOUBLE:			return new String("DOUBLE");
        case BasicType.LONG_DOUBLE:		return new String("LONG_DOUBLE");
        case BasicType.FLOAT_IMAGINARY:		return new String("FLOAT_IMAGINARY");
        case BasicType.DOUBLE_IMAGINARY:	return new String("DOUBLE_IMAGINARY");
        case BasicType.LONG_DOUBLE_IMAGINARY:	return new String("LONG_DOUBLE_IMAGINARY");
        case BasicType.FLOAT_COMPLEX:		return new String("FLOAT_COMPLEX");
        case BasicType.DOUBLE_COMPLEX:		return new String("DOUBLE_COMPLEX");
        case BasicType.LONG_DOUBLE_COMPLEX:	return new String("LONG_DOUBLE_COMPLEX");
        default:
          XMP.fatal("unsupported type");
      }
    }

    return null;
  }

  public static void mergeLists(XobjList dstList, XobjList srcList) {
    if (srcList == null) return;
    if (dstList == null) dstList = Xcons.List();

    for (XobjArgs i = srcList.getArgs(); i != null; i = i.nextArgs())
      dstList.add(i.getArg());
  }

  public static int countElmts(XobjList list) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      it.next();
      count++;
    }

    return count;
  }

  public static int countElmts(XobjList list, int constant) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.INT_CONSTANT) {
        if (x.getInt() == constant)
          count++;
      }
    }

    return count;
  }

  public static int countElmts(XobjList list, String string) {
    int count = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          count++;
      }
    }

    return count;
  }

  public static boolean hasElmt(XobjList list, int constant) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.INT_CONSTANT) {
        if (x.getInt() == constant)
          return true;
      }
    }

    return false;
  }

  public static boolean hasElmt(XobjList list, String string) {
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x == null) continue;

      if (x.Opcode() == Xcode.STRING) {
        if (x.getString().equals(string))
          return true;
      }
    }

    return false;
  }

  public static int getFirstIndex(XobjList list, int constant) throws XMPexception {
    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.INT_CONSTANT) {
          if (x.getInt() == constant)
            return index;
        }
      }

      index++;
    }

    throw new XMPexception("exception in exc.xcalablemp.XMPutil.getFirstIndex(), element does not exist");
  }

  public static int getFirstIndex(XobjList list, String string) throws XMPexception {
    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.STRING) {
          if (x.getString().equals(string))
            return index;
        }
      }

      index++;
    }

    throw new XMPexception("exception in exc.xcalablemp.XMPutil.getFirstIndex(), element does not exist");
  }

  public static int getLastIndex(XobjList list, int constant) throws XMPexception {
    int elmtIndex = 0;
    boolean hasFound = false;

    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.INT_CONSTANT) {
          if (x.getInt() == constant) {
            hasFound = true;
            elmtIndex = index;
          }
        }
      }

      index++;
    }

    if (hasFound) return elmtIndex;
    else
      throw new XMPexception("exception in exc.xcalablemp.XMPutil.getLastIndex(), element does not exist");
  }

  public static int getLastIndex(XobjList list, String string) throws XMPexception {
    int elmtIndex = 0;
    boolean hasFound = false;

    int index = 0;
    Iterator<Xobject> it = list.iterator();
    while (it.hasNext()) {
      Xobject x = it.next();
      if (x != null) {
        if (x.Opcode() == Xcode.STRING) {
          if (x.getString().equals(string)) {
            hasFound = true;
            elmtIndex = index;
          }
        }
      }

      index++;
    }

    if (hasFound) return elmtIndex;
    else
      throw new XMPexception("exception in exc.xcalablemp.XMPutil.getLastIndex(), element does not exist");
  }
}
