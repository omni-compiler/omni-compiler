/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPshadow {
  // defined in xmp_constant.h
  public final static int SHADOW_NONE	= 400;
  public final static int SHADOW_NORMAL	= 401;
  public final static int SHADOW_FULL	= 402;

  private int _type;
  private Xobject _lo;
  private Xobject _hi;

  public XMPshadow(int type, Xobject lo, Xobject hi) {
    _type = type;
    _lo = lo;
    _hi = hi;
  }

  public int getType() {
    return _type;
  }

  public Xobject getLo() {
    return _lo;
  }

  public Xobject getHi() {
    return _hi;
  }

  // FIXME incomplete, not checked
  public static void translateShadow(XobjList shadowDecl, XMPglobalDecl globalDecl,
                                     boolean isLocalPragma, PragmaBlock pb) throws XMPexception {
    // start translation
    XMPsymbolTable localXMPsymbolTable = null;
    if (isLocalPragma) {
      localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    }

    // find aligned array
    String arrayName = shadowDecl.getArg(0).getString();
    XMPalignedArray alignedArray = null;
    if (isLocalPragma) {
      alignedArray = localXMPsymbolTable.getXMPalignedArray(arrayName);
    }
    else {
      alignedArray = globalDecl.getXMPalignedArray(arrayName);
    }

    if (alignedArray == null) {
      throw new XMPexception("the aligned array '" + arrayName + "' is not found in local scope");
    }

    if (alignedArray.hasShadow()) {
      throw new XMPexception("the aligned array '" + arrayName + "' has the shadow declaration already");
    }

    // init shadow
    XobjList shadowFuncArgs = Xcons.List(alignedArray.getDescId().Ref());
    int arrayIndex = 0;
    int arrayDim = alignedArray.getDim();
    for (XobjArgs i = shadowDecl.getArg(1).getArgs(); i != null; i = i.nextArgs()) {
      if (arrayIndex == arrayDim) {
        throw new XMPexception("wrong shadow dimension indicated, too many");
      }

      XobjList shadowObj = (XobjList)i.getArg();
      XobjInt shadowType = (XobjInt)shadowObj.getArg(0);
      XobjList shadowBody = (XobjList)shadowObj.getArg(1);
      switch (shadowType.getInt()) {
        case XMPshadow.SHADOW_NONE:
          {
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NONE)));
            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NONE, null, null), arrayIndex);
            break;
          }
        case XMPshadow.SHADOW_NORMAL:
          {
            Xobject shadowLo = shadowBody.left();
            Xobject shadowHi = shadowBody.right();

            if (XMPutil.isZeroIntegerObj(shadowLo) && XMPutil.isZeroIntegerObj(shadowHi)) {
              shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NONE)));
              alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NONE, null, null), arrayIndex);
              break;
            }

            if (alignedArray.getAlignMannerAt(arrayIndex) == XMPalignedArray.NOT_ALIGNED) {
              throw new XMPexception("indicated dimension is not aligned");
            }
            else if (alignedArray.getAlignMannerAt(arrayIndex) == XMPalignedArray.DUPLICATION) {
              throw new XMPexception("indicated dimension is not distributed");
            }

            if (alignedArray.getAlignMannerAt(arrayIndex) != XMPalignedArray.BLOCK) {
              throw new XMPexception("shadow should be declared on block distbirution");
            }

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_NORMAL)));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowLo));
            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, shadowHi));
            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_NORMAL, shadowLo, shadowHi), arrayIndex);
            break;
          }
        case XMPshadow.SHADOW_FULL:
          {
            if (alignedArray.getAlignMannerAt(arrayIndex) == XMPalignedArray.NOT_ALIGNED) {
              throw new XMPexception("indicated dimension is not aligned");
            }
            else if (alignedArray.getAlignMannerAt(arrayIndex) == XMPalignedArray.DUPLICATION) {
              throw new XMPexception("indicated dimension is not distributed");
            }

            shadowFuncArgs.add(Xcons.Cast(Xtype.intType, Xcons.IntConstant(XMPshadow.SHADOW_FULL)));
            alignedArray.setShadowAt(new XMPshadow(XMPshadow.SHADOW_FULL, null, null), arrayIndex);
            break;
          }
        default:
          throw new XMPexception("unknown shadow type");
      }

      arrayIndex++;
    }

    if (arrayIndex != arrayDim) {
      throw new XMPexception("the number of <nodes/template-subscript> should be the same with the dimension");
    }

    if (isLocalPragma) {
      XMPlocalDecl.addConstructorCall("_XMP_init_shadow", shadowFuncArgs, globalDecl, pb);
    }
    else {
      globalDecl.addGlobalInitFuncCall("_XMP_init_shadow", shadowFuncArgs);
    }

    // set shadow flag
    alignedArray.setHasShadow();
  }
}
