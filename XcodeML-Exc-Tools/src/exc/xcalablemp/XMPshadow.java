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
      throw new XMPexception("the aligned array '" + arrayName + "' is not found");
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
	    
            if (shadowLo.isZeroConstant() && shadowHi.isZeroConstant()) {
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

  public static Block translateReflect(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    // start translation
    XobjList reflectDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    BlockList reflectFuncBody = Bcons.emptyBody();

    XobjList arrayList = (XobjList)reflectDecl.getArg(0);
    for (XobjArgs iter = arrayList.getArgs(); iter != null; iter = iter.nextArgs()) {
      String arrayName = iter.getArg().getString();
      XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(arrayName, localXMPsymbolTable);
      if (alignedArray == null) {
        throw new XMPexception("the aligned array '" + arrayName + "' is not found");
      }

      if (!alignedArray.hasShadow()) {
        throw new XMPexception("the aligned array '" + arrayName + "' has no shadow declaration");
      }

      int arrayDim = alignedArray.getDim();
      for (int i = 0; i < arrayDim; i++) {
        XMPshadow shadowObj = alignedArray.getShadowAt(i);
        switch (shadowObj.getType()) {
          case XMPshadow.SHADOW_NONE:
            break;
          case XMPshadow.SHADOW_NORMAL:
            createReflectNormalShadowFunc(pb, globalDecl, alignedArray, i, reflectFuncBody);
            break;
          case XMPshadow.SHADOW_FULL:
            createReflectFullShadowFunc(pb, globalDecl, alignedArray, i, reflectFuncBody);
            break;
          default:
            throw new XMPexception("unknown shadow type");
        }
      }
    }

    Block reflectFuncCallBlock = Bcons.COMPOUND(reflectFuncBody);
    pb.replace(reflectFuncCallBlock);

    return reflectFuncCallBlock;
  }

  private static void createReflectNormalShadowFunc(PragmaBlock pb, XMPglobalDecl globalDecl,
                                                    XMPalignedArray alignedArray, int arrayIndex,
                                                    BlockList reflectFuncBody) {
    String arrayName = alignedArray.getName();

    // decl buffers
    Ident loSendId = reflectFuncBody.declLocalIdent("_XMP_reflect_LO_SEND_" + arrayName, Xtype.voidPtrType);
    Ident loRecvId = reflectFuncBody.declLocalIdent("_XMP_reflect_LO_RECV_" + arrayName, Xtype.voidPtrType);
    Ident hiSendId = reflectFuncBody.declLocalIdent("_XMP_reflect_HI_SEND_" + arrayName, Xtype.voidPtrType);
    Ident hiRecvId = reflectFuncBody.declLocalIdent("_XMP_reflect_HI_RECV_" + arrayName, Xtype.voidPtrType);

    // pack shadow
    Ident packFuncId = globalDecl.declExternFunc("_XMP_pack_shadow_NORMAL");
    XobjList packFuncArgs = Xcons.List(loSendId.getAddr(), hiSendId.getAddr(), alignedArray.getAddrIdVoidRef(),
                                       alignedArray.getDescId().Ref(), Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(packFuncId.Call(packFuncArgs)));

    // exchange shadow
    Ident exchangeFuncId = globalDecl.declExternFunc("_XMP_exchange_shadow_NORMAL");
    XobjList exchangeFuncArgs = Xcons.List(loRecvId.getAddr(), hiRecvId.getAddr(), loSendId.Ref(), hiSendId.Ref());
    exchangeFuncArgs.add(alignedArray.getDescId().Ref());
    exchangeFuncArgs.add(Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(exchangeFuncId.Call(exchangeFuncArgs)));

    // unpack shadow
    Ident unpackFuncId = globalDecl.declExternFunc("_XMP_unpack_shadow_NORMAL");;
    XobjList unpackFuncArgs = Xcons.List(loRecvId.Ref(), hiRecvId.Ref(), alignedArray.getAddrIdVoidRef(),
                                         alignedArray.getDescId().Ref(), Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(unpackFuncId.Call(unpackFuncArgs)));
  }

  private static void createReflectFullShadowFunc(PragmaBlock pb, XMPglobalDecl globalDecl,
                                                  XMPalignedArray alignedArray, int arrayIndex,
                                                  BlockList reflectFuncBody) {
    Ident funcId = globalDecl.declExternFunc("_XMP_reflect_shadow_FULL");
    XobjList funcArgs = Xcons.List(alignedArray.getAddrIdVoidRef(), alignedArray.getDescId().Ref(), Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(funcId.Call(funcArgs)));
  }

  // FIXME implement full shadow
  public static Block translateGpuReflect(PragmaBlock pb, XMPglobalDecl globalDecl) throws XMPexception {
    // start translation
    XobjList reflectDecl = (XobjList)pb.getClauses();
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(pb);
    BlockList reflectFuncBody = Bcons.emptyBody();

    XobjList arrayList = (XobjList)reflectDecl.getArg(0);
    for (XobjArgs iter = arrayList.getArgs(); iter != null; iter = iter.nextArgs()) {
      String arrayName = iter.getArg().getString();

      XMPgpuData gpuData = XMPgpuDataTable.findXMPgpuData(arrayName, pb);
      if (gpuData == null) {
        throw new XMPexception("'" + arrayName + "' is not allocated on the accelerator");
      }

      XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
      if (alignedArray == null) {
        throw new XMPexception("the aligned array '" + arrayName + "' is not found");
      }

      if (!alignedArray.hasShadow()) {
        throw new XMPexception("the aligned array '" + arrayName + "' has no shadow declaration");
      }

      int arrayDim = alignedArray.getDim();
      for (int i = 0; i < arrayDim; i++) {
        XMPshadow shadowObj = alignedArray.getShadowAt(i);
        switch (shadowObj.getType()) {
          case XMPshadow.SHADOW_NONE:
            break;
          case XMPshadow.SHADOW_NORMAL:
            createGpuReflectNormalShadowFunc(pb, globalDecl, gpuData, i, reflectFuncBody);
            break;
          case XMPshadow.SHADOW_FULL:
            throw new XMPexception("not implemented yet");
          default:
            throw new XMPexception("unknown shadow type");
        }
      }
    }

    Block reflectFuncCallBlock = Bcons.COMPOUND(reflectFuncBody);
    pb.replace(reflectFuncCallBlock);

    return reflectFuncCallBlock;
  }

  private static void createGpuReflectNormalShadowFunc(PragmaBlock pb, XMPglobalDecl globalDecl,
                                                       XMPgpuData gpuData, int arrayIndex,
                                                       BlockList reflectFuncBody) {
    XMPalignedArray alignedArray = gpuData.getXMPalignedArray();
    String arrayName = alignedArray.getName();

    // decl buffers
    Ident loSendId = reflectFuncBody.declLocalIdent("_XMP_gpu_reflect_LO_SEND_" + arrayName, Xtype.voidPtrType);
    Ident loRecvId = reflectFuncBody.declLocalIdent("_XMP_gpu_reflect_LO_RECV_" + arrayName, Xtype.voidPtrType);
    Ident hiSendId = reflectFuncBody.declLocalIdent("_XMP_gpu_reflect_HI_SEND_" + arrayName, Xtype.voidPtrType);
    Ident hiRecvId = reflectFuncBody.declLocalIdent("_XMP_gpu_reflect_HI_RECV_" + arrayName, Xtype.voidPtrType);

    // pack shadow
    Ident packFuncId = globalDecl.declExternFunc("_XMP_gpu_pack_shadow_NORMAL");
    XobjList packFuncArgs = Xcons.List(gpuData.getHostDescId().Ref(), loSendId.getAddr(), hiSendId.getAddr(), Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(packFuncId.Call(packFuncArgs)));

    // exchange shadow (using HOST library)
    Ident exchangeFuncId = globalDecl.declExternFunc("_XMP_exchange_shadow_NORMAL");
    XobjList exchangeFuncArgs = Xcons.List(loRecvId.getAddr(), hiRecvId.getAddr(), loSendId.Ref(), hiSendId.Ref());
    exchangeFuncArgs.add(alignedArray.getDescId().Ref());
    exchangeFuncArgs.add(Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(exchangeFuncId.Call(exchangeFuncArgs)));

    // unpack shadow
    Ident unpackFuncId = globalDecl.declExternFunc("_XMP_gpu_unpack_shadow_NORMAL");;
    XobjList unpackFuncArgs = Xcons.List(gpuData.getHostDescId().Ref(), loRecvId.Ref(), hiRecvId.Ref(), Xcons.IntConstant(arrayIndex));

    reflectFuncBody.add(Bcons.Statement(unpackFuncId.Call(unpackFuncArgs)));
  }
}
