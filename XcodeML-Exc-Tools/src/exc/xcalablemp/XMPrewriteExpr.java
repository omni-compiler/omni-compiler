/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPrewriteExpr {
  private XMPglobalDecl		_globalDecl;

  public XMPrewriteExpr(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void rewrite(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // get symbol table
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(fb);

    // rewrite parameters
    rewriteParams(fb, localXMPsymbolTable);

    // rewrite declarations
    rewriteDecls(fb, localXMPsymbolTable);

    // rewrite Function Exprs
    rewriteFuncExprs(fb, localXMPsymbolTable);

    // create local object descriptors, constructors and desctructors
    XMPlocalDecl.setupObjectId(fb);
    XMPlocalDecl.setupConstructor(fb);
    XMPlocalDecl.setupDestructor(fb);

    def.Finalize();
  }

  private void rewriteParams(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    XobjList identList = funcBlock.getBody().getIdentList();
    if (identList == null) {
      return;
    } else {
      for(Xobject x : identList) {
        Ident id = (Ident)x;
        XMPalignedArray alignedArray = localXMPsymbolTable.getXMPalignedArray(id.getName());
        if (alignedArray != null) {
          id.setType(Xtype.Pointer(alignedArray.getType()));
        }
      }
    }
  }

  private void rewriteDecls(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    topdownBlockIterator iter = new topdownBlockIterator(funcBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      if (bl != null) {
        XobjList decls = (XobjList)bl.getDecls();
        if (decls != null) {
          try {
            for (Xobject x : decls) {
              rewriteExpr(x.getArg(1), localXMPsymbolTable);
            }
          } catch (XMPexception e) {
            XMP.error(b.getLineNo(), e.getMessage());
          }
        }
      }
    }
  }

  private void rewriteFuncExprs(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    BasicBlockExprIterator iter = new BasicBlockExprIterator(funcBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();

      try {
        rewriteExpr(expr, localXMPsymbolTable);
      } catch (XMPexception e) {
        XMP.error(expr.getLineNo(), e.getMessage());
      }
    }
  }

  private void rewriteExpr(Xobject expr, XMPsymbolTable localXMPsymbolTable) throws XMPexception {
    if (expr == null) return;

    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            String arrayName = myExpr.getArg(0).getSym();
            XMPalignedArray alignedArray = findXMPalignedArray(arrayName, localXMPsymbolTable);
            if (alignedArray != null) {
              if (alignedArray.checkRealloc()) {
                iter.setXobject(rewriteAlignedArrayExpr((XobjList)myExpr.getArg(1), alignedArray));
              }
            }
          } break;
        case SUB_ARRAY_REF:
          System.out.println(myExpr.toString());
          break;
        default:
      }
    }
  }

  private Xobject rewriteAlignedArrayExpr(XobjList refExprList,
                                          XMPalignedArray alignedArray) throws XMPexception {
    int arrayDimCount = 0;
    XobjList args = Xcons.List(alignedArray.getAddrId().Ref());
    if (refExprList != null) {
      for (Xobject x : refExprList) {
        args.add(getCalcIndexFuncRef(alignedArray, arrayDimCount, x));
        arrayDimCount++;
      }
    }

    return createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args);
  }

  private Xobject createRewriteAlignedArrayFunc(XMPalignedArray alignedArray, int arrayDimCount,
                                                XobjList getAddrFuncArgs) throws XMPexception {
    int arrayDim = alignedArray.getDim();
    Ident getAddrFuncId = null;

    if (arrayDim < arrayDimCount) {
      throw new XMPexception("wrong array ref");
    } else if (arrayDim == arrayDimCount) {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_E_" + arrayDim, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDim - 1; i++)
        getAddrFuncArgs.add(alignedArray.getAccIdAt(i).Ref());
    } else {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_" + arrayDimCount, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDimCount; i++)
        getAddrFuncArgs.add(alignedArray.getAccIdAt(i).Ref());
    }

    Xobject retObj = getAddrFuncId.Call(getAddrFuncArgs);
    if (arrayDim == arrayDimCount) {
      return Xcons.List(Xcode.POINTER_REF, retObj.Type(), retObj);
    } else {
      return retObj;
    }
  }

  private Xobject getCalcIndexFuncRef(XMPalignedArray alignedArray, int index, Xobject indexRef) throws XMPexception {
    switch (alignedArray.getAlignMannerAt(index)) {
      case XMPalignedArray.NOT_ALIGNED:
      case XMPalignedArray.DUPLICATION:
        return indexRef;
      case XMPalignedArray.BLOCK:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
            case XMPshadow.SHADOW_NORMAL:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
        }
      case XMPalignedArray.CYCLIC:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              throw new XMPexception("only block distribution allows shadow");
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
        }
      default:
        throw new XMPexception("unknown align manner for array '" + alignedArray.getName()  + "'");
    }
  }

  private XMPalignedArray findXMPalignedArray(String arrayName, XMPsymbolTable localXMPsymbolTable) throws XMPexception {
    XMPalignedArray a = localXMPsymbolTable.getXMPalignedArray(arrayName);
    if (a == null) {
      a = _globalDecl.getXMPalignedArray(arrayName);
    }

    return a;
  }
}
