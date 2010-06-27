package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPrewriteExpr {
  private XMPglobalDecl		_globalDecl;
  private XMPobjectTable	_globalObjectTable;

  public XMPrewriteExpr(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
    _globalObjectTable = globalDecl.getGlobalObjectTable();
  }

  public void rewrite(FuncDefBlock def) throws XMPexception {
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // rewrite expr
    XMPobjectTable localObjectTable = XMPlocalDecl.getObjectTable(fb);

    BasicBlockExprIterator iter = new BasicBlockExprIterator(fb);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject x = iter.getExpr();
      if (x == null) continue;

      Xobject newExpr = rewriteExpr(x, localObjectTable);
      if (newExpr == null) continue;

      iter.setExpr(newExpr);
    }

    // create local object descriptors, constructors and desctructors
    XMPlocalDecl.setupObjectId(fb);
    XMPlocalDecl.setupConstructor(fb);
    XMPlocalDecl.setupDestructor(fb);

    def.Finalize();
  }

  public Xobject rewriteExpr(Xobject expr, XMPobjectTable localObjectTable) throws XMPexception {
    if (expr == null) return null;

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject newExpr = null;
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) continue;

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            XMPalignedArray alignedArray = _globalObjectTable.getAlignedArray(myExpr.getSym());
            if (alignedArray != null) {
              if (alignedArray.checkRealloc()) 
                rewriteAlignedArrayExpr(iter, alignedArray);
            }

            break;
          }
        case VAR:
          {
            if (localObjectTable == null) break;

            XMPalignedArray alignedArray = localObjectTable.getAlignedArray(myExpr.getSym());
            if (alignedArray != null) {
              if (alignedArray.checkRealloc())
                rewriteAlignedArrayExpr(iter, alignedArray);
            }

            break;
          }
        default:
      }
    }

    return iter.topXobject();
  }

  private void rewriteAlignedArrayExpr(bottomupXobjectIterator iter, XMPalignedArray alignedArray) throws XMPexception {
    XobjList getAddrFuncArgs = Xcons.List(alignedArray.getAddrId().Ref());

    int arrayDimCount = parseArrayExpr(iter, alignedArray, 0, getAddrFuncArgs);
    int arrayDim = alignedArray.getDim();
    Ident getAddrFuncId = null;
    if (arrayDimCount == arrayDim) {
      getAddrFuncId = XMP.getMacroId("_XCALABLEMP_M_GET_ELMT_" + arrayDim);
      for (int i = 0; i < arrayDim - 1; i++) {
        getAddrFuncArgs.add(alignedArray.getGtolAccIdAt(i).Ref());
      }
    }
    else {
      getAddrFuncId = XMP.getMacroId("_XCALABLEMP_M_GET_ADDR_" + arrayDimCount);
      for (int i = 0; i < arrayDimCount; i++) {
        getAddrFuncArgs.add(alignedArray.getGtolAccIdAt(i).Ref());
      }
    }

    iter.setXobject(getAddrFuncId.Call(getAddrFuncArgs));
  }

  private int parseArrayExpr(bottomupXobjectIterator iter,
                             XMPalignedArray alignedArray, int arrayDimCount, XobjList args) throws XMPexception {
    // myExpr: VAR, ARRAY_REF, POINTER_REF, PLUS_EXPR
    Xobject myExpr = iter.getXobject();
    Xobject parentExpr = iter.getParent();
    if (parentExpr == null)
      XMP.error(myExpr.getLineNo(), "incorrect array reference");

    switch (parentExpr.Opcode()) {
      case PLUS_EXPR:
        iter.next(); // goto PLUS_EXPR
        myExpr = iter.getXobject();
        args.add(getCalcIndexFuncRef(alignedArray, arrayDimCount, myExpr.right()));
        return parseArrayExpr(iter, alignedArray, arrayDimCount+1, args);
      case POINTER_REF:
        iter.next();
        return parseArrayExpr(iter, alignedArray, arrayDimCount, args);
      default:
        return arrayDimCount;
    }
  }

  private Xobject getCalcIndexFuncRef(XMPalignedArray alignedArray, int index, Xobject indexRef) throws XMPexception {
    int distManner = alignedArray.getDistMannerAt(index);
    switch (distManner) {
      case XMPalignedArray.NO_ALIGN:
      case XMPtemplate.DUPLICATION:
        return indexRef;
      case XMPtemplate.BLOCK:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref(),
                                           shadow.getLo());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK_W_SHADOW").Call(args);
              }
            default:
              XMP.error(alignedArray.getLineNo(), "unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_BLOCK").Call(args);
        }
      case XMPtemplate.CYCLIC:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_CYCLIC").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              XMP.error(alignedArray.getLineNo(), "only block distribution allows shadow");
            default:
              XMP.error(alignedArray.getLineNo(), "unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XCALABLEMP_M_CALC_INDEX_CYCLIC").Call(args);
        }
      default:
        XMP.error(alignedArray.getLineNo(), "unknown distribute manner for array '" + alignedArray.getName()  + "'");
    }

    // XXX not reach here
    return null;
  }
}
