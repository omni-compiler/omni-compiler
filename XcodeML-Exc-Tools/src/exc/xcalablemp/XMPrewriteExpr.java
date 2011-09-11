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
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            Xobject arrayAddr = myExpr.getArg(0);
            String arrayName = arrayAddr.getSym();
            XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, localXMPsymbolTable);
            if (alignedArray != null) {
              Xobject newExpr = null;
              XobjList arrayRefList = normArrayRefList((XobjList)myExpr.getArg(1), alignedArray);
              if (alignedArray.checkRealloc()) {
                newExpr = rewriteAlignedArrayExpr(arrayRefList, alignedArray);
              } else {
                newExpr = Xcons.arrayRef(myExpr.Type(), arrayAddr, arrayRefList);
              }

              newExpr.setIsRewrittedByXmp(true);
              iter.setXobject(newExpr);
            }
          } break;
        case SUB_ARRAY_REF:
          System.out.println(myExpr.toString());
          break;
        default:
      }
    }
  }

  public static XobjList normArrayRefList(XobjList refExprList,
                                          XMPalignedArray alignedArray) {
    if (refExprList == null) {
      return null;
    } else {
      XobjList newRefExprList = Xcons.List();

      int arrayIndex = 0;
      for (Xobject x : refExprList) {
        Xobject normExpr = alignedArray.getAlignNormExprAt(arrayIndex);
        if (normExpr != null) {
          newRefExprList.add(Xcons.binaryOp(Xcode.PLUS_EXPR, x, normExpr));
        } else {
          newRefExprList.add(x);
        }

        arrayIndex++;
      }

      return newRefExprList;
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

  public static Xobject createRewriteAlignedArrayFunc(XMPalignedArray alignedArray, int arrayDimCount,
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
      case XMPalignedArray.BLOCK_CYCLIC:
        {
          XMPtemplate t = alignedArray.getAlignTemplate();
          int ti = alignedArray.getAlignSubscriptIndexAt(index).intValue();
          XMPnodes n = t.getOntoNodes();
          int ni = t.getOntoNodesIndexAt(ti).getInt();

          if (alignedArray.hasShadow()) {
            XMPshadow shadow = alignedArray.getShadowAt(index);
            switch (shadow.getType()) {
              case XMPshadow.SHADOW_NONE:
                {
                  XobjList args = Xcons.List(indexRef, n.getSizeAt(ni), t.getWidthAt(ti));
                  return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK_CYCLIC").Call(args);
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
            XobjList args = Xcons.List(indexRef, n.getSizeAt(ni), t.getWidthAt(ti));
            return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK_CYCLIC").Call(args);
          }
        }
      default:
        throw new XMPexception("unknown align manner for array '" + alignedArray.getName()  + "'");
    }
  }

  public static void rewriteArrayRefInLoop(Xobject expr,
                                           XMPglobalDecl globalDecl, XMPsymbolTable localXMPsymbolTable) throws XMPexception {
    if (expr == null) return;

    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            Xobject arrayAddr = myExpr.getArg(0);
            String arrayName = arrayAddr.getSym();
            XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(arrayName, localXMPsymbolTable);
            if (alignedArray != null) {
              Xobject newExpr = null;
              XobjList arrayRefList = XMPrewriteExpr.normArrayRefList((XobjList)myExpr.getArg(1), alignedArray);
              if (alignedArray.checkRealloc()) {
                newExpr = XMPrewriteExpr.rewriteAlignedArrayExprInLoop(arrayRefList, alignedArray);
              } else {
                newExpr = Xcons.arrayRef(myExpr.Type(), arrayAddr, arrayRefList);
              }

              newExpr.setIsRewrittedByXmp(true);
              iter.setXobject(newExpr);
            }
          } break;
        default:
      }
    }
  }

  private static Xobject rewriteAlignedArrayExprInLoop(XobjList refExprList,
                                                       XMPalignedArray alignedArray) throws XMPexception {
    int arrayDimCount = 0;
    XobjList args = Xcons.List(alignedArray.getAddrId().Ref());
    if (refExprList != null) {
      for (Xobject x : refExprList) {
        args.add(x);
        arrayDimCount++;
      }
    }

    return XMPrewriteExpr.createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args);
  }

  public static void rewriteLoopIndexInLoop(Xobject expr, String loopIndexName,
                                            XMPtemplate templateObj, int templateIndex,
                                            XMPglobalDecl globalDecl, XMPsymbolTable localXMPsymbolTable) throws XMPexception {
    if (expr == null) return;

    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case VAR:
          {
            if (loopIndexName.equals(myExpr.getSym())) {
              iter.setXobject(calcLtoG(templateObj, templateIndex, myExpr));
            }
          } break;
        case ARRAY_REF:
          {
            XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(myExpr.getArg(0).getSym(), localXMPsymbolTable);
            if (alignedArray == null) {
              rewriteLoopIndexVar(templateObj, templateIndex, loopIndexName, myExpr);
            } else {
              myExpr.setArg(1, rewriteLoopIndexArrayRefList(templateObj, templateIndex, alignedArray,
                                                            loopIndexName, (XobjList)myExpr.getArg(1)));
            }
          } break;
        default:
      }
    }
  }

  private static void rewriteLoopIndexVar(XMPtemplate templateObj, int templateIndex,
                                          String loopIndexName, Xobject expr) throws XMPexception {
    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case VAR:
          {
            if (loopIndexName.equals(myExpr.getString())) {
              Xobject newExpr = calcLtoG(templateObj, templateIndex, myExpr);
              iter.setXobject(newExpr);
            }
          } break;
        default:
      }
    }
  }

  private static XobjList rewriteLoopIndexArrayRefList(XMPtemplate t, int ti, XMPalignedArray a,
                                                       String loopIndexName, XobjList arrayRefList) throws XMPexception {
    if (arrayRefList == null) {
      return null;
    }

    XobjList newArrayRefList = Xcons.List();

    int arrayDimIdx = 0;
    for (Xobject x : arrayRefList) {
      newArrayRefList.add(rewriteLoopIndexArrayRef(t, ti, a, arrayDimIdx, loopIndexName, x));
      arrayDimIdx++;
    }

    return newArrayRefList;
  }

  private static Xobject rewriteLoopIndexArrayRef(XMPtemplate t, int ti,
                                                  XMPalignedArray a, int ai,
                                                  String loopIndexName, Xobject arrayRef) throws XMPexception {
    if (arrayRef.Opcode() == Xcode.VAR) {
      if (loopIndexName.equals(arrayRef.getString())) {
        return calcShadow(t, ti, a, ai, arrayRef);
      } else {
        return arrayRef;
      }
    }

    topdownXobjectIterator iter = new topdownXobjectIterator(arrayRef);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case VAR:
          {
            if (loopIndexName.equals(myExpr.getString())) {
              iter.setXobject(calcShadow(t, ti, a, ai, myExpr));
            }
          } break;
        default:
      }
    }

    return arrayRef;
  }

  private static Xobject calcShadow(XMPtemplate t, int ti, XMPalignedArray a, int ai,
                                    Xobject expr) throws XMPexception {
    expr.setIsRewrittedByXmp(true);

    XMPtemplate alignedTemplate = a.getAlignTemplate();
    if (t != alignedTemplate) {
      throw new XMPexception("array '" + a.getName() + "' is aligned by template '" + alignedTemplate.getName() +
                             "'. loop is distributed by template '" + t.getName() + "'.");
    }

    if (ti != a.getAlignSubscriptIndexAt(ai).intValue()) {
      throw new XMPexception("array ref is not consistent with array alignment");
    }

    XMPshadow shadow = a.getShadowAt(ai);
    switch (shadow.getType()) {
      case XMPshadow.SHADOW_NONE:
        return expr;
      case XMPshadow.SHADOW_NORMAL:
        return Xcons.binaryOp(Xcode.PLUS_EXPR, expr, shadow.getLo());
      case XMPshadow.SHADOW_FULL:
        return calcLtoG(t, ti, expr);
      default:
        throw new XMPexception("unknown shadow type");
    }
  }

  private static Xobject calcLtoG(XMPtemplate t, int ti, Xobject expr) throws XMPexception {
    expr.setIsRewrittedByXmp(true);

    if (!t.isDistributed()) {
      return expr;
    }

    XMPnodes n = t.getOntoNodes();
    int ni = t.getOntoNodesIndexAt(ti).getInt();

    XobjList args = null;
    switch (t.getDistMannerAt(ti)) {
      case XMPtemplate.DUPLICATION:
        return expr;
      case XMPtemplate.BLOCK:
        // _XMP_M_LTOG_TEMPLATE_BLOCK(_l, _m, _N, _P, _p)
        args = Xcons.List(expr, t.getLowerAt(ti), t.getSizeAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      case XMPtemplate.CYCLIC:
        // _XMP_M_LTOG_TEMPLATE_CYCLIC(_l, _m, _P, _p)
        args = Xcons.List(expr, t.getLowerAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      case XMPtemplate.BLOCK_CYCLIC:
        // _XMP_M_LTOG_TEMPLATE_BLOCK_CYCLIC(_l, _b, _m, _P, _p)
        args = Xcons.List(expr, t.getWidthAt(ti), t.getLowerAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      default:
        throw new XMPexception("unknown distribution manner");
    }

    return XMP.getMacroId("_XMP_M_LTOG_TEMPLATE_" + t.getDistMannerStringAt(ti), Xtype.intType).Call(args);
  }
}
