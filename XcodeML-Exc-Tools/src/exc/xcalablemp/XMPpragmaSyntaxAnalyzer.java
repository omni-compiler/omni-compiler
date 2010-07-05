package exc.xcalablemp;

import xcodeml.XmException;
import exc.object.*;
import static exc.object.PragmaLexer.*;


/**
 * XcalableMP pragma lexer
 */
public class XMPpragmaSyntaxAnalyzer implements ExternalPragmaLexer {
  private PragmaLexer _lexer;
    
  public XMPpragmaSyntaxAnalyzer(PragmaLexer lexer) {
    _lexer = lexer;
  }

  @Override
  public Result continueLex() throws XmException {
    XobjList[] retArgs = { null };
    PragmaSyntax[] retSyntax = { null };
    XMPpragma pragmaDir = null;

    try {
      pragmaDir = lexXMPdirective(retSyntax, retArgs);
    } catch (XMPexception e) {
      String message = e.getMessage();
      if (message != null) System.out.println("[XcalableMP] " + message);

      return new Result(_lexer, null);
    }

    XobjList xobj = null;
    if (retSyntax[0] == PragmaSyntax.SYN_PREFIX) {
      xobj = Xcons.List(Xcode.XMP_PRAGMA,
                        Xcons.String(retSyntax[0].toString()),
                        Xcons.String(pragmaDir.toString()),
                        retArgs[0]);
    }
    else {
      xobj = Xcons.List(Xcode.XMP_PRAGMA,
                        Xcons.String(pragmaDir.toString()),
                        retArgs[0],
                        null);
      xobj.setIsParsed(true);
    }

    return new Result(_lexer, xobj);
  }

  private boolean pg_is_ident(String name) {
    return _lexer.pg_is_ident(name);
  }

  private void pg_get_token() {
    _lexer.pg_get_token();
  }

  private char pg_tok() {
    return _lexer.pg_tok();
  }

  private String pg_tok_buf() {
    return _lexer.pg_tok_buf();
  }

  private Xobject pg_parse_expr() throws XmException {
    return _lexer.pg_parse_expr();
  }

  private XobjList pg_parse_int_triplet(boolean allowIntExpr) throws XmException, XMPexception {
    Xobject lower = null, upper = null, stride = null;

    // parse <triplet-lower>
    if (pg_tok() == ':') lower = null;
    else {
      lower = pg_parse_int_expr();

      if (allowIntExpr) {
        if (pg_tok() == ',' || pg_tok() == ')') {
          upper = lower;
          stride = null;
          return Xcons.List(lower, upper, stride);
        }
      }

      if (pg_tok() != ':')
        error(": is expected after <triplet-stride>");
    }

    // parse <triplet-upper>
    pg_get_token();
    if (pg_tok() == ',' || pg_tok() == ')') {
      upper = null;
      stride = null;
    }
    else if (pg_tok() == ':') {
      upper = null;

      pg_get_token();
      stride = pg_parse_int_expr();
    }
    else {
      upper = pg_parse_int_expr();

      if (pg_tok() == ':') {
        pg_get_token();
        stride = pg_parse_int_expr();
      }
      else if (pg_tok() == ',' || pg_tok() == ')') stride = null;
      else
        error("syntax error on <triplet>");
    }

    return Xcons.List(lower, upper, stride);
  }

  private Xobject pg_parse_int_expr() throws XmException, XMPexception {
    Xobject x = _lexer.pg_parse_expr();
    if (x == null)
      error("syntax error on <int-expr>");

    if (!XMPutil.isIntegerType(x.Type()))
      error("<int-expr> is expected, parsed expression has a non-integer type");

    return x;
  }

  private void error(String s) throws XMPexception {
    _lexer.error("[XcalableMP] " + s);
    throw new XMPexception();
  }

  private XMPpragma lexXMPdirective(PragmaSyntax[] retSyntax, XobjList[] retArgs) throws XmException, XMPexception {
    PragmaSyntax syntax = null;
    XobjList args = null;
    XMPpragma pragmaDir = null;

    pg_get_token();
    if (pg_tok() != PG_IDENT)
      error("unknown XcalableMP directive, '" + pg_tok_buf() + "'");

    if (pg_is_ident("nodes")) {
      pragmaDir = XMPpragma.NODES;
      syntax = PragmaSyntax.SYN_DECL;

      pg_get_token();
      args = parse_NODES_clause();
    }
    else if (pg_is_ident("template")) {
      pragmaDir = XMPpragma.TEMPLATE;
      syntax = PragmaSyntax.SYN_DECL;

      pg_get_token();
      args = parse_TEMPLATE_clause();
    }
    else if (pg_is_ident("distribute")) {
      pragmaDir = XMPpragma.DISTRIBUTE;
      syntax = PragmaSyntax.SYN_DECL;

      pg_get_token();
      args = parse_DISTRIBUTE_clause();
    }
    else if (pg_is_ident("align")) {
      pragmaDir = XMPpragma.ALIGN;
      syntax = PragmaSyntax.SYN_DECL;

      pg_get_token();
      args = parse_ALIGN_clause();
    }
    else if (pg_is_ident("shadow")) {
      pragmaDir = XMPpragma.SHADOW;
      syntax = PragmaSyntax.SYN_DECL;

      pg_get_token();
      args = parse_SHADOW_clause();
    }
    else if (pg_is_ident("task")) {
      pragmaDir = XMPpragma.TASK;
      syntax = PragmaSyntax.SYN_PREFIX;

      pg_get_token();
      args = parse_TASK_clause();
    }
    else if (pg_is_ident("tasks")) {
      pragmaDir = XMPpragma.TASKS;
      syntax = PragmaSyntax.SYN_PREFIX;

      pg_get_token();
      args = parse_TASKS_clause();
    }
    else if (pg_is_ident("loop")) {
      pragmaDir = XMPpragma.LOOP;
      syntax = PragmaSyntax.SYN_PREFIX;

      pg_get_token();
      args = parse_LOOP_clause();
    }
    else if (pg_is_ident("reflect")) {
      pragmaDir = XMPpragma.REFLECT;
      syntax = PragmaSyntax.SYN_EXEC;

      pg_get_token();
      args = parse_REFLECT_clause();
    }
    else if (pg_is_ident("barrier")) {
      pragmaDir = XMPpragma.BARRIER;
      syntax = PragmaSyntax.SYN_EXEC;

      pg_get_token();
      args = parse_BARRIER_clause();
    }
    else if (pg_is_ident("reduction")) {
      pragmaDir = XMPpragma.REDUCTION;
      syntax = PragmaSyntax.SYN_EXEC;

      pg_get_token();
      args = parse_REDUCTION_clause();
    }
    else if (pg_is_ident("bcast")) {
      pragmaDir = XMPpragma.BCAST;
      syntax = PragmaSyntax.SYN_EXEC;

      pg_get_token();
      args = parse_BCAST_clause();
    }
    else
      error("unknown XcalableMP directive, '" + pg_tok_buf() + "'");
 
    if (pg_tok() != 0 || pragmaDir == null)
      error("extra arguments for XcalableMP directive");

    retSyntax[0] = syntax;
    retArgs[0] = args;

    return pragmaDir;
  }

  private XobjList parse_NODES_clause() throws XmException, XMPexception {
    // parse [<map-type>]
    XobjInt mapType = null;
    if (pg_tok() == '(') {
      pg_get_token();
      if (pg_is_ident("regular")) {
        mapType = Xcons.IntConstant(XMPnodes.MAP_REGULAR);

        pg_get_token();
        if (pg_tok() != ')')
          error("')' is expected after <map-type>");
        else pg_get_token();
      }
      else
        error("'" + pg_tok_buf() + "' is not allowed for <map-type>");
    }
    else mapType = null;

    // parse <nodes-name>
    if (pg_tok() != PG_IDENT)
      error("nodes directive has no <nodes-name>");

    XobjString nodesName = Xcons.String(pg_tok_buf());

    // parse (<nodes-size>, ...)
    pg_get_token();
    if (pg_tok() != '(')
      error("'(' is expected after <nodes-name>");

    XobjList nodesSizeList = Xcons.List();
    do {
      // parse <nodes-size> := { * | <int-expr> }
      pg_get_token();
      if(pg_tok() == '*') {
        pg_get_token();
        if (pg_tok() == ')') {
          nodesSizeList.add(null);
          break;
        }
        else
          error("'*' can be used only in the last demension");
      }
      else nodesSizeList.add(pg_parse_int_expr());

      if (pg_tok() == ')') break;
      else if (pg_tok() == ',') continue;
      else
        error("')' or ',' is expected after <nodes-size>");
    } while (true);

    // parse { <empty> | =* | =<nodes-ref> }
    XobjList inheritedNodes = null;
    pg_get_token();
    if (pg_tok() == '=') {
      pg_get_token();
      if (pg_tok() == '*') {
        inheritedNodes = Xcons.List(Xcons.IntConstant(XMPnodes.INHERIT_EXEC), null);

        pg_get_token();
      }
      else inheritedNodes = Xcons.List(Xcons.IntConstant(XMPnodes.INHERIT_NODES), parse_ON_REF(false, false));
    }
    else inheritedNodes = Xcons.List(Xcons.IntConstant(XMPnodes.INHERIT_GLOBAL));

    return Xcons.List(mapType, nodesName, nodesSizeList, inheritedNodes);
  }

  private XobjList parse_TEMPLATE_clause() throws XmException, XMPexception {
    // parse <template-name>
    if (pg_tok() != PG_IDENT)
      error("template directive has no <template-name>");

    XobjString templateName = Xcons.String(pg_tok_buf());

    // parse (<template-spec>, ...)
    pg_get_token();
    if (pg_tok() != '(')
      error("'(' is expected after <template-name>");

    XobjList templateSpecList = Xcons.List();
    do {
      // parse <template-spec> := { : | [int-expr :] int-expr }
      pg_get_token();
      if (pg_tok() == ':') {
        pg_get_token();
        if (pg_tok() == ')') {
          templateSpecList.add(null);
          break;
        }
        else
          error("':' can be used only in the last demension");
      }
      else {
        Xobject templateSpec = pg_parse_int_expr();
        if (pg_tok() == ':') {
          pg_get_token();
          templateSpecList.add(Xcons.List(templateSpec, pg_parse_int_expr()));
        }
        else templateSpecList.add(Xcons.List(Xcons.IntConstant(1), templateSpec));
      }

      if (pg_tok() == ')') break;
      else if (pg_tok() == ',') continue;
      else
        error("')' or ',' is expected after <template-spec>");
    } while (true);

    pg_get_token();
    return Xcons.List(templateName, templateSpecList);
  }

  private XobjList parse_DISTRIBUTE_clause() throws XMPexception {
    // parse <template-name>
    if (pg_tok() != PG_IDENT)
      error("distribute directive has no <template-name>");

    XobjString templateName = Xcons.String(pg_tok_buf());

    // parse (<dist-format>, ...)
    pg_get_token();
    if (pg_tok() != '(')
      error("'(' is expected after <template-name>");

    XobjList distFormatList = Xcons.List();
    do {
      // FIXME support cyclic(w), gblock
      // parse <dist-format> := { * | block | cyclic }
      pg_get_token();
      if (pg_tok() == '*')		distFormatList.add(Xcons.IntConstant(XMPtemplate.DUPLICATION));
      else if (pg_is_ident("block"))	distFormatList.add(Xcons.IntConstant(XMPtemplate.BLOCK));
      else if (pg_is_ident("cyclic"))	distFormatList.add(Xcons.IntConstant(XMPtemplate.CYCLIC));
      else
        error("unknown distribution manner");

      pg_get_token();
      if (pg_tok() == ')') break;
      else if (pg_tok() == ',') continue;
      else
        error("')' or ',' is expected after <dist-format>");
    } while (true);

    // parse <onto-clause>
    pg_get_token();
    if (!pg_is_ident("onto"))
      error("distribute directive has no <onto-clause>");

    pg_get_token();
    XobjString nodesName = null;
    if (pg_tok() == PG_IDENT) nodesName = Xcons.String(pg_tok_buf());
    else
      error("<nodes-name> is expected after 'onto'");

    pg_get_token();
    return Xcons.List(templateName, distFormatList, nodesName);
  }

  private XobjList parse_ALIGN_clause() throws XmException, XMPexception {
    // parse <array-name>
    if (pg_tok() != PG_IDENT)
      error("align directive has no <array-name>");

    XobjString arrayName = Xcons.String(pg_tok_buf());

    // parse [align-source] ...
    XobjList alignSourceList = Xcons.List();
    pg_get_token();
    if (pg_tok() != '[')
      error("'[' is expected after <array-name>");

    pg_get_token();
    parse_ALIGN_SOURCE(alignSourceList);

    if (pg_tok() != ']')
      error("']' is expected after <align-source>");

    do {
      pg_get_token();
      if (pg_tok() == '[') {
        pg_get_token();
        parse_ALIGN_SOURCE(alignSourceList);

        if (pg_tok() != ']')
          error("']' is expected after <align-source>");
      }
      else break;
    } while (true);

    // parse <template-name>
    pg_get_token();
    if (pg_tok() != PG_IDENT)
      error("align directive has no <template-name>");

    XobjString templateName = Xcons.String(pg_tok_buf());

    // parse [align-subscript] ...
    pg_get_token();
    if (pg_tok() != '(')
      error("'(' is expected after <template-name>");

    XobjList alignSubscriptList = Xcons.List(Xcons.List(), Xcons.List());
    do {
      pg_get_token();
      parse_ALIGN_SUBSCRIPT(alignSubscriptList);

      if (pg_tok() == ')') break;
      else if (pg_tok() == ',') continue;
      else
        error("')' or ',' is expected after <align-subscript>");
    } while (true);

    pg_get_token();
    return Xcons.List(arrayName, alignSourceList, templateName, alignSubscriptList);
  }

  private void parse_ALIGN_SOURCE(XobjList alignSourceList) throws XMPexception {
    if (pg_tok() == '*')
      alignSourceList.add(Xcons.IntConstant(XMPalignedArray.NO_ALIGN));
    else if (pg_tok() == ':')
      alignSourceList.add(Xcons.IntConstant(XMPalignedArray.SIMPLE_ALIGN));
    else if (pg_tok() == PG_IDENT)
      alignSourceList.add(Xcons.String(pg_tok_buf()));
    else
      error("'*' or <scalar-int-variable> is expected for <align-source>");

    pg_get_token();
  }

  private void parse_ALIGN_SUBSCRIPT(XobjList alignSubscriptList) throws XmException, XMPexception {
    if (pg_tok() == '*') {
      alignSubscriptList.left().add(Xcons.IntConstant(XMPalignedArray.NO_ALIGN));
      alignSubscriptList.right().add(null);
      pg_get_token();
      return;
    }
    else if (pg_tok() == ':') {
      alignSubscriptList.left().add(Xcons.IntConstant(XMPalignedArray.SIMPLE_ALIGN));
      alignSubscriptList.right().add(null);
      pg_get_token();
      return;
    }
    else if (pg_tok() == PG_IDENT) {
      XobjString var = Xcons.String(pg_tok_buf());
      alignSubscriptList.left().add(var);

      Xobject expr = null;
      pg_get_token();
      if (pg_tok() == '+') {
        pg_get_token();
        expr = pg_parse_int_expr();
      }
      else if (pg_tok() == '-') expr = pg_parse_int_expr();
      else expr = null;
      alignSubscriptList.right().add(expr);

      return;
    }
    else
      error("'*' or \"<scalar-int-variable> [+/- <int-expr>]\" is expected for <align-subscript>");
  }

  private XobjList parse_SHADOW_clause() throws XmException, XMPexception {
    // parse <array-name>
    if (pg_tok() != PG_IDENT)
      error("shadow directive has no <array-name>");

    XobjString arrayName = Xcons.String(pg_tok_buf());

    // parse [shadow-width] ...
    pg_get_token();
    XobjList shadowWidthList = Xcons.List();
    if (pg_tok() != '[')
      error("'[' is expected after <array-name>");

    pg_get_token();
    parse_SHADOW_WIDTH(shadowWidthList);

    if (pg_tok() != ']')
      error("']' is expected after <shadow-width>");

    do {
      pg_get_token();
      if (pg_tok() == '[') {
        pg_get_token();
        parse_SHADOW_WIDTH(shadowWidthList);

        if (pg_tok() != ']')
          error("']' is expected after <shadow-width>");
      }
      else break;
    } while (true);

    return Xcons.List(arrayName, shadowWidthList);
  }

  private void parse_SHADOW_WIDTH(XobjList shadowWidthList) throws XmException, XMPexception {
    if (pg_tok() == '*') {
      shadowWidthList.add(null);

      pg_get_token();
      return;
    }
    else {
      Xobject width = pg_parse_int_expr();
      if (pg_tok() == ':') {
        pg_get_token();
        shadowWidthList.add(Xcons.List(width, pg_parse_int_expr()));
        return;
      }
      else {
        shadowWidthList.add(Xcons.List(width, width));
        return;
      }
    }
  }

  private XobjList parse_TASK_clause() throws XmException, XMPexception {
    if (!pg_is_ident("on"))
      error("task directive has no <on-ref>");

    // parse <on-ref>
    pg_get_token();
    return Xcons.List(parse_ON_REF(true, false));
  }

  private XobjList parse_TASKS_clause() {
    XobjInt nowait = null;
    if (pg_is_ident("nowait")) {
      nowait = Xcons.IntConstant(XMPtask.NOWAIT_TASKS);
      pg_get_token();
    }
    else nowait = null;

    return Xcons.List(nowait);

    // check body in translator: task directive list
  }

  private XobjList parse_LOOP_clause() throws XmException, XMPexception {
    XobjList loopIndexList = null;
    if (pg_tok() == '(') {
      loopIndexList = Xcons.List();
      do {
        pg_get_token();
        if (pg_tok() == PG_IDENT) loopIndexList.add(Xcons.String(pg_tok_buf()));
        else
          error("<loop-index> is not found");

        pg_get_token();
        if (pg_tok() == ',') continue;
        else if (pg_tok() == ')') break;
        else
          error("')' or ',' is expected after <loop-index>");
      } while (true);

      pg_get_token();
    }

    if (!pg_is_ident("on"))
      error("loop directive has no <on-ref>");

    // parse <on-ref>
    pg_get_token();
    XobjList onRef = parse_ON_REF(true, true);

    // parse [<reduction-ref>], ...
    XobjList reductionRefList = null;
    if (pg_is_ident("reduction")) {
      reductionRefList = Xcons.List();

      pg_get_token();
      reductionRefList.add(parse_REDUCTION_REF());

      do {
        if (pg_is_ident("reduction")) {
          pg_get_token();
          reductionRefList.add(parse_REDUCTION_REF());
        }
        else break;
      } while (true);
    }

    return Xcons.List(loopIndexList, onRef, reductionRefList);

    // check body in translator: for loop
  }

  private XobjList parse_REFLECT_clause() throws XmException, XMPexception {
    XobjList arrayNameList = Xcons.List();
    do {
      pg_get_token();
      if (pg_tok() == PG_IDENT) arrayNameList.add(Xcons.String(pg_tok_buf()));
      else
        error("<array-name> is not found");

      pg_get_token();
      if (pg_tok() == ',') continue;
      else break;
    } while (true);

    return Xcons.List(arrayNameList);
  }

  private XobjList parse_BARRIER_clause() throws XmException, XMPexception {
    if (pg_is_ident("on")) {
      pg_get_token();
      return Xcons.List(parse_ON_REF(true, false));
    }
    else return Xcons.List((Xobject)null);
  }

  private XobjList parse_REDUCTION_clause() throws XmException, XMPexception {
    XobjList reductionRef = parse_REDUCTION_REF();

    if (pg_is_ident("on")) {
      pg_get_token();
      return Xcons.List(reductionRef, parse_ON_REF(true, false));
    }
    else return Xcons.List(reductionRef, null);
  }

  private XobjList parse_BCAST_clause() throws XmException, XMPexception {
    XobjList varList = Xcons.List();
    do {
      if (pg_tok() != PG_IDENT)
        error("<variable> for bcast directive is expected");
      else varList.add(Xcons.String(pg_tok_buf()));

      pg_get_token();
      if (pg_tok() == ',') {
        pg_get_token();
        continue;
      }
      else break;
    } while (true);

    XobjList fromRef = null;
    if (pg_is_ident("from")) {
      pg_get_token();
      fromRef = parse_ON_REF(true, false);
    }
    else fromRef = null;

    XobjList onRef = null;
    if (pg_is_ident("on")) {
      pg_get_token();
      onRef = parse_ON_REF(true, false);
    }
    else onRef = null;

    return Xcons.List(varList, fromRef, onRef);
  }

  private XobjList parse_ON_REF(boolean isExecPragma, boolean isLoopPragma) throws XmException, XMPexception {
    if (pg_tok() == PG_IDENT) {
      // parse <named-obj-ref>
      XobjString objName = Xcons.String(pg_tok_buf());

      pg_get_token();
      if (pg_tok() != '(') return Xcons.List(objName, null);

      XobjList objSubscriptList = Xcons.List();
      do {
        pg_get_token();
        parse_OBJ_SUBSCRIPT(objSubscriptList, isExecPragma, isLoopPragma);

        if (pg_tok() == ')') break;
        else if (pg_tok() == ',') continue;
        else
          error("')' or ',' is expected after <nodes/template-subscript>");
      } while (true);

      pg_get_token();
      return Xcons.List(objName, objSubscriptList);
    }
    else {
      if (isLoopPragma)
        error("<node-number-ref> cannot be used in for directive");

      // parse <node-number-ref>
      XobjList nodeNumberRef = null;
      if (pg_tok() == '(') {
        pg_get_token();
        nodeNumberRef = pg_parse_int_triplet(false);

        if (pg_tok() == ')') pg_get_token();
        else
          error("')' is expected after <triplet>");
      }
      else {
        Xobject nodeNumber = pg_parse_int_expr();
        nodeNumberRef = Xcons.List(nodeNumber, nodeNumber, Xcons.IntConstant(1));
      }

      return Xcons.List((Xobject)null, nodeNumberRef);
    }
  }

  private void parse_OBJ_SUBSCRIPT(XobjList nodesSubscriptList,
                                   boolean isExecPragma, boolean isLoopPragma) throws XmException, XMPexception {
    if (isLoopPragma) {
      if (pg_tok() == '*') nodesSubscriptList.add(Xcons.String(XMP.ASTERISK));
      else if (pg_tok() == ':') nodesSubscriptList.add(Xcons.String(XMP.COLON));
      else {
        if (pg_tok() == PG_IDENT) nodesSubscriptList.add(Xcons.String(pg_tok_buf()));
        else error("syntax error on <nodes/template-subscript> in for directive");
      }

      pg_get_token();
      return;
    }
    else {
      if (pg_tok() == '*') {
        if (isExecPragma) {
          nodesSubscriptList.add(null);

          pg_get_token();
          return;
        }
        else
          error("'*' can be used only in execution directives");
      }
      else {
        nodesSubscriptList.add(pg_parse_int_triplet(true));
        return;
      }
    }
  }

  private XobjList parse_REDUCTION_REF() throws XMPexception {
    // 'reduction' is already parsed
    if (pg_tok() != '(')
      error("'(' is expected after 'reduction'");

    // parse <reduction-kind>
    pg_get_token();
    XobjInt reductionKind = parse_REDUCTION_KIND();

    if (pg_tok() != ':')
      error("':' is expected after <reduction-kind>");

    // parse <reduction-spec>, ...
    XobjList reductionSpecList = Xcons.List();
    do {
      pg_get_token();
      reductionSpecList.add(parse_REDUCTION_SPEC());

      if (pg_tok() == ')') break;
      else if (pg_tok() == ',') continue;
      else
        error("')' or ',' is expected after <reduction-spec>");
    } while (true);

    pg_get_token();
    return Xcons.List(reductionKind, reductionSpecList);
  }

  private XobjInt parse_REDUCTION_KIND() throws XMPexception {
    // FIXME incomplete implementation
    XobjInt reductionKind = null;
    if (pg_tok() == '+') reductionKind =  Xcons.IntConstant(XMPcollective.REDUCE_SUM);
    else if (pg_tok() == '*') reductionKind =  Xcons.IntConstant(XMPcollective.REDUCE_PROD);
    else
      error("'" + pg_tok_buf() +  "' is not allowed for <reduction-spec>");

    pg_get_token();
    return reductionKind;
  }

  private XobjString parse_REDUCTION_SPEC() throws XMPexception {
    // FIXME incomplete implementation
    if (pg_tok() == PG_IDENT) {
      pg_get_token();
      return Xcons.String(pg_tok_buf());
    }
    else
      error("syntax error on <reduction-spec>");

    // not reach here
    return null;
  }
}
