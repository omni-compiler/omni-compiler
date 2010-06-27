/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-compile-opt.c
 */

#define ICON_EQ(z, c)  (IS_ICON(z) && TN_CONST_INT(z) == (c))
#define COMMUTE { e = lp;  lp = rp;  rp = e; }

TN *cons_expr(opcode, lp, rp)
     enum tree_opcode opcode;
     TN *lp, *rp;
{
    TN *e, *e1;
    enum datatype etype;
    enum datatype ltype, rtype;
    enum tnode_type ltag, rtag;
    TN *fold();
    
    ltype = TN_TYPE(lp);
    ltag = TN_TAG(lp);
    if(rp && opcode != OPCALL && opcode != OPCCALL)
      {
          rtype = TN_TYPE(rp);
          rtag = TN_TAG(rp);
      }
    else  rtype = TYUNKNOWN;
    
    etype = cktype(opcode, ltype, rtype);
    if(etype == TYERROR)  goto error;
    
    switch(opcode)
      {
          /* check for multiplication by 0 and 1 and addition to 0 */
      case OPSTAR:
          if(IS_CONST(lp))  COMMUTE;
              
          if(IS_ICON(rp))
            {
                if(TN_CONST_INT(rp) == 0) goto retright;
                goto mulop;
            }
          break;
          
      case OPSLASH:
      case OPMOD:
          if(ICON_EQ(rp, 0))
            {
                err("attempted division by zero");
                rp = TN_ICON(1);
                break;
            }
          if(opcode == OPMOD)   break;
          
      mulop:
          if(IS_ICON(rp))
            {
                if(TN_CONST_INT(rp) == 1) goto retleft;
                if(TN_CONST_INT(rp) == -1)
                  {
                      return(cons_expr(OPNEG, lp, NULL));
                  }
            }
          
          if(IS_STAROP(lp) && IS_ICON(TN_RIGHT(lp)))
            {
                if(opcode == OPSTAR)
                  e = cons_expr(OPSTAR, TN_RIGHT(lp), rp);
                else  if(IS_ICON(rp) &&
                         (TN_CONST_INT(TN_RIGHT(lp))%TN_CONST_INT(rp)) == 0)
                  e = cons_expr(OPSLASH, TN_RIGHT(lp), rp);
                else  break;
                
                e1 = TN_LEFT(lp);
                return(cons_expr(OPSTAR, e1, e));
            }
          break;
          
      case OPPLUS:
          if(IS_CONST(lp))  COMMUTE;
          goto addop;
          
      case OPMINUS:
          if(ICON_EQ(lp, 0))
            {
                return(cons_expr(OPNEG, rp, NULL));
            }
          if(IS_CONST(rp))
            {
                opcode = OPPLUS;
                const_negop(rp);
            }
          
      addop:
          if(IS_ICON(rp))
            {
                if(TN_CONST_INT(rp) == 0) goto retleft;
                if(TN_TAG(lp) == T_EXPR && TN_OPCODE(lp) == OPPLUS 
                   && IS_ICON(TN_RIGHT(lp)))
                  {
                      e = cons_expr(OPPLUS, TN_RIGHT(lp), rp);
                      e1 = TN_LEFT(lp);
                      return(cons_expr(OPPLUS, e1, e));
                  }
            }
          break;
          
      case OPPOWER:
          break;
          
      case OPNEG:
          if(ltag == T_EXPR && TN_OPCODE(lp) == OPNEG)
            {
                e = TN_LEFT(lp);
                return(e);
            }
          break;
          
      case OPNOT:
          if(ltag == T_EXPR && TN_OPCODE(lp) == OPNOT)
            {
                e = TN_LEFT(lp);
                return(e);
            }
          break;
          
      case OPCALL:
      case OPCCALL:
          etype = ltype;
          if(rp != NULL && TN_LIST(rp) == NULL)
            {
                rp = NULL;
            }
          break;
          
      case OPAND:
      case OPOR:
          /* logical OR/AND */
          if(IS_CONST(lp)) COMMUTE;
              
          if(IS_CONST(rp))
            {
                if(TN_CONST_INT(rp) == 0)
                  {
                      /* FALSE */
                      if(opcode == OPOR)
                        goto retleft;
                      else
                        goto retright;
                  }
                else 
                  {
                      if(opcode == OPOR)
                        goto retright;
                      else
                        goto retleft;
                  }
            }
      case OPEQV:
      case OPNEQV:
          
      case OPBITAND:
      case OPBITOR:
      case OPBITXOR:
      case OPBITNOT:
      case OPLSHIFT:
      case OPRSHIFT:
          
      case OPLT:
      case OPGT:
      case OPLE:
      case OPGE:
      case OPEQ:
      case OPNE:
          
      case OPCONCAT:
          break;
      case OPMIN:
      case OPMAX:
          
      case OPASSIGN:
      case OPPLUSEQ:
      case OPSTAREQ:
          
      case OPCONV:
      case OPADDR:
          
      case OPCOMMA:
      case OPQUEST:
      case OPCOLON:
          break;

#ifdef HAVE_BL
      case OPMKBL:
          break;
#endif
          
      default:
          fatal("badop, cons_expr", opcode);
      }
    
    e = TN_ALLOC(TN_EXPR);
    TN_TAG(e) = T_EXPR;
    TN_OPCODE(e) = opcode;
    TN_TYPE(e) = etype;
    TN_LEFT(e) = lp;
    TN_RIGHT(e) = rp;

    if(ltag == T_CONST && (rp == 0 || rtag == T_CONST))
      e = fold(e);
    return(e);
    
 retleft:
    return(lp);
    
 retright:
    return(rp);
    
 error:
    return(TN_ERROR);
}


char *powint[ ] = { "pow_ii", "pow_ri", "pow_di", "pow_ci", "pow_zi" };

TN *make_power(p)
     TN *p;
{
    TN *q, *lp, *rp;
    enum datatype ltype, rtype, mtype;
    
    lp = TN_LEFT(p);
    rp = TN_RIGHT(p);
    ltype = TN_TYPE(lp);
    rtype = TN_TYPE(rp);
    
    if(IS_ICON(rp))
      {
          if(TN_CONST_INT(rp) == 0)
            {
                if(IS_INT(ltype))
                  return(TN_ICON(1));
                else
                  return(putconst(make_conv(ltype,TN_ICON(1))));
            }

          if(TN_CONST_INT(rp) < 0)
            {
                if(IS_INT(ltype))
                  {
                      err("integer**negative");
                      return(TN_ERROR);
                  }
                TN_CONST_INT(rp) = - TN_CONST_INT(rp);
                TN_LEFT(p) = lp = fixexpr(cons_expr(OPSLASH, TN_ICON(1),lp));
            }
          if(TN_CONST_INT(rp) == 1)
            {
                return(lp);
            }
          
          if(IS_INT(ltype)||IS_REAL(ltype))
            {
                TN_TYPE(p) = ltype;
                return(p);
            }
      }

    if(IS_INT(rtype))
      {
          if(ltype == TYSHORT && rtype==TYSHORT 
             && (!IS_CONST(lp) || tyint == TYSHORT))
            q = call2(TYSHORT, "pow_hh", lp, rp);
          else  
            {
                if(ltype == TYSHORT)
                  {
                      ltype = TYLONG;
                      lp = make_conv(TYLONG,lp);
                }
                q = call2(ltype, powint[(int)ltype-(int)TYLONG], 
                          lp, make_conv(TYLONG, rp));
            }
      }
    else if(IS_REAL((mtype = maxtype(ltype,rtype))))
      q = call2(mtype, "pow_dd", 
                make_conv(TYDREAL,lp), make_conv(TYDREAL,rp));
    else        
      {
          q  = call2(TYDCOMPLEX, "pow_zz",
                     make_conv(TYDCOMPLEX,lp), make_conv(TYDCOMPLEX,rp));
          if(mtype == TYCOMPLEX)
            q = make_conv(TYCOMPLEX, q);
      }
    return(q);
}




