package exc.openmp;

//import xcodeml.util.XmOption;
import exc.object.*;
import exc.block.*;
import exc.object.*;

/**
 * pass2: check and write variables
 */
public class OMPDDRDrewriteExpr
{
    public OMPDDRDrewriteExpr()
    {
    }

    public XobjList getAllArrays(Xobject obj) {
	XobjList objs = new XobjList();
	if (obj.Opcode() == Xcode.F_ARRAY_REF) {
	    objs.add(obj);
	    XobjList tmps;
	    tmps = getAllArrays(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	} else if (obj.Opcode() == Xcode.MEMBER_REF ||
		   obj.Opcode() == Xcode.F_VAR_REF) {
	    XobjList tmps;
	    tmps = getAllArrays(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	} else if (obj.Opcode() == Xcode.PLUS_EXPR) {
	    XobjList tmps;
	    tmps = getAllArrays(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	    tmps = getAllArrays(obj.getArg(1));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	}
	return objs;
    }

    public XobjList getAllVarsForWriteStatements(Xobject obj) {
	XobjList objs = new XobjList();
	if (obj.Opcode() == Xcode.F_ARRAY_REF) {
	    objs.add(obj);
	    XobjList tmps;
	    tmps = getAllVarsForWriteStatements(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	} else if (obj.Opcode() == Xcode.MEMBER_REF ||
		   obj.Opcode() == Xcode.F_VAR_REF) {
	    objs.add(obj);
	    XobjList tmps;
	    tmps = getAllVarsForWriteStatements(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	} else if (obj.Opcode() == Xcode.PLUS_EXPR) {
	    XobjList tmps;
	    tmps = getAllVarsForWriteStatements(obj.getArg(0));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	    tmps = getAllVarsForWriteStatements(obj.getArg(1));
	    for (Xobject tmp : tmps) {
		objs.add(tmp);
	    }
	} else if (obj.Opcode() == Xcode.VAR) {
	    objs.add(obj);
	}
	return objs;
    }

    public Xobject updateArrayByDDRD(Xobject x, Block b, String kind, boolean assumed_shape, Xobject input)
    {
        OMPvar v;
        XobjectIterator i = new topdownXobjectIterator(x);
	XobjList args;

	for(i.init(); !i.end(); i.next()) {
            Xobject xx = i.getXobject();
	    if (xx != null) {
		if(xx.Opcode() == Xcode.VAR) {
		    v = OMPinfo.refOMPvar(b, xx.getName());
		    if (v.is_shared) {
			if (xx.Opcode() == Xcode.VAR &&
			    xx.Type().isFarray() &&
			    i.getParent().Opcode() == Xcode.F_VAR_REF) {
			    Xobject varShare = v.getSharedAddr();
			    Xtype ts = varShare.Type();
			    int motolen = ts.getFarraySizeExpr().length;
			    Xobject sizeExprs[] = new Xobject[motolen+1];
			    for (int k=0; k < motolen; k++) {
				sizeExprs[k] = ts.getFarraySizeExpr()[k];
			    }
			    sizeExprs[motolen] = Xcons.FindexRange(Xcons.IntConstant(0),
								   Xcons.IntConstant(OMPDDRD.DDRD_NUM_THREADS-1));
			    Xtype sizeArrayType = Xtype.Farray(Xtype.FintType, sizeExprs);
			    Ident id = Ident.Fident("ddrd"+kind+"_"+xx.getName(), sizeArrayType, false, true, null);
			    i.setXobject(id.Ref());
			}
		    }
		} else if(xx.Opcode() == Xcode.F_ARRAY_REF) {
		    XobjList mmm = new XobjList(Xcode.LIST, xx.getArg(1).getArgs());
		    if (assumed_shape) {
			int tmp = mmm.Nargs();
			for (int m = 0; m < tmp; m++) {
			    mmm.setArg(m,Xcons.List(Xcode.F_ARRAY_INDEX, input));
			}
		    }
		    mmm.add(Xcons.List(Xcode.F_ARRAY_INDEX, input));
		}
	    }
        }
        return i.topXobject();
    }

    public void run(FuncDefBlock def, OMPfileEnv env)
    {
        OMP.debug("pass2:");
        FunctionBlock fb = def.getBlock();

	if (fb == null) return;

        int threads = 32;
	Xobject fixedSizeExprs[] = new Xobject[2];
	fixedSizeExprs[0] = Xcons.FindexRange(Xcons.IntConstant(0),
					      Xcons.IntConstant(threads-1));
	fixedSizeExprs[1] = Xcons.FindexRange(Xcons.IntConstant(0),
					      Xcons.IntConstant(threads-1));
	Xtype fixedSizeArrayType = Xtype.Farray(Xtype.FintType, fixedSizeExprs);
	Ident myclock = Ident.FidentNotExternal("ddrdclock", fixedSizeArrayType);
	Ident mylock = Ident.FidentNotExternal("ddrdlock", Xtype.FintType);
	Ident omp_get_thread_num = Ident.FidentNotExternal("omp_get_thread_num()", Xtype.FintFunctionType);

	BlockIterator iter5 = new topdownBlockIterator(fb);
	for (iter5.init(); !iter5.end(); iter5.next()) {
	    Block b5 = iter5.getBlock();
	    switch (b5.Opcode()) {
	    case OMP_PRAGMA: {
		PragmaBlock pb = (PragmaBlock)b5;
//		    System.out.println("pragma "+pb);
		OMPinfo i = (OMPinfo)pb.getProp(OMP.prop);
		switch(i.pragma) {
		case PARALLEL: {
		    BasicBlockIterator iter6 = new BasicBlockIterator(pb);
		    for (iter6.init(); !iter6.end(); iter6.next()){
			Block b = iter6.getBasicBlock().getParent();
			StatementIterator iter7 = iter6.getBasicBlock().statements();
			while (iter7.hasNext()){
			    Statement st = iter7.next();
			    Xobject x = st.getExpr();
			    if (x == null)  continue;
			    switch (x.Opcode()) {
			    case F_ASSIGN_STATEMENT: {
				Ident omp_set_lock_name = Ident.FidentNotExternal("omp_set_lock", Xtype.FsubroutineType);
				st.insert(omp_set_lock_name.callSubroutine(Xcons.List(mylock.Ref())));

				/* write mark to array */
				XobjList lhs_objs = getAllArrays(x.getArg(0));
				for (Xobject tmp0 : lhs_objs) {
				    if (tmp0.Opcode() == Xcode.F_ARRAY_REF) {
					Xobject tmp6 = tmp0.copy();
					Xobject tmp7 = tmp0.copy();
					updateArrayByDDRD(tmp6,b,"wr",true,Xcons.FindexRangeOfAssumedShape());
					updateArrayByDDRD(tmp7,b,"rd",true,Xcons.FindexRangeOfAssumedShape());
					pb.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp6,
							     Xcons.IntConstant(0)));
					pb.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp7,
							     Xcons.IntConstant(0)));

					Xobject condExpr = Xcons.FlogicalConstant(false);
					Xobject write_st = Xcons.List(Xcode.F_STOP_STATEMENT);
					for(int k = 0; k < OMPDDRD.DDRD_NUM_THREADS; k++) {
					    Xobject tmp_if_rd = tmp0.copy();
					    Xobject tmp_if_wr = tmp0.copy();
					    Xobject tmp_clock = Xcons.FarrayRef(myclock.Ref(),
										omp_get_thread_num,
										Xcons.IntConstant(k));
					    updateArrayByDDRD(tmp_if_rd,b,"rd",false,Xcons.IntConstant(k));
					    updateArrayByDDRD(tmp_if_wr,b,"wr",false,Xcons.IntConstant(k));

					    condExpr = Xcons.binaryOp(Xcode.LOG_OR_EXPR,
								      condExpr,
								      Xcons.binaryOp(Xcode.LOG_OR_EXPR,
										     Xcons.binaryOp(Xcode.LOG_LT_EXPR,
												    tmp_clock,
												    tmp_if_rd
												    ),
										     Xcons.binaryOp(Xcode.LOG_LT_EXPR,
												    tmp_clock,
												    tmp_if_wr
												    )
										     )
								      );
					    //System.out.println(getAllVarsForWriteStatements(tmp0));
					    write_st = Xcons.FstatementList
						(Xcons.List(Xcode.F_WRITE_STATEMENT,
							    Xcons.List(Xcode.F_NAMED_VALUE_LIST,
								       Xcons.List(Xcode.F_NAMED_VALUE,
										  Xcons.String("unit"),
										  Xcons.String("*")
										  ),
								       Xcons.List(Xcode.F_NAMED_VALUE,
										  Xcons.String("fmt"),
										  Xcons.String("*")
										  )
								       ),
							    Xcons.List(Xcode.F_VALUE_LIST,
								       Xcons.List(Xcode.F_VALUE,
										  Xcons.IntConstant(b.getLineNo().lineNo())
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  Xcons.FcharacterConstant(Xtype.FcharacterType, tmp0.getName(), null)
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  tmp_clock
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  tmp_if_rd
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  tmp_if_wr
										  )
								       )
							    ),
						 write_st);
					}
					XobjList ifBlock = Xcons.List(Xcode.F_IF_STATEMENT,
								      (Xtype)null,
								      (Xobject)null,
								      condExpr,     // IF condition
								      //Xcons.List(Xcode.F_STOP_STATEMENT),    // THEN block
								      write_st,
								      null);                 // ELSE block
					st.insert(ifBlock);

					Xobject tmp1 = tmp0.copy();
					updateArrayByDDRD(tmp1,b,"wr",false,omp_get_thread_num);
					st.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp1,
							     Xcons.FarrayRef(myclock.Ref(),
									     omp_get_thread_num,
									     omp_get_thread_num)
							     ));
				    }
				}

				/* read mark to array */
				XobjList rhs_objs = getAllArrays(x.getArg(1));
				for (Xobject tmp0 : rhs_objs) {
				    if (tmp0.Opcode() == Xcode.F_ARRAY_REF) {
					Xobject tmp8 = tmp0.copy();
					Xobject tmp9 = tmp0.copy();
					updateArrayByDDRD(tmp8,b,"wr",true,Xcons.FindexRangeOfAssumedShape());
					updateArrayByDDRD(tmp9,b,"rd",true,Xcons.FindexRangeOfAssumedShape());
					pb.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp8,
							     Xcons.IntConstant(0)));
					pb.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp9,
							     Xcons.IntConstant(0)));

					Xobject condExpr = Xcons.FlogicalConstant(false);
					Xobject write_st = Xcons.List(Xcode.F_STOP_STATEMENT);
					for(int k = 0; k < OMPDDRD.DDRD_NUM_THREADS; k++) {
					    Xobject tmp_if_wr = tmp0.copy();
					    Xobject tmp_clock = Xcons.FarrayRef(myclock.Ref(),
										omp_get_thread_num,
										Xcons.IntConstant(k));
					    updateArrayByDDRD(tmp_if_wr,b,"wr",false,Xcons.IntConstant(k));
					    condExpr = Xcons.binaryOp(Xcode.LOG_OR_EXPR,
								      condExpr,
								      Xcons.binaryOp(Xcode.LOG_LT_EXPR,
										     tmp_clock,
										     tmp_if_wr
										     )
								      );
					    write_st = Xcons.FstatementList
						(Xcons.List(Xcode.F_WRITE_STATEMENT,
							    Xcons.List(Xcode.F_NAMED_VALUE_LIST,
								       Xcons.List(Xcode.F_NAMED_VALUE,
										  Xcons.String("unit"),
										  Xcons.String("*")
										  ),
								       Xcons.List(Xcode.F_NAMED_VALUE,
										  Xcons.String("fmt"),
										  Xcons.String("*")
										  )
								       ),
							    Xcons.List(Xcode.F_VALUE_LIST,
								       Xcons.List(Xcode.F_VALUE,
										  Xcons.IntConstant(b.getLineNo().lineNo())
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  Xcons.FcharacterConstant(Xtype.FcharacterType, tmp0.getName(), null)
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  tmp_clock
										  ),
								       Xcons.List(Xcode.F_VALUE,
										  tmp_if_wr
										  )
								       )
							    ),
						 write_st);
					}
					XobjList ifBlock = Xcons.List(Xcode.F_IF_STATEMENT,
								      (Xtype)null,
								      (Xobject)null,
								      condExpr,     // IF condition
								      //Xcons.List(Xcode.F_STOP_STATEMENT),
								      write_st,  // THEN block
								      null);                 // ELSE block
					st.insert(ifBlock);

					Xobject tmp2 = tmp0.copy();
					updateArrayByDDRD(tmp2,b,"rd",false,omp_get_thread_num);
					st.insert(Xcons.List(Xcode.F_ASSIGN_STATEMENT,
							     tmp2,
							     Xcons.FarrayRef(myclock.Ref(),
									     omp_get_thread_num,
									     omp_get_thread_num)
							     ));
				    }
				}

				Ident omp_unset_lock_name = Ident.FidentNotExternal("omp_unset_lock", Xtype.FsubroutineType);
				st.add(omp_unset_lock_name.callSubroutine(Xcons.List(mylock.Ref())));

				break;
			    }
			    case F_WRITE_STATEMENT: {
				break;
			    }
			    default: {
				System.out.println("Unsupported");
				System.exit(255);
			    }
			    }
			}
		    }

		    for(int k = 0; k < OMPDDRD.DDRD_NUM_THREADS; k++) {
			for(int l = 0; l < OMPDDRD.DDRD_NUM_THREADS; l++) {
			    XobjList st = Xcons.List(Xcode.F_ASSIGN_STATEMENT,
						     Xcons.FarrayRef(myclock.Ref(),
								     Xcons.IntConstant(k),
								     Xcons.IntConstant(l)),
						     (k==l ? Xcons.IntConstant(1) : Xcons.IntConstant(0))
						     );
			    pb.insert(st);
			}
		    }

		    break;
		}
		}
		break;
	    }
	    }
	}
    }
}
