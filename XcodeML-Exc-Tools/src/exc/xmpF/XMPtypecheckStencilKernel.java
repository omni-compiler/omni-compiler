package exc.xmpF;

import java.io.Serializable;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.Reader;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Vector;

import exc.object.*;
import exc.block.*;
import exc.xcodeml.*;

import exc.xmpF.XMPotype;


/**
 * typecheck XcalableMP pragma
 * optional pass: typecheck
 */
public class XMPtypecheckStencilKernel
{
    XMPenv env;

    public XMPtypecheckStencilKernel() {}

    public void run(FuncDefBlock def, XMPenv env) {
	this.env = env;
	env.setCurrentDef(def);

	HashMap<String,XMPotype> funcTable = new HashMap<String,XMPotype>();
	File dir = new File(".");
	File[] list = dir.listFiles();
	for(int i=0; i<list.length; i++) {
	    String filename = list[i].getName();
	    if(filename.contains(".xi")) {
		try (ObjectInputStream objIn = new ObjectInputStream(new FileInputStream(filename))) {
		    funcTable.put(filename,(XMPotype)objIn.readObject());
		} catch (ClassNotFoundException e) {
		    e.printStackTrace();
		} catch (IOException e) {
		    e.printStackTrace();
		} catch (Exception e) {
		    e.printStackTrace();
		}
	    }
	}

	if(env.getStencilTypecheckFlag()) {
	    if (!(env.getStencilTypecheckQuietFlag())) {
		System.out.println("\n=== typecheck begin ===\n");
	    }
	    Block fblock = def.getBlock();
	    String ident = def.getBlock().getName()+".xi";
	    DDirectives ddctv = new DDirectives(new XobjList(),
						new HashMap<String,XobjList>(),
						new HashMap<String,XobjList>(),
						new HashMap<String,XobjList>(),
						new HashMap<String,String>(),
						new HashMap<Pair<String,Integer>,Pair<String,Integer>>()
						);
	    ddctv = importXMPdirectives(ddctv,fblock.getBody().getDecls());
	    XMPotype otype = new XMPotype(new HashSet<String>(),
				    new HashSet<String>(),
				    new HashSet<String>(),
				    new String(),
				    new HashMap<String,XobjList>(),
				    //
				    new HashSet<String>(),
				    new HashSet<String>()
				    );
	    otype = typecheckBlock(funcTable,ddctv,fblock);
	    if (!(env.getStencilTypecheckQuietFlag())) {
		System.out.println("\n=== typecheck end ===\n");
	    }
	    try (ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(ident))) {
		otype.expander = new HashMap<String,XobjList>();
		objOut.writeObject(otype);
		objOut.flush();
		objOut.reset();
	    } catch (FileNotFoundException e) {
		e.printStackTrace();
	    } catch (IOException e) {
		e.printStackTrace();
	    }
	}
	return ;
    }

    // check use and import if use of module is found.
    private DDirectives importXMPdirectives(DDirectives ddctv, Xobject decls){
	if (!(decls == null)) {
	    for (Xobject decl: (XobjList)decls) {
		if (decl.Opcode() != Xcode.F_USE_DECL) continue;
		String module_name = decl.getArg(0).getName();
		if (XMP.debugFlag) System.out.println("module read begin: "+module_name);
		String mod_file_name = module_name+".xmod";
		Reader reader = null;
		String mod_file_name_with_path = "";
		boolean found = false;
		File mod_file;
		for (String spath: XcodeMLtools_Fmod.getSearchPath()) {
		    mod_file_name_with_path = spath + "/" + mod_file_name;
		    mod_file = new File(mod_file_name_with_path);
		    if (mod_file.exists()){
			found = true;
			break;
		    }
		}
		if (!found){
		    mod_file_name_with_path = mod_file_name;
		    mod_file = new File(mod_file_name_with_path);
		    if (mod_file.exists()){
			found = true;
		    }
		}
		try {
		    reader = new BufferedReader(new FileReader(mod_file_name_with_path));
		} catch(Exception e){
		    XMP.error("cannot open module file '"+mod_file_name+"'");
		    return ddctv;
		}
		XcodeMLtools_Fmod tools = new XcodeMLtools_Fmod();
		tools.read(reader);
		Vector<Xobject> aux_info = tools.getAuxInfo();
		for (Xobject x: aux_info) {
		    if(x.Opcode() != Xcode.XMP_PRAGMA) continue;
		    XMPpragma pragma = XMPpragma.valueOf(x.getArg(0));
		    analyzeDeclarativePragma(ddctv,pragma,x.getArg(1));
		}
	    }
	}
	return ddctv;
    }

    private XMPotype typecheckBlock(HashMap<String,XMPotype> funcTable, DDirectives ddctv, Block fblock) {
	XMPotype otype = new XMPotype(new HashSet<String>(),
				new HashSet<String>(),
				new HashSet<String>(),
				new String(),
				new HashMap<String,XobjList>(),
				//
				new HashSet<String>(),
				new HashSet<String>()
				);
	BlockIterator i = new topdownBlockIterator(fblock);
	for(i.init(); !i.end(); i.next()) {
	    Block b = i.getBlock();
	    if(XMP.debugFlag) System.out.println("optional pass=" + b);
	    if (b.Opcode().equals(Xcode.XMP_PRAGMA)){
		XMPpragma p = ((XMPinfo)b.getProp(XMP.prop)).pragma;
		PragmaBlock pb = (PragmaBlock)b;
		switch (p) {
		case NODES:
		case TEMPLATE:
		case DISTRIBUTE:
		case ALIGN:
		case SHADOW:
		    ddctv = analyzeDeclarativePragma(ddctv,p,pb.getClauses());
		    break;
		case REFLECT:
		case LOOP:
		    otype = analyzePragma(b.getLineNo(),ddctv,otype,p,pb);
		    break;
		default:
		    System.out.println((char)27+"[1;34m"+"fatal error: "+p+" is unsupported"+(char)27+"[0m");
		    if (!(env.getStencilTypecheckallFlag())) {
			System.exit(63);
		    }
		}
	    } else {
		// typecheck function call
		for (Xobject x : (XobjList)b.toXobject()) {
		    if (!(x == null)) {
			switch (x.Opcode()) {
			case EXPR_STATEMENT:
			    for (Xobject y : (XobjList)x) {
				if (!(y == null)) {
				    switch (y.Opcode()) {
				    case FUNCTION_CALL:
					for (String z : funcTable.keySet()) {
					    if (z.equals(y+".xi")) {
						for (String l : otype.wr_a) {
						    for (String m : funcTable.get(z).first_rd_a) {
							if (l.equals(m)) {
							    System.out.println((char)27+"[1;31m"+"  "+b.getLineNo()+" type error: missing reflect "+l+(char)27+"[0m");
							    if (!(env.getStencilTypecheckallFlag())) {
								System.exit(32);
							    }
							}
						    }
						}
						otype.wr_a.removeAll(funcTable.get(z).reflected);
						otype.wr_a.addAll(funcTable.get(z).wr_a);
					    }
					}
					break;
				    default:
					;
				    }
				}
			    }
			    break;
			default:
			    ;
			}
		    }
		}
	    }
	}
	return otype;
    }

    private DDirectives analyzeDeclarativePragma(DDirectives ddctv, XMPpragma p, Xobject x) {
	switch (p){
	case NODES:
	    ddctv.nodes = (XobjList)x.getArg(1);
	    break;
	case TEMPLATE:
	    ddctv.template.put(x.getArg(0).getArg(0).getName(),(XobjList)x.getArg(1));
	    break;
	case DISTRIBUTE:
	    ddctv.distribute.put(x.getArg(0).getArg(0).getName(),(XobjList)x.getArg(1));
	    break;
	case ALIGN:
	    for (Xobject y : (XobjList)x.getArg(0)) {
		ddctv.aligns_simple.put(y.getName(),x.getArg(2).getName());
		int j = 0;
		for (Xobject z : (XobjList)x.getArg(1)) {
		    if (!(z == null)) {
			int i = 0;
			if (x.getArg(3).getArg(i) == null) {
			    i++;
			} else {
			    while (!(z.getName().equals(x.getArg(3).getArg(i).getName()))) {
				i++;
			    }

			}
			ddctv.aligns.put(new Pair<String,Integer>(y.getName(),j),new Pair<String,Integer>(x.getArg(2).getName(),i));
		    }
		    j++;
		}
	    }
	    break;
	case SHADOW:
	    for (Xobject y : (XobjList)x.getArg(0)) {
		ddctv.shadows.put(y.getName(),(XobjList)x.getArg(1));
	    }
	    break;
	}
	return ddctv;
    }

    private XMPotype analyzePragma(LineNo ln, DDirectives ddctv, XMPotype otype, XMPpragma p, PragmaBlock pb) {
	Xobject x = pb.getClauses();
	switch (p){
	case REFLECT:
	    for (Xobject y : (XobjList)x.getArg(0)) {
		otype.wr_a.remove(y.getName());
		if (otype.first_rd_a.isEmpty()) {
		    otype.reflected.add(y.getName());
		}
	    }
	    break;
	case LOOP:
	    {
		otype.rd_a = new HashSet<String>();
		otype.aligned = x.getArg(1).getArg(0).getName();
		if (!(x.getArg(2) == null)) {
		    if (x.getArg(2).getArg(0).getInt() == XMP.LOOP_EXPAND) {
			for (Xobject y : (XobjList)x.getArg(1).getArg(1)) {
			    otype.expander.put(y.getName(),(XobjList)x.getArg(2).getArg(1).getArg(0));
			}
		    }
		}
		otype = typecheckLoop(pb,otype,ddctv);
		if (otype.first_rd_a.isEmpty()) {
		    otype.first_rd_a.addAll(otype.rd_a);
		    otype.first_rd_a.removeAll(otype.reflected);
		}
	    }
	    break;
	}
	return otype;
    }

    private XMPotype typecheckLoop(PragmaBlock pb, XMPotype otype, DDirectives ddctv) {
	XMPinfo info = (XMPinfo)pb.getProp(XMP.prop);
	for(int k = 0; k < info.getLoopDim(); k++){
	    XMPdimInfo d_info = info.getLoopDimInfo(k);
	    Xobject loop_var = d_info.getLoopVar();
	    ForBlock for_block = d_info.getLoopBlock();
	    FdoBlock do_block = (FdoBlock)for_block;
	    HashSet<String> origin = new HashSet<String>();
	    for (Xobject y : (XobjList)pb.getClauses().getArg(0)) {
		origin.add(y.getName());
	    }
	    topdownBlockIterator iter = new topdownBlockIterator(do_block);
	    for (iter.init(); !iter.end(); iter.next()) {
		Block b = iter.getBlock();
		if (b != null) {
		    switch(b.Opcode()) {
		    case F_STATEMENT_LIST:
			if (!(env.getStencilTypecheckQuietFlag())) {
			    System.out.println(b.getLineNo()+" enter to typecheck "+b.Opcode()+" ");
			}
			BasicBlock bb = b.getBasicBlock();
			StatementIterator si = bb.statements();
			while (si.hasNext()) {
			    Statement st = si.next();
			    Xobject assign = st.getExpr();
			    switch (assign.Opcode()) {
			    case F_ASSIGN_STATEMENT:
				if (!(env.getStencilTypecheckQuietFlag())) {
				    System.out.println(" "+assign.getLineNo()+" enter to typecheck "+assign.Opcode()+" ");
				}
				otype = typecheckAssignment(loop_var,assign.getLineNo(),origin,assign,otype,ddctv);
				if (!(env.getStencilTypecheckQuietFlag())) {
				    System.out.println(" "+assign.getLineNo()+" exit to typecheck "+assign.Opcode()+" ");
				}
				break;
			    default:
				if (!(env.getStencilTypecheckQuietFlag())) {
				    System.out.println(assign.getLineNo()+" skip to typecheck "+assign+" ");
				}
				break;
			    }
			}
			if (!(env.getStencilTypecheckQuietFlag())) {
			    System.out.println(b.getLineNo()+" exit from to typecheck "+b.Opcode()+" ");
			}
			break;
		    default:
			if (!(env.getStencilTypecheckQuietFlag())) {
			    System.out.println(b.getLineNo()+" skip to typecheck "+b.Opcode()+" ");
			}
			break;
		    }
		}
	    }
	}
	return otype;
    }

    private XMPotype typecheckAssignment(Xobject loop_var, LineNo ln, HashSet<String> origin, Xobject assign, XMPotype otype, DDirectives ddctv) {
	HashSet<String> write_array_names = otype.wr_a;
	HashSet<String> read_array_names = otype.rd_a;
	HashSet<String> write_var_names = otype.wr_v;

	switch (assign.getArg(0).Opcode()) {
	case F_ARRAY_REF:
	    isAlignedTheSameTemplate(ln,assign.getArg(0),otype,ddctv);
	    if (isOnTheNode(origin,assign.getArg(0),ddctv,otype.aligned)) {
		if (!(env.getStencilTypecheckQuietFlag())) {
		    System.out.println("  "+ln+" in the sub-language: "+assign.getArg(0));
		}
		write_array_names.add(assign.getArg(0).getName());
	    } else {
		System.out.println((char)27+"[1;34m"+"  "+ln+" out of the sub-language: "+assign.getArg(0).getName()+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(15);
		}
	    }
	    break;
	case VAR:
	    if (!(env.getStencilTypecheckQuietFlag())) {
		System.out.println("  "+ln+" in the sub-language (local variable condition): "+assign.getArg(0).getName());
	    }
	    write_var_names.add(assign.getArg(0).getName());
	    break;
	default:
	    System.out.println((char)27+"[1;34m"+"  "+ln+" fatal error"+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(63);
		}
	    break;
	}

	XobjectIterator i = new topdownXobjectIterator(assign.getArg(1));
	HashSet<String> tmp = new HashSet<String>(); // avoid to misunderstand array as variable
	HashSet<String> non_aligned_vars_in_arrays = new HashSet<String>();
	for(i.init(); !i.end(); i.next()) {
	    Xobject x = i.getXobject();
	    switch (x.Opcode()) {
	    case F_ARRAY_REF:
		isAlignedTheSameTemplate(ln,x.getArg(0),otype,ddctv);
		if (isDistributedArray(ddctv,x.getArg(0).getName())) {
		    if (write_array_names.contains(x.getName()) && (!(isOnTheNode(origin,x,ddctv,otype.aligned)))) {
			System.out.println((char)27+"[1;31m"+"  "+ln+" type error: overlapped read/write arrays or missing reflect "+x.getName()+(char)27+"[0m");
			if (!(env.getStencilTypecheckallFlag())) {
			    System.exit(31);
			}
		    } else {
			if (typecheckMissingReflect(origin,x,ddctv,otype)) {
			    if (!(env.getStencilTypecheckQuietFlag())) {
				System.out.println("  "+ln+" read/write arrays are non-overlapped and no reflect is missing");
			    }
			} else {
			    System.out.println((char)27+"[1;31m"+"  "+ln+" type error: missing reflect "+x+(char)27+"[0m");
			    if (!(env.getStencilTypecheckallFlag())) {
				System.exit(32);
			    }
			}
		    }
		    if (!(isOnTheNode(origin,x,ddctv,otype.aligned))) {
			read_array_names.add(x.getName());
		    }
		}
		tmp.add(x.getName());

		//		System.out.println(x.getArg(0).getName());
		int n = x.getArg(0).getArg(0).Type().getNumDimensions();
		for (int k = 0; k < n; k++) {
		    int j = 0;
		    Xobject y = x.getArg(1).getArg(k).getArg(0);
		    switch (y.Opcode()) {
		    case VAR:
			if (!(origin.contains(y.getName()))) {
			    //			    System.out.println(y.getName());
			    non_aligned_vars_in_arrays.add(y.getName());
			}
			break;
		    case PLUS_EXPR:
		    case MINUS_EXPR:
			if (!(origin.contains(y.getArg(0).getName()))) {
			    //			    System.out.println(y.getArg(0).getName());
			    non_aligned_vars_in_arrays.add(y.getArg(0).getName());
			}
			break;
		    default:
			break;
		    }
		}
		break;
	    case VAR:
		if (write_var_names.contains(x.getName()) ||
		    tmp.contains(x.getName()) || // avoid to misunderstand array as variable
		    origin.contains(x.getName())
		    ) {
		    if (!(env.getStencilTypecheckQuietFlag())) {
			System.out.println("  "+ln+" in the sub-language (local variable condition): "+x);
		    }
		} else {
		    if ((!(x.Type().isFparameter())) &&
			(!(non_aligned_vars_in_arrays.contains(x.getName())))
			) {
			System.out.println((char)27+"[1;33m"+"  "+ln+" warning: check that "+x.getName()+" is invariant to threads by yourself"+(char)27+"[0m");
			if (!(env.getStencilTypecheckallFlag())) {
			    System.exit(47);
			}
		    }
		}
		break;
	    case FUNCTION_CALL:
		if (x.getArg(0).getName().equals("float") ||
		    x.getArg(0).getName().equals("dble") ||
		    x.getArg(0).getName().equals("cos") ||
		    x.getArg(0).getName().equals("sin") ||
		    x.getArg(0).getName().equals("min") ||
		    x.getArg(0).getName().equals("max") ||
		    x.getArg(0).getName().equals("abs") ||
		    x.getArg(0).getName().equals("sqrt")
		    ){
		} else {
		    System.out.println((char)27+"[1;34m"+"  "+ln+" fatal error: "+x+(char)27+"[0m");
		    if (!(env.getStencilTypecheckallFlag())) {
			System.exit(63);
		    }
		}
	    case F_ARRAY_INDEX: // included by cases of F_ARRAY_REF
	    case F_VAR_REF:     // included by cases of F_ARRAY_REF
	    case LIST:          // included by cases of F_ARRAY_REF or VAR
	    case IDENT:         // included by cases of FUNCTON_CALL
		//
	    case PLUS_EXPR:
	    case MINUS_EXPR:
	    case MUL_EXPR:
	    case DIV_EXPR:
	    case F_POWER_EXPR:
	    case UNARY_MINUS_EXPR:
		//
	    case ARRAY_ADDR:
	    case FUNC_ADDR:
	    case VAR_ADDR:
	    case INT_CONSTANT:
	    case STRING_CONSTANT:
	    case LONGLONG_CONSTANT:
	    case FLOAT_CONSTANT:
	    case MOE_CONSTANT:
	    case SIZE_OF_EXPR:
	    case GCC_ALIGN_OF_EXPR:
		break;
	    default:
		System.out.println((char)27+"[1;34m"+"  "+ln+" fatal error: "+x+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(63);
		}
	    }
	}
	return new XMPotype(write_array_names,
			 read_array_names,
			 write_var_names,
			 otype.aligned,
			 otype.expander,
			 //
			 otype.reflected,
			 otype.first_rd_a
			 );
    }

    private boolean isDistributedArray(DDirectives ddctv,String x) {
	if (ddctv.aligns_simple.get(x) == null) {
	    return false;
	}
	return true;
    }

    private boolean isAlignedTheSameTemplate(LineNo ln, Xobject array,XMPotype otype,DDirectives ddctv) {
	if (isDistributedArray(ddctv,array.getName())) {
	    if (!(otype.aligned.equals(ddctv.aligns_simple.get(array.getName())))) {
		System.out.println((char)27+"[1;31m"+" "+ln+" type error: there exists a distinct template for "+array.getName()+" in a loop "+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(34);
		}
		return false;
	    }
	}
	return true;
    }

    private boolean isVarContainedByVertex(HashSet<String> origin, Xobject var) {
	return (origin.contains(var.getName()));
    }

    private boolean isOnTheNode(HashSet<String> origin, Xobject array, DDirectives ddctv, String type_t) {
	int n = array.getArg(0).getArg(0).Type().getNumDimensions();
	for (int i = 0; i < n; i++){
	    int j = 0;
	    Xobject loopv = array.getArg(1).getArg(i);
	    Xobject loopvar = loopv.getArg(0);
	    switch (loopvar.Opcode()) {
	    case VAR:
		if (origin.contains(isDistributeMember(loopv))) {
		    if (!(origin.contains(loopvar.getName()))) {
			return false;
		    }
		}
		break;
	    case ARRAY_ADDR:
	    case FUNC_ADDR:
	    case VAR_ADDR:
	    case INT_CONSTANT:
	    case STRING_CONSTANT:
	    case LONGLONG_CONSTANT:
	    case FLOAT_CONSTANT:
	    case MOE_CONSTANT:
	    case SIZE_OF_EXPR:
	    case GCC_ALIGN_OF_EXPR:
		break;
	    case PLUS_EXPR:
	    case MINUS_EXPR:
		int m = ddctv.nodes.Nargs();
		if (origin.contains(isDistributeMember(loopv))) {
		    if (ddctv.distribute.get(type_t).getArg(j).getArg(0).getName().equals("block")) {
			if (!(ddctv.nodes.getArg(j).getInt() == 1)) {
			    return false;
			}
		    } else if (ddctv.distribute.get(type_t).getArg(j).getArg(0).getName().equals("cyclic")) {
			int tmp = 1;
			if (!(ddctv.distribute.get(type_t).getArg(j).getArg(1) == null)) {
			    tmp = ddctv.distribute.get(type_t).getArg(j).getArg(1).getInt();
			}
			if (loopvar.getArg(1).isIntConstant()) {
			    if (origin.contains(loopvar.getArg(0).getName()) &&
				loopvar.getArg(1).getInt() % (tmp * ddctv.nodes.getArg(j).getInt()) == 0
				) {
				;
			    } else {
				return false;
			    }
			} else {
			    System.out.println("  "+(char)27+"[1;33m"+"warning: check "+
					       loopvar.getArg(1).getString()+
					       " % "+
					       ddctv.nodes.getArg(j).getInt()+
					       " * "+
					       tmp+
					       " == 0"+
					       " by yourself"+(char)27+"[0m");
			}
		    } else {
			System.out.println("  "+(char)27+"[1;34m"+"fatal error: not supported"+(char)27+"[0m");
			System.exit(63);
		    }
		}
		j++;
		break;
	    default:
		System.out.println("  "+(char)27+"[1;34m"+"fatal error: "+loopvar.Opcode()+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(63);
		}
		return false;
	    }
	}
	return true;
    }

    private String isDistributeMember(Xobject x) {
	switch (x.getArg(0).Opcode()) {
	case PLUS_EXPR:
	case MINUS_EXPR:
	    return x.getArg(0).getArg(0).getName();
	case VAR:
	    return x.getArg(0).getName();
	}
	return "";
    }

    private boolean typecheckMissingReflect(HashSet<String> origin, Xobject array,DDirectives ddctv, XMPotype otype) {
	int n = array.getArg(0).getArg(0).Type().getNumDimensions();
	Xobject loopvar = null;
	for (int i = 0; i < n; i++){
	    loopvar = array.getArg(1).getArg(i).getArg(0);
	    switch (loopvar.Opcode()) {
	    case VAR:
	    case INT_CONSTANT:
	    case FLOAT_CONSTANT:
		break;
	    case PLUS_EXPR:
	    case MINUS_EXPR:
		{
		    int m = ddctv.nodes.Nargs();
		    int lower_or_upper = 1;
		    switch (loopvar.Opcode()) {
		    case MINUS_EXPR:
			lower_or_upper = 0;
			break;
		    default:
			break;
		    }
		    for (Xobject x : (XobjList)array.getArg(1)) {
			if (origin.contains(isDistributeMember(x))) {
			    if (loopvar.getArg(1).isIntConstant() &&
				(otype.expander.isEmpty() ? true : otype.expander.get(isDistributeMember(x)).getArg(lower_or_upper).isIntConstant())
				) {
				//				System.out.println("2 "+loopvar);
				if (isDistributedArray(ddctv,loopvar.getArg(1).getName())) {
				    return loopvar.getArg(1).getInt() + (otype.expander.isEmpty() ? 0 : otype.expander.get(isDistributeMember(x)).getArg(lower_or_upper).getInt()) <=
					ddctv.shadows.get(array.getArg(0).getName()).getArg(i).getArg(lower_or_upper).getInt();
				}
			    } else {
				System.out.println("  "+(char)27+"[1;33m"+"warning: check "+
						   (loopvar.getArg(1).isIntConstant() ? loopvar.getArg(1).getInt() : loopvar.getArg(1).getString())+
						   " + "+
						   (otype.expander.isEmpty() ? "0" : otype.expander.get(isDistributeMember(x)).getArg(lower_or_upper).getName())+
						   " <= "+
						   ddctv.shadows.get(array.getArg(0).getName()).getArg(i).getArg(lower_or_upper).getInt()+
						   " by yourself"+(char)27+"[0m");
			    }
			}
		    }
		}
		break;
	    default:
		System.out.println("  "+(char)27+"[1;34m"+"fatal error: "+loopvar.Opcode()+(char)27+"[0m");
		if (!(env.getStencilTypecheckallFlag())) {
		    System.exit(63);
		}
		//		System.out.println("3 "+loopvar);
		return false;
	    }
	}
	return true;
    }

    private class DDirectives {
	public XobjList nodes;
	public final HashMap<String,XobjList> template;
	public final HashMap<String,XobjList> distribute;
	public final HashMap<String,XobjList> shadows;
	public final HashMap<String,String> aligns_simple;
	public final HashMap<Pair<String,Integer>,Pair<String,Integer>> aligns;
	DDirectives (XobjList nodes,
		     HashMap<String,XobjList> template,
		     HashMap<String,XobjList> distribute,
		     HashMap<String,XobjList> shadows,
		     HashMap<String,String> aligns_simple,
		     HashMap<Pair<String,Integer>,Pair<String,Integer>> aligns
		     ) {
	    this.nodes = nodes;
	    this.template = template;
	    this.distribute = distribute;
	    this.shadows = shadows;
	    this.aligns_simple = aligns_simple;
	    this.aligns = aligns;
	}
    }

    private class Pair<F,S> {
	public final F first;
	public final S second;
	Pair (F first, S second) {
	    this.first = first;
	    this.second = second;
	}
    }
}
