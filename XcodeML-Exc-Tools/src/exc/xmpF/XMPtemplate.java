 /*
  * $TSUKUBA_Release: $
  * $TSUKUBA_Copyright:
  *  $
  */

package exc.xmpF;

 import exc.block.*;
 import exc.object.*;
 import java.util.Vector;

 public class XMPtemplate extends XMPobject {
   // defined in xmp_constant.h
   public final static int DUPLICATION	= 100;
   public final static int BLOCK		= 101;
   public final static int CYCLIC	= 102;
   public final static int GENBLOCK	= 103;

   private boolean		isFixed;
   private boolean		isDistributed;
   private XMPnodes		ontoNodes;

   private Vector<XMPdimInfo>       scripts;

   // null constructor
   public XMPtemplate() {
     super(XMPobject.TEMPLATE);

     isFixed = false;
     isDistributed = false;
     ontoNodes = null;
     scripts = new Vector<XMPdimInfo>();
   }

   public String toString(){
     String s = "{Template("+_name+"):";
     if(isFixed) s += "fixed:";
     if(isDistributed) s += "distributed:";
     if(ontoNodes != null){
       s += "on_ref="+ontoNodes+":";
     }
     s += scripts;
     s +="}";
     return s;
   }

   public void setIsFixed() {
     isFixed = true;
   }

   public boolean isFixed() {
     return isFixed;
   }

   public void setIsDistributed() {
     isDistributed = true;
   }

   public boolean isDistributed() {
     return isDistributed;
   }

   public void setOntoNodes(XMPnodes nodes) {
     ontoNodes = nodes;
   }

   public XMPnodes getOntoNodes() {
     return ontoNodes;
   }

   public void setDistMannerAt(int index, int distManner, Xobject distArg) {
     scripts.elementAt(index).setDistManner(distManner,distArg);
   }

   public int getDistMannerAt(int index){
     if (!isDistributed) {
       XMP.error("template " + getName() + " is not distributed");
       return 0;
     }
     return scripts.elementAt(index).getDistManner();
   }

   public String getDistMannerStringAt(int index){
     if (!isDistributed) {
       XMP.error("template " + getName() + " is not distributed");
       return null;
     }
     return(distMannerName(getDistMannerAt(index)));
   }

   public static String distMannerName(int dist){
     switch (dist) {
       case DUPLICATION:
	 return new String("DUPLICATION");
       case BLOCK:
	 return new String("BLOCK");
       case CYCLIC:
	 return new String("CYCLIC");
       default:
	 return "???";
     }
   }

   public Xobject getUpperAt(int index) {
     return scripts.elementAt(index).getUpper();
   }

   public Xobject getLowerAt(int index) {
     return scripts.elementAt(index).getLower();
   }

   public static String getDistMannerString(int manner) {
     switch (manner) {
       case DUPLICATION:
	 return new String("DUPLICATION");
       case BLOCK:
	 return new String("BLOCK");
       case CYCLIC:
	 return new String("CYCLIC");
       default:
	 XMP.fatal("getDistMannerString: unknown distribute manner");
	 return null;
     }
   }

   /* 
    * analyze and handle template pragma 
    *  templateDecl = ( list_of_name list_of_dimensions)
    *    list_of_demsions = ( { () | (size () ()) | (lower uppper ())}* )
    */
   public static void analyzeTemplate(Xobject name, Xobject templateDecl, 
				      XMPenv env, PragmaBlock pb) {
     XMPtemplate tempObject = new XMPtemplate();
     tempObject.parsePragma(name,templateDecl,env,pb);
     env.declXMPobject(tempObject,pb);
     if(XMP.debugFlag){
       System.out.println("tempObject="+tempObject);
     }
   }

   void parsePragma(Xobject name, Xobject decl, XMPenv env, PragmaBlock pb) {

     // check name collision
     _name = name.getString();
     if(env.findXMPobject(_name,pb) != null){
       XMP.error("XMP object '"+_name+"' is already declared");
       return;
     }

     // declare template desciptor
     _descId =  env.declObjectId(XMP.DESC_PREFIX_ + _name, pb);

     // declare template object
     _dim = 0;
     boolean templateIsFixed = true;
     for (XobjArgs i = decl.getArgs(); i != null; i = i.nextArgs()){
       XMPdimInfo info = XMPdimInfo.parseDecl(i.getArg());
       if(info.isStar()) templateIsFixed = false;
       scripts.add(info);
       _dim++;
     }

     if (templateIsFixed)  setIsFixed();

     if (_dim > XMP.MAX_DIM) {
       XMP.error("template dimension should be less than " + (XMP.MAX_DIM + 1));
       return;
     }
   }

   /* 
    * Translate distribute directive 
    *   distDecl = (list_of_names list_of_dimsions_dist nodes_name)
    */
   public static void analyzeDistribute(Xobject templ, Xobject distArgs,
					  Xobject nodes, XMPenv env,
					  PragmaBlock pb) {
     // get template object
     String templateName = templ.getString();
     XMPtemplate templateObject = null;

     templateObject = env.getXMPtemplate(templateName,pb);

     if (templateObject == null) {
       XMP.error("template '" + templateName  + "' is not declared");
     }

     if (!templateObject.isFixed()) {
       XMP.error("template '" + templateName + "' is not fixed");
     }

     if (templateObject.isDistributed()) {
       XMP.error("template '" + templateName +  "' is already distributed");
     }

     // get nodes object
     String nodesName = nodes.getString();
     XMPnodes nodesObject = nodesObject = env.findXMPnodes(nodesName, pb);
     if (nodesObject == null) {
       XMP.error("nodes '" + nodesName + "' is not declared");
     }

     // set onto Nodes.
     templateObject.setOntoNodes(nodesObject);

     int templateDim = templateObject.getDim();
     int templateDimIdx = 0;
     int nodesDim = nodesObject.getDim();
     int nodesDimIdx = 0;
     Xobject distArg = null;

     // distDecl.getArg(1) = the_list_of_dimension 
     for (XobjArgs i = distArgs.getArgs();  i != null; i = i.nextArgs()) {
       if (templateDimIdx >= templateDim) {
	 XMP.error("wrong template dimension indicated, too many");
	 break;
       }
       // ({block|cyclic|genblock) arg) 
       int distManner = XMPtemplate.DUPLICATION;
       if(i.getArg() == null)
	 distManner = XMPtemplate.DUPLICATION;
       else if(i.getArg().isIntConstant())
	 distManner = i.getArg().getInt();
       else {
	 String dist_fmt = i.getArg().getArg(0).getString();
	 if(dist_fmt.equalsIgnoreCase("BLOCK"))
	   distManner = XMPtemplate.BLOCK;
	 else if(dist_fmt.equalsIgnoreCase("CYCLIC"))
	   distManner = XMPtemplate.CYCLIC;
	 else if(dist_fmt.equalsIgnoreCase("GENBLOCK"))
	   distManner = XMPtemplate.GENBLOCK;
	 else {
	   XMP.fatal("unknown distribution format,"+dist_fmt);
	 }
	 distArg = i.getArg().getArg(1);
       }

       if(distManner != XMPtemplate.DUPLICATION){
	 if (nodesDimIdx >= nodesDim) {
	   XMP.error("the number of <dist-format> (except '*') should be the same with the nodes dimension");
	   return;
	 }
	 nodesDimIdx++;
       }

       templateObject.setDistMannerAt(templateDimIdx,distManner,distArg);

       templateDimIdx++;
     }

     // check nodes, template dimension
     if (nodesDimIdx != nodesDim) {
       XMP.error("the number of <dist-format> (except '*') should be the same with the nodes dimension");
     }

     if (templateDimIdx != templateDim) {
       XMP.error("wrong template dimension indicated, too few");
     }

     // set distributed
     templateObject.setIsDistributed();
     if(XMP.debugFlag) System.out.println("distribute="+templateObject);
   }

   /* rewrite for template directive:
    *  ! _xmpf_template_alloc__(t_desc,#dim)
    *  ! _xmpf_template_dim_info__(t_desc,i_dim,lower_b,upper_b,
    *                                     dist_manner,n_idx,chunk)
    *  !  xmpf_template_init__(t_desc,n_desc)
    */
   public void buildConstructor(BlockList body, XMPenv env){
     Ident f = env.declExternIdent(XMP.template_alloc_f,Xtype.FsubroutineType);
     Xobject args = Xcons.List(_descId.Ref(),Xcons.IntConstant(_dim),
			       Xcons.IntConstant(1)); // fixed only
     body.add(f.callSubroutine(args));
     
     /* template size */
     f = env.declExternIdent(XMP.template_dim_info_f,Xtype.FsubroutineType);
     for(int i = 0; i < _dim; i++){
       XMPdimInfo info = scripts.elementAt(i);
       Xobject dist_arg = info.getDistArg();

       if(dist_arg == null) dist_arg = Xcons.IntConstant(0);
       args = Xcons.List(_descId.Ref(),Xcons.IntConstant(i),
			 info.getLower(),info.getUpper(),
			 Xcons.IntConstant(info.getDistManner()),
			 dist_arg);
       body.add(f.callSubroutine(args));
     }

     /* init */
     f = env.declExternIdent(XMP.template_init_f,Xtype.FsubroutineType);
     body.add(f.callSubroutine(Xcons.List(_descId.Ref(),
					ontoNodes.getDescId().Ref())));
   }

   public void buildDestructor(BlockList body, XMPenv env){
     Ident f = env.declExternIdent(XMP.template_dealloc_f,Xtype.
				   FsubroutineType);
     Xobject args = Xcons.List(_descId.Ref());
     body.add(f.callSubroutine(args));
   }
}
