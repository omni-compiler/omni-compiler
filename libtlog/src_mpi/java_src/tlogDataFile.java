// $Id: tlogDataFile.java,v 1.1.1.1 2005/06/20 09:56:18 msato Exp $
// $RWC_Release$
// $RWC_Copyright$

import java.io.*;
import java.util.Vector;
import java.util.Stack;

public class tlogDataFile {
  static final int MAX_PE = 1024;
  static final int TLOG_BLOCK_SIZE = 1024;
  int npe;
  Vector logData[];
  double min_time, max_time;
  int dispHint[];
  static final int MAX_DATA_BLOCK = 20000;

  public boolean Input(String filename){
    int i;
    min_time = Double.MAX_VALUE;
    max_time = Double.MIN_VALUE;
    logData = new Vector [MAX_PE];
    dispHint = new int [MAX_PE];
    
    for(i = 0; i < MAX_PE; i++) logData[i] = new Vector();
    byte [] block = new byte[TLOG_BLOCK_SIZE];
    FileInputStream inFile = null;
    int nbytes = 0;

    try {
      inFile = new FileInputStream(filename);
    } catch(FileNotFoundException e){
      System.err.println("cannot open logfile '"+filename+"'");
      System.exit(1);
    }
    if(inFile == null) return false;

    int n_blk = 0;
    try {
      while(true){
	nbytes = inFile.read(block);
	n_blk++;
	if(n_blk > MAX_DATA_BLOCK){
	  System.err.println("too big log file, truncated (Max Num. of Blocks ="+
			     MAX_DATA_BLOCK+")");
	  break;
	}
	if(nbytes == -1) break;
	if(nbytes != TLOG_BLOCK_SIZE){
	  System.err.println("bad log file");
	  System.exit(1);
	}
	DataInputStream in = 
	  new DataInputStream(new ByteArrayInputStream(block));
      
	try {
	  while(true){
	      // byte type = in.readByte();
	      // int id = (int)(in.readByte());
	      // int arg1 = (int)(in.readShort());
	       int id = (int)(in.readShort());
	       byte type = in.readByte();
	       int arg1 = (int)(in.readByte());

	    int arg2 = in.readInt();
	    double time = in.readDouble();

	    if(type == 0) continue; // skip TLOG_UNDEF 

	    if(min_time > time) min_time = time;
	    if(max_time < time) max_time = time;
	    tlogData d = new tlogData(type,id,arg1,arg2,time);
	    /* System.out.println("data="+d); */
	    if(id >= 0 && id < MAX_PE)
	      logData[id].addElement(d);
	    else {
	      System.err.println("warning: bad data was found, id("+
			       id+") out of range, ingored");
	    }
	  }
	} catch(EOFException e){ }
      }
    } catch(IOException e){
      System.err.println("bad log data file");
      System.exit(1);
    } 
    for(i = 0; i < MAX_PE; i++)
      if(logData[i].size() != 0) npe = i+1;
    // System.out.println("npe="+npe);

    for(i = 0; i < npe; i++)  tlogCheckNested(logData[i]);
    return true;
  }

  void tlogCheckNested(Vector logs){
    Stack nested_events = new Stack();
    for(int i = 0; i < logs.size(); i++){
      tlogData d = (tlogData) logs.elementAt(i);
      if(!nested_events.empty())
	d.setNested((tlogData)nested_events.peek());
      if(tlogData.type_kind[d.type] == tlogData.TYPE_OUT){
	nested_events.pop();  // pop up IN event.
      }
      if(tlogData.type_kind[d.type] == tlogData.TYPE_IN){
	nested_events.push(d);
      }
    }
  }
}





